import time
import os
import mlflow
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scaleforge.checkpoint import CheckpointManager
from scaleforge.distributed import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
from scaleforge.metrics import MetricsLogger
from scaleforge.profiler import TrainingProfiler
from scaleforge.mlflow_utils import (
    setup_mlflow,
    log_config,
    finish_mlflow,
)
from scaleforge.rank_metrics import RankMetricsLogger

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        StateDictType,
    )
except ImportError:
    FSDP = None
    FullStateDictConfig = None
    StateDictType = None

from scaleforge.strategy import get_training_strategy

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
except ImportError:
    ZeroRedundancyOptimizer = None


class Trainer:

    def __init__(
        self,
        model,
        dataset,
        config,
        resume_path: str | None = None,
    ):        
        self.config = config
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.training_strategy = get_training_strategy(config)

        if torch.cuda.is_available():
            local_rank = int(
                os.environ.get("LOCAL_RANK", "0")
            )

            device_count = torch.cuda.device_count()

            device_index = local_rank % device_count

            torch.cuda.set_device(device_index)

            self.device = torch.device(
                f"cuda:{device_index}"
            )
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        use_fsdp = (
            config.get("fsdp", {}).get("enabled", False)
        )

        if use_fsdp:
            if FSDP is None:
                raise RuntimeError(
                    "FSDP is not available in this PyTorch installation."
                )

            if not is_distributed():
                raise RuntimeError(
                    "FSDP requires distributed training. Use local_ddp.py or torchrun."
                )

            self.model = FSDP(
                self.model,
                device_id=self.device,
            )

        elif is_distributed():
            self.model = DistributedDataParallel(self.model)

        self.dataset = dataset

        self.sampler = None

        if is_distributed() and hasattr(dataset, "__len__"):
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )

        is_iterable_dataset = not hasattr(dataset, "__len__")

        self.loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(
                self.sampler is None
                and not is_iterable_dataset
            ),
            sampler=self.sampler,
        )

        use_zero = config.get("zero", {}).get("enabled", False)

        if use_zero:
            if ZeroRedundancyOptimizer is None:
                raise RuntimeError(
                    "ZeroRedundancyOptimizer is not available in this PyTorch installation."
                )

            if not is_distributed():
                raise RuntimeError(
                    "ZeRO optimizer requires distributed training. Use local_ddp.py."
                )

            self.optimizer = ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=config["training"]["learning_rate"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["training"]["learning_rate"],
            )

        self.criterion = nn.CrossEntropyLoss()

        self.metrics = None
        self.checkpoints = None

        if is_main_process():
            metrics_file = (
                f"{config['paths']['metrics_dir']}/train_metrics.csv"
            )
            self.metrics = MetricsLogger(metrics_file)

            self.checkpoints = CheckpointManager(
                config["paths"]["checkpoint_dir"]
            )

        self.global_step = 0

        if resume_path is not None:
            checkpoint_manager = CheckpointManager(
                config["paths"]["checkpoint_dir"]
            )

            raw_model = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )

            self.global_step = checkpoint_manager.load(
                path=resume_path,
                model=raw_model,
                optimizer=self.optimizer,
                device=self.device,
            )

        self.profiler = TrainingProfiler()

        self.rank_metrics = RankMetricsLogger(
            metrics_dir=config["paths"]["metrics_dir"],
            rank=self.rank,
            strategy=self.training_strategy,
        )

        if is_main_process():
            setup_mlflow(
                experiment_name=config["project"]["name"],
                config=config,
            )

            log_config(config)

    def save_checkpoint(self) -> None:
        use_fsdp = self.config.get("fsdp", {}).get("enabled", False)
        use_zero = self.config.get("zero", {}).get("enabled", False)

        if use_fsdp:
            if FSDP is None:
                raise RuntimeError("FSDP is not available.")

            save_policy = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                model_state = self.model.state_dict()

            optimizer_state = self.optimizer.state_dict()

            if is_main_process():
                self.checkpoints.save_state_dicts(
                    model_state=model_state,
                    optimizer_state=optimizer_state,
                    step=self.global_step,
                )

            return

        raw_model = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        if use_zero:
            if is_main_process():
                self.checkpoints.save_state_dicts(
                    model_state=raw_model.state_dict(),
                    optimizer_state={},
                    step=self.global_step,
                )

            return

        if is_main_process():
            self.checkpoints.save(
                model=raw_model,
                optimizer=self.optimizer,
                step=self.global_step,
            )

    def train(self):
        max_steps = self.config["training"]["max_steps"]
        checkpoint_every_steps = self.config["training"]["checkpoint_every_steps"]
        fail_after_steps = self.config["training"].get("fail_after_steps")

        self.model.train()

        while self.global_step < max_steps:

            if self.sampler is not None:
                self.sampler.set_epoch(self.global_step)

            data_start_time = time.time()

            for tokens, labels in self.loader:
                data_load_time = time.time() - data_start_time

                step_start_time = time.time()

                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                forward_start_time = time.time()
                logits = self.model(tokens)
                loss = self.criterion(logits, labels)
                forward_time = time.time() - forward_start_time

                backward_start_time = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                backward_time = time.time() - backward_start_time

                optimizer_start_time = time.time()
                self.optimizer.step()
                optimizer_time = time.time() - optimizer_start_time

                step_time = time.time() - step_start_time

                profile = self.profiler.snapshot(
                    batch_size=tokens.shape[0],
                    seq_len=tokens.shape[1],
                    step_time=step_time,
                )

                self.rank_metrics.log(
                    {
                        "step": self.global_step,
                        "rank": self.rank,
                        "training_strategy": self.training_strategy,
                        "step_time": step_time,
                        "tokens_per_sec": profile["tokens_per_sec"],
                    }
                )

                self.global_step += 1

                checkpoint_time = 0.0

                if is_main_process():
                    should_checkpoint = (
                        self.global_step % checkpoint_every_steps == 0
                    )

                    if should_checkpoint:
                        checkpoint_start_time = time.time()

                        raw_model = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )

                        self.save_checkpoint()

                        checkpoint_time = time.time() - checkpoint_start_time

                    self.metrics.log(
                        {
                            "step": self.global_step,
                            "rank": self.rank,
                            "world_size": self.world_size,
                            "training_strategy": self.training_strategy,
                            "loss": float(loss.item()),
                            "step_time": step_time,
                            "data_load_time": data_load_time,
                            "forward_time": forward_time,
                            "backward_time": backward_time,
                            "optimizer_time": optimizer_time,
                            "checkpoint_time": checkpoint_time,
                            "tokens_per_sec": profile["tokens_per_sec"],
                            "cpu_memory_mb": profile["cpu_memory_mb"],
                            "gpu_memory_mb": profile["gpu_memory_mb"],
                        }
                    )

                    mlflow.log_metric(
                        "loss",
                        float(loss.item()),
                        step=self.global_step,
                    )

                    mlflow.log_metric(
                        "tokens_per_sec",
                        profile["tokens_per_sec"],
                        step=self.global_step,
                    )

                    mlflow.log_metric(
                        "step_time",
                        step_time,
                        step=self.global_step,
                    )

                    print(
                        f"step={self.global_step} "
                        f"rank={self.rank} "
                        f"world_size={self.world_size} "
                        f"training_strategy={self.training_strategy} "
                        f"loss={loss.item():.4f} "
                        f"tokens/sec={profile['tokens_per_sec']:.0f}"
                    )

                if (
                    fail_after_steps is not None
                    and self.global_step >= fail_after_steps
                ):
                    raise RuntimeError(
                        f"Fault injection triggered at step {self.global_step}"
                    )

                if self.global_step >= max_steps:
                    break

                data_start_time = time.time()

        if is_main_process():
            finish_mlflow()