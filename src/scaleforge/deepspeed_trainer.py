import time

import deepspeed
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scaleforge.checkpoint import CheckpointManager
from scaleforge.distributed import is_main_process
from scaleforge.metrics import MetricsLogger
from scaleforge.mlflow_utils import finish_mlflow, log_config, setup_mlflow
from scaleforge.profiler import TrainingProfiler
from scaleforge.rank_metrics import RankMetricsLogger


class DeepSpeedTrainer:

    def __init__(
        self,
        model,
        dataset,
        config,
    ):
        self.config = config
        self.rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
        self.world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        self.training_strategy = "deepspeed_zero"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.profiler = TrainingProfiler()

        self.metrics = None
        self.checkpoints = None

        if is_main_process():
            setup_mlflow(
                experiment_name=config["project"]["name"],
                config=config,
            )
            log_config(config)

            metrics_file = f"{config['paths']['metrics_dir']}/train_metrics.csv"
            self.metrics = MetricsLogger(metrics_file)

            self.checkpoints = CheckpointManager(
                config["paths"]["checkpoint_dir"]
            )

        self.rank_metrics = RankMetricsLogger(
            metrics_dir=config["paths"]["metrics_dir"],
            rank=self.rank,
            strategy=self.training_strategy,
        )

        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config["deepspeed"]["config_path"],
        )

        self.global_step = 0

    def train(self):
        max_steps = self.config["training"]["max_steps"]
        checkpoint_every_steps = self.config["training"]["checkpoint_every_steps"]

        self.model_engine.train()

        try:
            while self.global_step < max_steps:
                data_start_time = time.time()

                for tokens, labels in self.loader:
                    data_load_time = time.time() - data_start_time
                    step_start_time = time.time()

                    tokens = tokens.to(self.model_engine.device)
                    labels = labels.to(self.model_engine.device)

                    forward_start_time = time.time()
                    logits = self.model_engine(tokens)
                    loss = self.criterion(logits, labels)
                    forward_time = time.time() - forward_start_time

                    backward_start_time = time.time()
                    self.model_engine.backward(loss)
                    backward_time = time.time() - backward_start_time

                    optimizer_start_time = time.time()
                    self.model_engine.step()
                    optimizer_time = time.time() - optimizer_start_time

                    step_time = time.time() - step_start_time
                    self.global_step += 1

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

                    checkpoint_time = 0.0

                    if is_main_process():
                        if self.global_step % checkpoint_every_steps == 0:
                            checkpoint_start = time.time()
                            self.model_engine.save_checkpoint(
                                self.config["paths"]["checkpoint_dir"],
                                tag=f"ds_step_{self.global_step}",
                            )
                            checkpoint_time = time.time() - checkpoint_start

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

                        mlflow.log_metric("loss", float(loss.item()), step=self.global_step)
                        mlflow.log_metric("tokens_per_sec", profile["tokens_per_sec"], step=self.global_step)
                        mlflow.log_metric("step_time", step_time, step=self.global_step)

                        print(
                            f"step={self.global_step} "
                            f"rank={self.rank} "
                            f"world_size={self.world_size} "
                            f"training_strategy={self.training_strategy} "
                            f"loss={loss.item():.4f} "
                            f"tokens/sec={profile['tokens_per_sec']:.0f}"
                        )

                    if self.global_step >= max_steps:
                        break

                    data_start_time = time.time()

        finally:
            if is_main_process():
                finish_mlflow()