import argparse

from scaleforge.config import load_config
from scaleforge.data import ShardedDataset
from scaleforge.streaming_data import StreamingShardedDataset
from scaleforge.distributed import cleanup_distributed, setup_distributed
from scaleforge.model import TinyTransformer
from scaleforge.trainer import Trainer
from scaleforge.deepspeed_trainer import DeepSpeedTrainer


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank passed by DeepSpeed or torch distributed launch.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    config = load_config(args.config)

    setup_distributed(
        backend=config["distributed"]["backend"]
    )

    try:
        if config["data"].get("streaming", False):
            dataset = StreamingShardedDataset(
                config["data"]["data_dir"]
            )
        else:
            dataset = ShardedDataset(
                config["data"]["data_dir"]
            )

        model = TinyTransformer(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_classes=config["model"]["num_classes"],
        )

        if config.get("deepspeed", {}).get("enabled", False):
            trainer = DeepSpeedTrainer(
                model=model,
                dataset=dataset,
                config=config,
            )
        else:
            trainer = Trainer(
                model=model,
                dataset=dataset,
                config=config,
                resume_path=args.resume,
            )

        trainer.train()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()