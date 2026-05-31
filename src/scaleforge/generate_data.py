import argparse
import json
import random
from pathlib import Path

from scaleforge.config import load_config


def generate_record(seq_len: int, vocab_size: int) -> dict:
    tokens = [random.randint(1, vocab_size - 1) for _ in range(seq_len)]

    label = int(sum(tokens) % 2 == 0)

    return {
        "tokens": tokens,
        "label": label,
    }


def generate_shards(config_path: str) -> None:
    config = load_config(config_path)

    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config["project"]["seed"])

    num_shards = data_cfg["num_shards"]
    records_per_shard = data_cfg["records_per_shard"]
    seq_len = data_cfg["seq_len"]
    vocab_size = data_cfg["vocab_size"]

    for shard_id in range(num_shards):
        shard_path = data_dir / f"shard_{shard_id:03d}.jsonl"

        with shard_path.open("w", encoding="utf-8") as f:
            for _ in range(records_per_shard):
                record = generate_record(seq_len, vocab_size)
                f.write(json.dumps(record) + "\n")

        print(f"Wrote {shard_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    generate_shards(args.config)


if __name__ == "__main__":
    main()