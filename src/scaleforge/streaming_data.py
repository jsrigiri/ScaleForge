import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from scaleforge.distributed import get_rank, get_world_size


class StreamingShardedDataset(IterableDataset):
    """
    Streams JSONL shards without loading all records into memory.

    Each distributed rank reads a different subset of shards.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        self.shard_files = sorted(
            self.data_dir.glob("shard_*.jsonl")
        )

        if not self.shard_files:
            raise ValueError(f"No shard files found in {data_dir}")

    def __iter__(self):
        rank = get_rank()
        world_size = get_world_size()

        assigned_shards = self.shard_files[rank::world_size]

        for shard_file in assigned_shards:
            with shard_file.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)

                    tokens = torch.tensor(
                        row["tokens"],
                        dtype=torch.long,
                    )

                    label = torch.tensor(
                        row["label"],
                        dtype=torch.long,
                    )

                    yield tokens, label