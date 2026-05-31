import os

import torch.distributed as dist


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "gloo") -> None:
    if is_distributed():
        dist.init_process_group(backend=backend)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()