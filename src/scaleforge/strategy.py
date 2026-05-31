from scaleforge.distributed import is_distributed


def get_training_strategy(config: dict) -> str:
    if config.get("fsdp", {}).get("enabled", False):
        return "fsdp"

    if config.get("zero", {}).get("enabled", False):
        return "zero"

    if is_distributed():
        return "ddp"

    return "single"