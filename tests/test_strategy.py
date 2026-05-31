from scaleforge.strategy import get_training_strategy


def test_strategy_single():
    config = {
        "fsdp": {
            "enabled": False,
        }
    }

    assert get_training_strategy(config) == "single"


def test_strategy_fsdp():
    config = {
        "fsdp": {
            "enabled": True,
        }
    }

    assert get_training_strategy(config) == "fsdp"