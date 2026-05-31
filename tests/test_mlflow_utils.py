from scaleforge.mlflow_utils import build_run_name


def test_build_run_name_from_experiment():
    config = {
        "experiment": {
            "run_name": "custom_run"
        },
        "data": {
            "streaming": True
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "max_steps": 60,
        },
    }

    assert build_run_name(config) == "custom_run"


def test_build_run_name_default():
    config = {
        "data": {
            "streaming": True
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "max_steps": 60,
        },
    }

    assert build_run_name(config) == "streaming_bs8_lr0.001_steps60"