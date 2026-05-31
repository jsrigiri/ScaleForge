import mlflow


def build_run_name(config: dict) -> str:
    if "experiment" in config and "run_name" in config["experiment"]:
        return config["experiment"]["run_name"]

    training_mode = (
        "streaming"
        if config["data"].get("streaming", False)
        else "inmemory"
    )

    return (
        f"{training_mode}"
        f"_bs{config['training']['batch_size']}"
        f"_lr{config['training']['learning_rate']}"
        f"_steps{config['training']['max_steps']}"
    )


def setup_mlflow(
    experiment_name: str,
    config: dict,
):
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        experiment_name
    )

    run_name = build_run_name(config)

    mlflow.start_run(
        run_name=run_name
    )

    mlflow.set_tags(
        {
            "project": config["project"]["name"],
            "training_mode": (
                "streaming"
                if config["data"].get("streaming", False)
                else "in_memory"
            ),
            "training_strategy": (
                "fsdp"
                if config.get("fsdp", {}).get("enabled", False)
                else "single_or_ddp"
            ),
            "batch_size": str(config["training"]["batch_size"]),
            "distributed_backend": config["distributed"]["backend"],
        }
    )


def log_config(config: dict):
    def flatten(prefix, obj):
        items = {}

        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k

            if isinstance(v, dict):
                items.update(flatten(key, v))
            else:
                items[key] = v

        return items

    mlflow.log_params(
        flatten("", config)
    )


def finish_mlflow():
    if mlflow.active_run() is not None:
        mlflow.end_run()