import os
import subprocess
import sys
from pathlib import Path


def run_experiment(config_path: Path) -> None:
    print("=" * 80)
    print(f"Running experiment: {config_path}")
    print("=" * 80)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "--config",
            str(config_path),
        ],
        check=False,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Experiment failed: {config_path}"
        )


def main() -> None:
    experiment_dir = Path(
        "configs/experiments"
    )

    config_files = sorted(
        experiment_dir.glob("*.yaml")
    )

    if not config_files:
        raise FileNotFoundError(
            f"No experiment configs found in {experiment_dir}"
        )

    for config_path in config_files:
        run_experiment(config_path)

    print("=" * 80)
    print("All experiments completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()