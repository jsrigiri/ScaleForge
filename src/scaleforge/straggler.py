from pathlib import Path

import pandas as pd


def detect_stragglers(
    metrics_dir: str,
    threshold: float = 1.5,
):
    metrics_path = Path(metrics_dir)

    rank_files = sorted(
        metrics_path.glob(
            "*_rank_*_metrics.csv"
        )
    )

    averages = {}

    for file in rank_files:

        df = pd.read_csv(file)

        parts = file.stem.split("_")
        rank = int(parts[2])

        averages[rank] = (
            df["step_time"]
            .mean()
        )

    if not averages:
        return None, {}

    overall_mean = (
        sum(averages.values())
        / len(averages)
    )

    stragglers = []

    for rank, value in averages.items():

        if value > (
            threshold * overall_mean
        ):
            stragglers.append(rank)

    return stragglers, averages