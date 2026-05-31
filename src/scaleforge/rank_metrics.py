import csv
from pathlib import Path


class RankMetricsLogger:

    def __init__(
        self,
        metrics_dir: str,
        rank: int,
        strategy: str,
    ):
        self.output_file = (
            Path(metrics_dir)
            / f"{strategy}_rank_{rank}_metrics.csv"
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(self, row: dict):
        write_header = not self.output_file.exists()

        with self.output_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=row.keys(),
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)