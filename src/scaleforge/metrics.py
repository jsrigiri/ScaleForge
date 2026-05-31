import csv
from pathlib import Path


class MetricsLogger:

    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = None

    def log(self, row: dict):
        current_fields = list(row.keys())

        write_header = not self.output_file.exists()

        if self.output_file.exists():
            with self.output_file.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                existing_fields = first_line.split(",") if first_line else []

            if existing_fields != current_fields:
                self.output_file.unlink()
                write_header = True

        with self.output_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=current_fields,
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)