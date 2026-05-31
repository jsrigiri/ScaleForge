import pandas as pd

from scaleforge.metrics import MetricsLogger


def test_metrics_logger_writes_csv(tmp_path):
    output_file = tmp_path / "metrics.csv"

    logger = MetricsLogger(str(output_file))

    logger.log(
        {
            "step": 1,
            "loss": 0.5,
            "step_time": 0.1,
        }
    )

    df = pd.read_csv(output_file)

    assert len(df) == 1
    assert df.iloc[0]["step"] == 1
    assert df.iloc[0]["loss"] == 0.5