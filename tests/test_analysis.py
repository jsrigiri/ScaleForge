import pandas as pd

from scaleforge.analysis import find_bottleneck


def test_find_bottleneck(tmp_path):
    metrics_file = tmp_path / "train_metrics.csv"

    df = pd.DataFrame(
        {
            "data_load_time": [0.01, 0.02],
            "forward_time": [0.10, 0.20],
            "backward_time": [0.03, 0.04],
            "optimizer_time": [0.01, 0.01],
            "checkpoint_time": [0.00, 0.00],
        }
    )

    df.to_csv(metrics_file, index=False)

    bottleneck, averages = find_bottleneck(
        str(metrics_file)
    )

    assert bottleneck == "forward_time"
    assert averages["forward_time"] > averages["backward_time"]