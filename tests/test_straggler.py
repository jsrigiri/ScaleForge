import pandas as pd

from scaleforge.straggler import detect_stragglers


def test_detect_straggler(tmp_path):
    rank0 = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "rank": [0, 0, 0],
            "step_time": [0.1, 0.1, 0.1],
        }
    )

    rank1 = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "rank": [1, 1, 1],
            "step_time": [0.5, 0.5, 0.5],
        }
    )

    rank0.to_csv(
        tmp_path / "ddp_rank_0_metrics.csv",
        index=False,
    )

    rank1.to_csv(
        tmp_path / "ddp_rank_1_metrics.csv",
        index=False,
    )

    stragglers, averages = detect_stragglers(
        str(tmp_path),
        threshold=1.2,
    )

    assert stragglers == [1]
    assert averages[1] > averages[0]