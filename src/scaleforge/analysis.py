import pandas as pd


def find_bottleneck(metrics_file: str):

    df = pd.read_csv(metrics_file)

    timing_columns = [
        "data_load_time",
        "forward_time",
        "backward_time",
        "optimizer_time",
        "checkpoint_time",
    ]

    available_columns = [
        col
        for col in timing_columns
        if col in df.columns
    ]

    averages = (
        df[available_columns]
        .mean()
        .sort_values(ascending=False)
    )

    bottleneck = averages.index[0]

    return bottleneck, averages