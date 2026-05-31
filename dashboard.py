from pathlib import Path

import pandas as pd
import streamlit as st

from scaleforge.analysis import find_bottleneck
from scaleforge.straggler import (
    detect_stragglers
)

st.set_page_config(
    page_title="ScaleForge Dashboard",
    layout="wide",
)

st.title("🚀 ScaleForge Dashboard")

metrics_file = Path(
    "metrics/train_metrics.csv"
)

if not metrics_file.exists():
    st.warning(
        "No metrics file found. Run training first."
    )
    st.stop()

df = pd.read_csv(metrics_file)

latest = df.iloc[-1]

stragglers, averages = detect_stragglers(
    "metrics"
)

st.subheader(
    "Distributed Training Health"
)

if stragglers:
    st.error(
        f"⚠ Straggler ranks detected: {stragglers}"
    )
else:
    st.success(
        "✓ No stragglers detected"
    )

if averages:

    st.subheader(
        "Average Rank Step Time"
    )

    rank_df = pd.DataFrame(
        {
            "rank": list(averages.keys()),
            "avg_step_time": list(
                averages.values()
            ),
        }
    )

    st.bar_chart(
        rank_df.set_index("rank"),
        x_label="Rank",
        y_label="Average Step Time (seconds)",
    )

bottleneck, averages = find_bottleneck(
    metrics_file
)

st.subheader("Average Timing Bottleneck")

st.info(
    f"Primary Bottleneck: {bottleneck}"
)

st.bar_chart(
    averages,
    x_label="Training Component",
    y_label="Average Seconds",
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Current Step",
    int(latest["step"])
)

col2.metric(
    "Current Loss",
    round(float(latest["loss"]), 4)
)

col3.metric(
    "Tokens/Sec",
    round(
        float(latest["tokens_per_sec"]),
        0,
    )
)

col4.metric(
    "CPU MB",
    round(
        float(latest["cpu_memory_mb"]),
        1,
    )
)

gpu_memory = (
    float(latest["gpu_memory_mb"])
    if "gpu_memory_mb" in latest
    else 0.0
)

col5.metric(
    "GPU MB",
    round(gpu_memory, 1)
)


st.subheader("Raw Metrics")

st.dataframe(df.tail(20))

col1, col2 = st.columns(2)

with col1:

    st.subheader("Loss")

    st.line_chart(
        df.set_index("step")["loss"],
        x_label="Training Step",
        y_label="Loss",
    )

with col2:

    st.subheader("Step Time")

    st.line_chart(
        df.set_index("step")["step_time"],
        x_label="Training Step",
        y_label="Seconds",
    )

timing_cols = [
    "data_load_time",
    "forward_time",
    "backward_time",
    "optimizer_time",
    "checkpoint_time",
]

available_timing_cols = [
    col for col in timing_cols if col in df.columns
]

if available_timing_cols:
    st.subheader("Training Step Timing Breakdown")

    st.line_chart(
        df.set_index("step")[available_timing_cols],
        x_label="Training Step",
        y_label="Seconds",
    )

st.subheader("Tokens / Second")

st.line_chart(
    df.set_index("step")["tokens_per_sec"],
    x_label="Training Step",
    y_label="Tokens / Second",
)

if "cpu_memory_mb" in df.columns:

    st.subheader("CPU Memory")

    st.line_chart(
        df.set_index("step")["cpu_memory_mb"],
        x_label="Training Step",
        y_label="Megabytes (MB)",
    )

if "gpu_memory_mb" in df.columns:

    st.subheader("GPU Memory")

    st.line_chart(
        df.set_index("step")["gpu_memory_mb"],
        x_label="Training Step",
        y_label="Megabytes (MB)",
    )