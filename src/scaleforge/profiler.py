import psutil
import time

import torch


class TrainingProfiler:

    def __init__(self):
        self.process = psutil.Process()

    def snapshot(
        self,
        batch_size: int,
        seq_len: int,
        step_time: float,
    ) -> dict:

        tokens_processed = batch_size * seq_len

        tokens_per_sec = (
            tokens_processed / step_time
            if step_time > 0
            else 0
        )

        cpu_memory_mb = (
            self.process.memory_info().rss
            / 1024
            / 1024
        )

        gpu_memory_mb = 0.0

        if torch.cuda.is_available():
            gpu_memory_mb = (
                torch.cuda.memory_allocated()
                / 1024
                / 1024
            )

        return {
            "tokens_per_sec": tokens_per_sec,
            "cpu_memory_mb": cpu_memory_mb,
            "gpu_memory_mb": gpu_memory_mb,
        }