from __future__ import annotations

from collections import deque
from statistics import median
from typing import Deque, Dict, Iterable


def percentile(values: Iterable[float], pct: float) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    index = int((len(items) - 1) * pct)
    return float(items[index])


class SlidingMetrics:
    def __init__(self, max_len: int = 5000) -> None:
        self.latencies: Deque[float] = deque(maxlen=max_len)
        self.fps_samples: Deque[float] = deque(maxlen=max_len)
        self.throughput_samples: Deque[float] = deque(maxlen=max_len)

    def add(self, latency_ms: float, fps: float, throughput_fps: float) -> None:
        self.latencies.append(latency_ms)
        self.fps_samples.append(fps)
        self.throughput_samples.append(throughput_fps)

    def summary(self) -> Dict[str, float]:
        return {
            "latency_p50_ms": median(self.latencies) if self.latencies else 0.0,
            "latency_p95_ms": percentile(self.latencies, 0.95),
            "fps_avg": sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0.0,
            "throughput_avg": sum(self.throughput_samples) / len(self.throughput_samples) if self.throughput_samples else 0.0,
        }
