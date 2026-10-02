"""Latency and rate measurement.

Owns: rolling percentiles per stage, displayed frame rate, drop accounting.
Does NOT own: display, or the JSON report (see report.py).

Five numbers matter here and three of them are rates, which is where most
benchmarks go wrong:

  source FPS         what the source can deliver     (camera: 30, file: unbounded)
  throughput         what the pipeline can process   (measured by replay)
  displayed FPS      results actually drawn          (live: min of the two)
  inference latency  the model call alone
  end-to-end latency capture_ts to the moment it is drawn

Reporting `1/displayed_FPS` as latency would give 33 ms against a true ~25 ms,
and would be wrong in both directions. They are different quantities.

p95 is tracked alongside p50 because dropped-frame stalls live in the tail,
and the tail is what a user notices.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..config import MetricsConfig


class RollingStats:
    """Percentiles over the last N samples.

    A rolling window rather than a running total: the interesting question is
    "how is it behaving now", not "how has it behaved on average since start".
    A stall five minutes ago should not still be moving the number.
    """

    __slots__ = ("_samples",)

    def __init__(self, window: int) -> None:
        self._samples: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._samples.append(value)

    def percentile(self, pct: float) -> float:
        if not self._samples:
            return float("nan")
        ordered = sorted(self._samples)
        idx = min(int(len(ordered) * pct / 100.0), len(ordered) - 1)
        return ordered[idx]

    @property
    def mean(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else float("nan")

    @property
    def count(self) -> int:
        return len(self._samples)


class RateCounter:
    """Events per second, over a trailing time window.

    Time-based rather than count-based, so the answer stays meaningful when the
    pipeline stalls: a count-based window would report the rate from before the
    stall until enough new events arrived to flush it.
    """

    __slots__ = ("_window_s", "_events")

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_s = window_s
        self._events: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._events.append(now)
        cutoff = now - self._window_s
        while self._events and self._events[0] < cutoff:
            self._events.popleft()

    @property
    def rate(self) -> float:
        if len(self._events) < 2:
            return 0.0
        span = self._events[-1] - self._events[0]
        return (len(self._events) - 1) / span if span > 0 else 0.0


@dataclass
class PipelineMetrics:
    """Every timing the application reports.

    Stages are timed separately because the assignment asks for model latency
    and end-to-end latency as distinct numbers, and because knowing *which*
    stage grew is what makes a regression actionable.
    """

    cfg: MetricsConfig
    inference_ms: RollingStats = field(init=False)
    biomech_ms: RollingStats = field(init=False)
    render_ms: RollingStats = field(init=False)
    end_to_end_ms: RollingStats = field(init=False)
    displayed: RateCounter = field(init=False)
    frames_processed: int = 0
    frames_without_person: int = 0
    source_drops: int = 0

    def __post_init__(self) -> None:
        window = self.cfg.window_frames
        self.inference_ms = RollingStats(window)
        self.biomech_ms = RollingStats(window)
        self.render_ms = RollingStats(window)
        self.end_to_end_ms = RollingStats(window)
        self.displayed = RateCounter()

    def frame_displayed(self, capture_ts: float) -> None:
        """Record that one frame reached the screen.

        End-to-end latency is measured here, from the timestamp stamped at
        capture. That single field travelling with the frame is the whole
        reason this number can be honest.
        """
        self.end_to_end_ms.add((time.perf_counter() - capture_ts) * 1000.0)
        self.displayed.tick()
        self.frames_processed += 1

    def snapshot(self) -> dict[str, float]:
        """Current values, for the health panel and the JSON report."""
        return {
            "displayed_fps": self.displayed.rate,
            # Mean as well as median: the brief asks for average latency, and a
            # mean is the one statistic a long tail actually moves. Where the two
            # disagree, the gap between them IS the finding.
            "inference_mean_ms": self.inference_ms.mean,
            "inference_p50_ms": self.inference_ms.percentile(50),
            "inference_p95_ms": self.inference_ms.percentile(95),
            "biomech_p50_ms": self.biomech_ms.percentile(50),
            "render_p50_ms": self.render_ms.percentile(50),
            "render_p95_ms": self.render_ms.percentile(95),
            "end_to_end_mean_ms": self.end_to_end_ms.mean,
            "end_to_end_p50_ms": self.end_to_end_ms.percentile(50),
            "end_to_end_p95_ms": self.end_to_end_ms.percentile(95),
            "frames_processed": float(self.frames_processed),
            "frames_without_person": float(self.frames_without_person),
            "source_drops": float(self.source_drops),
        }
