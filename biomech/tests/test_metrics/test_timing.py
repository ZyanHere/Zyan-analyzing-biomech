"""Timing measurement.

These tests exist because the easiest way to publish a misleading benchmark is
to conflate a rate with a latency, or to let a stall five minutes ago keep
moving the current number.
"""

from __future__ import annotations

import math
import time

from biomech.config import MetricsConfig
from biomech.metrics.timing import PipelineMetrics, RateCounter, RollingStats


class TestRollingStats:
    def test_empty_reports_nan_rather_than_zero(self) -> None:
        """Zero would read as 'instant'. NaN reads as 'no data', which is true."""
        stats = RollingStats(window=10)
        assert math.isnan(stats.percentile(50))
        assert math.isnan(stats.mean)

    def test_percentiles_over_a_known_distribution(self) -> None:
        stats = RollingStats(window=100)
        for value in range(1, 101):
            stats.add(float(value))
        assert stats.percentile(50) == 51.0
        assert stats.percentile(95) == 96.0

    def test_window_forgets_old_samples(self) -> None:
        """A stall from five minutes ago should not still move the number."""
        stats = RollingStats(window=3)
        for value in (100.0, 100.0, 100.0, 1.0, 1.0, 1.0):
            stats.add(value)
        assert stats.percentile(50) == 1.0
        assert stats.count == 3

    def test_p95_exceeds_p50_when_there_is_a_tail(self) -> None:
        """p95 is tracked because dropped-frame stalls live in the tail."""
        stats = RollingStats(window=100)
        for _ in range(95):
            stats.add(20.0)
        for _ in range(5):
            stats.add(200.0)
        assert stats.percentile(50) == 20.0
        assert stats.percentile(95) > 100.0


class TestRateCounter:
    def test_no_rate_before_two_events(self) -> None:
        counter = RateCounter()
        assert counter.rate == 0.0
        counter.tick()
        assert counter.rate == 0.0

    def test_measures_a_known_rate(self) -> None:
        counter = RateCounter(window_s=10.0)
        base = 1000.0
        for i in range(21):  # 20 intervals of 50 ms = 20 per second
            counter.tick(base + i * 0.05)
        assert 19.0 < counter.rate < 21.0

    def test_trailing_window_drops_stale_events(self) -> None:
        """Time-based rather than count-based, so a stall reads as a stall
        instead of reporting the rate from before it."""
        counter = RateCounter(window_s=1.0)
        base = 500.0
        for i in range(10):
            counter.tick(base + i * 0.01)
        counter.tick(base + 60.0)
        assert counter.rate == 0.0


class TestPipelineMetrics:
    def _metrics(self) -> PipelineMetrics:
        return PipelineMetrics(MetricsConfig(window_frames=50))

    def test_end_to_end_measures_from_capture_not_from_now(self) -> None:
        """This is the number that makes latency honest: it starts at the
        moment of capture, not when the pipeline got around to the frame."""
        metrics = self._metrics()
        captured_at = time.perf_counter() - 0.040  # 40 ms ago
        metrics.frame_displayed(captured_at)
        measured = metrics.end_to_end_ms.percentile(50)
        assert 35.0 < measured < 60.0

    def test_snapshot_exposes_every_reported_number(self) -> None:
        metrics = self._metrics()
        metrics.inference_ms.add(24.0)
        metrics.render_ms.add(3.0)
        metrics.frame_displayed(time.perf_counter())

        snap = metrics.snapshot()
        for key in (
            "displayed_fps", "inference_p50_ms", "inference_p95_ms",
            "render_p50_ms", "end_to_end_p50_ms", "end_to_end_p95_ms",
            "frames_processed", "source_drops",
        ):
            assert key in snap, f"snapshot is missing {key}"

    def test_frames_without_a_person_are_counted_separately(self) -> None:
        """Processed and detected are different numbers; a run that saw nobody
        should not look like a run that measured them."""
        metrics = self._metrics()
        metrics.frames_without_person += 3
        metrics.frame_displayed(time.perf_counter())
        snap = metrics.snapshot()
        assert snap["frames_processed"] == 1.0
        assert snap["frames_without_person"] == 3.0

    def test_latency_and_rate_are_not_derived_from_each_other(self) -> None:
        """1/FPS is not latency. A pipeline can display 30 FPS while each frame
        takes 25 ms, and reporting either as the other is wrong."""
        metrics = self._metrics()
        base = time.perf_counter()
        for i in range(10):
            metrics.inference_ms.add(25.0)
            metrics.frame_displayed(base - 0.025 + i * 0.001)
        snap = metrics.snapshot()
        assert snap["inference_p50_ms"] == 25.0
        assert snap["displayed_fps"] != 1000.0 / snap["inference_p50_ms"]
