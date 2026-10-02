"""The performance record.

A published number is only a claim if it carries the hardware and the settings
that produced it. These tests guard the two ways that goes wrong: omitting the
configuration, and labelling a replay's processing latency as end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

from biomech.config import Config, MetricsConfig
from biomech.metrics.report import build_report, hardware, write_report
from biomech.metrics.timing import PipelineMetrics


def _metrics() -> PipelineMetrics:
    m = PipelineMetrics(MetricsConfig())
    for value in (19.0, 20.0, 21.0):
        m.inference_ms.add(value)
        m.render_ms.add(2.8)
    m.frame_displayed(0.0)
    return m


class TestHardware:
    def test_names_the_machine_without_a_dependency(self) -> None:
        """Stdlib only: describing the host must not be harder to reproduce
        than the measurement itself."""
        hw = hardware()
        assert hw["platform"]
        assert hw["python"]
        assert isinstance(hw["cpu_count"], int)


class TestBuildReport:
    def test_carries_the_configuration_that_produced_it(self) -> None:
        report = build_report(
            Config(), _metrics(), source="video x.mp4", label="run", latency_kind="end_to_end"
        )
        config = report["config"]
        assert isinstance(config, dict)
        assert config["validity"]["min_visibility"] == 0.55
        assert config["model"]["variant"] == "full"

    def test_latency_kind_is_explicit(self) -> None:
        """Replay has no capture instant, so its latency is a different
        quantity. A reader of the JSON has no other way to tell."""
        replay = build_report(
            Config(), _metrics(), source="landmarks s.jsonl", label="r",
            latency_kind="processing",
        )
        assert replay["latency_kind"] == "processing"

    def test_metrics_are_included_not_summarised(self) -> None:
        report = build_report(
            Config(), _metrics(), source="s", label="l", latency_kind="end_to_end"
        )
        metrics = report["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["inference_p50_ms"] == 20.0
        assert metrics["frames_processed"] == 1.0


class TestWriteReport:
    def test_writes_valid_json_and_creates_its_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "run.json"
        write_report(path, build_report(
            Config(), _metrics(), source="s", label="l", latency_kind="end_to_end"
        ))
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["schema"] == 1
        assert loaded["label"] == "l"
        assert loaded["recorded_at"].endswith("+00:00")
