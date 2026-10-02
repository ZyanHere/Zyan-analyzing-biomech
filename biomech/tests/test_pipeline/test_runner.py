"""The composition root, exercised end to end.

The runner was the one module with no test, and both bugs found in phase 9 were
of exactly the kind a test here catches: a collaborator built in `run` but read
in `_loop`, and an import removed by an autofix. Neither is visible to a unit
test of any single part, because the fault is in the wiring.

Replay makes this possible with no camera and no model, so it runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomech.cli import build_parser, config_from_args
from biomech.config import Config
from biomech.errors import CaptureError
from biomech.pipeline.runner import run
from biomech.recording.writer import SessionRecorder
from biomech.types import Frame, PoseResult

from ..synthetic import build, landmark_set


def _recording(path: Path, frames: int = 40) -> Path:
    """A short synthetic session, moving so the filter and gate both engage."""
    recorder = SessionRecorder(Config(), path, None)
    for seq in range(1, frames + 1):
        skeleton = build(elbow=20.0 + seq, knee=float(seq), side="L", body_yaw=90.0)
        image = np.zeros((480, 640, 3), np.uint8)
        recorder.write(
            Frame(image=image, capture_ts=seq / 30.0, seq=seq),
            PoseResult(landmark_set(skeleton), seq / 30.0, seq, 20.0),
        )
    recorder.close()
    return path


def _args(path: Path, *extra: str):
    """Parsed through the real CLI, so the test fails if the runner and the
    parser disagree about an argument's name."""
    return build_parser().parse_args(
        ["--source", "landmarks", "--path", str(path), "--no-ui", *extra]
    )


class TestRunToCompletion:
    def test_processes_every_frame_and_exits_cleanly(self, tmp_path: Path) -> None:
        args = _args(_recording(tmp_path / "s.jsonl"))
        assert run(config_from_args(args), args) == 0

    def test_a_finished_source_is_not_an_error(self, tmp_path: Path) -> None:
        """Reaching the end of a recording is the normal outcome, so the exit
        code must not suggest a fault."""
        args = _args(_recording(tmp_path / "s.jsonl", frames=3))
        assert run(config_from_args(args), args) == 0


class TestMetricsExport:
    def test_writes_the_run_it_actually_performed(self, tmp_path: Path) -> None:
        out = tmp_path / "metrics.json"
        args = _args(_recording(tmp_path / "s.jsonl", frames=25), "--metrics-out", str(out))
        run(config_from_args(args), args)

        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["metrics"]["frames_processed"] == 25.0
        assert report["metrics"]["source_drops"] == 0.0

    def test_replay_latency_is_not_labelled_end_to_end(self, tmp_path: Path) -> None:
        """A recording's timestamps are relative to when it was made. Measuring
        wall-clock latency against them gave 127,089,632 ms; the quantity
        reported for a replay is processing time, and must say so."""
        out = tmp_path / "metrics.json"
        args = _args(_recording(tmp_path / "s.jsonl"), "--metrics-out", str(out))
        run(config_from_args(args), args)

        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["latency_kind"] == "processing"
        assert report["metrics"]["end_to_end_p50_ms"] < 1000.0

    def test_no_metrics_file_is_written_unless_asked(self, tmp_path: Path) -> None:
        args = _args(_recording(tmp_path / "s.jsonl", frames=3))
        run(config_from_args(args), args)
        assert list(tmp_path.glob("*.json")) == []


class TestFailureHandling:
    def test_an_unreadable_source_fails_rather_than_running_empty(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        args = _args(empty)
        with pytest.raises(CaptureError):
            run(config_from_args(args), args)
