"""Inference benchmark: the model alone, on recorded frames.

Owns: measuring model latency and landmark quality in isolation.
Does NOT own: the pipeline, the camera, or anything downstream.

Runs on a video file rather than a camera so the numbers are reproducible and
so the model can be measured without a device attached. This is also how
throughput is established independently of a camera that caps at 30 FPS: the
pipeline's headroom is a property of the machine, not of the webcam.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..capture.video import VideoSource
from ..config import ModelConfig
from ..errors import SourceExhaustedError
from ..types import LANDMARK_COUNT
from .pose import PoseEstimator

log = logging.getLogger(__name__)


def benchmark_inference(path: Path, cfg: ModelConfig) -> dict[str, float]:
    """Replay a video through the model and report latency and detection quality."""
    source = VideoSource(path)
    estimator = PoseEstimator(cfg)

    latencies: list[float] = []
    detected = 0
    frames = 0
    mean_visibility: list[float] = []

    try:
        while True:
            try:
                frame = source.next_frame()
            except SourceExhaustedError:
                break
            if frame is None:
                continue

            frames += 1
            result = estimator.estimate(frame)
            latencies.append(result.inference_ms)

            if result.has_person:
                detected += 1
                marks = result.landmarks
                assert marks is not None  # narrowed by has_person
                if marks.image_xy.shape[0] != LANDMARK_COUNT:
                    raise AssertionError(
                        f"expected {LANDMARK_COUNT} landmarks, got {marks.image_xy.shape[0]}"
                    )
                mean_visibility.append(float(marks.visibility.mean()))
    finally:
        estimator.close()
        source.close()

    latencies.sort()
    result_dict = {
        "frames": float(frames),
        "detected_pct": 100.0 * detected / frames if frames else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "implied_fps": 1000.0 / _percentile(latencies, 50) if latencies else 0.0,
        "mean_visibility": float(np.mean(mean_visibility)) if mean_visibility else 0.0,
    }
    _report(path, cfg, result_dict)
    return result_dict


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(len(sorted_values) * pct / 100.0), len(sorted_values) - 1)
    return sorted_values[idx]


def _report(path: Path, cfg: ModelConfig, r: dict[str, float]) -> None:
    log.info("inference benchmark: %s, %s model, %s mode",
             path.name, cfg.variant, cfg.running_mode)
    log.info("  frames              : %6.0f", r["frames"])
    log.info("  person detected     : %5.0f%%", r["detected_pct"])
    log.info("  latency p50 / p95   : %6.1f / %.1f ms", r["latency_p50_ms"], r["latency_p95_ms"])
    log.info("  implied single-thread rate: %.0f FPS", r["implied_fps"])
    log.info("  mean visibility     : %6.2f", r["mean_visibility"])

    if r["implied_fps"] < 30.0:
        log.warning(
            "Model alone is below 30 FPS before any other work. VIDEO mode should give "
            "43-60 FPS on the reference machine (FINDINGS.md F36); check --running-mode.",
        )
