"""The composition root.

Owns: wiring, the main loop, lifecycle and shutdown. The only module that knows
all the others.
Does NOT own: any domain logic. Every decision here is about *when* things
happen, never about what an angle means or whether it can be trusted.

The loop is sequential by design. VIDEO-mode tracking assumes a single
contiguous stream, and splitting frames across workers breaks it - knee jump
rose from 5.31 to 9.32 degrees in testing (FINDINGS.md F36). Only capture runs
on its own thread, so a slow consumer gets a fresh frame rather than a backlog.

Phase 3 scope: capture -> inference -> skeleton -> timing. Angles, validity and
the measurement panel arrive in phases 4 to 8. Proving the frame rate before
anything is layered on top is why this exists at this point in the build.
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from ..biomechanics.angles import compute_angles
from ..biomechanics.frame import anterior_cues
from ..capture.camera import CameraSource
from ..capture.source import FrameSource
from ..capture.video import VideoSource
from ..config import Config
from ..errors import CaptureError, SourceExhaustedError
from ..inference.pose import PoseEstimator
from ..metrics.timing import PipelineMetrics
from ..types import MeasurementName, PoseResult, Side
from ..ui.overlay import (
    draw_banner,
    draw_health,
    draw_measurements,
    draw_orientation,
    draw_skeleton,
)
from ..validity.orientation import OrientationState, PlaneGate, yaw_degrees

log = logging.getLogger(__name__)

WINDOW = "biomech"
_QUIT_KEYS = (ord("q"), 27)  # q or Escape


def open_source(cfg: Config, args: argparse.Namespace) -> FrameSource:
    """Build the frame source the arguments select."""
    if args.source == "camera":
        return CameraSource(cfg.capture)
    if args.source == "video":
        return VideoSource(args.path)
    raise CaptureError(
        f"Source {args.source!r} is not available yet. "
        "Landmark replay arrives with the recording module."
    )


def run(cfg: Config, args: argparse.Namespace) -> int:
    """Run the pipeline until the source ends or the user stops it."""
    source = open_source(cfg, args)
    estimator = PoseEstimator(cfg.model)
    metrics = PipelineMetrics(cfg.metrics)
    gate = PlaneGate(cfg.orientation)
    show_ui = not args.no_ui

    if show_ui:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, cfg.ui.window_width, cfg.ui.window_height)

    deadline = time.perf_counter() + args.duration if args.duration else None
    log.info("running: %s -> %s", source.description, estimator.description)

    try:
        _loop(cfg, source, estimator, metrics, gate, show_ui, deadline)
    except SourceExhaustedError as end:
        log.info("%s", end)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        estimator.close()
        source.close()
        if show_ui:
            cv2.destroyAllWindows()
        _report(metrics, source)
    return 0


def _loop(
    cfg: Config,
    source: FrameSource,
    estimator: PoseEstimator,
    metrics: PipelineMetrics,
    gate: PlaneGate,
    show_ui: bool,
    deadline: float | None,
) -> None:
    """Take the newest frame, infer, draw, measure. Repeat."""
    while deadline is None or time.perf_counter() < deadline:
        frame = source.next_frame()
        if frame is None:
            # A live camera between frames. Yielding keeps this loop from
            # spinning a core while waiting on hardware.
            time.sleep(0.001)
            continue

        result = estimator.estimate(frame)
        metrics.inference_ms.add(result.inference_ms)
        if not result.has_person:
            metrics.frames_without_person += 1

        started = time.perf_counter()
        yaw = yaw_degrees(result.landmarks.world_xyz) if result.landmarks else float("nan")
        orientation = gate.update(yaw)
        angles = _measure(result)
        metrics.biomech_ms.add((time.perf_counter() - started) * 1000.0)

        if show_ui:
            started = time.perf_counter()
            canvas = _render(cfg, frame.image, result, angles, orientation, metrics, source)
            cv2.imshow(WINDOW, canvas)
            metrics.render_ms.add((time.perf_counter() - started) * 1000.0)
            if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                raise KeyboardInterrupt

        metrics.source_drops = getattr(source, "drops", 0)
        metrics.frame_displayed(frame.capture_ts)


def _measure(result: PoseResult) -> dict[tuple[MeasurementName, Side], float]:
    """Compute the twelve angles, or an empty map when nobody is in frame.

    The anterior direction is taken as unanimous agreement of the three cues,
    and None otherwise. A guessed sign turns flexion into extension, so no
    guess is made. The temporal hold that makes this robust arrives with the
    validity layer.
    """
    if result.landmarks is None:
        return {}
    cues = anterior_cues(result.landmarks.image_xy)
    unanimous = cues[0] if cues[0] == cues[1] == cues[2] else None
    return compute_angles(result.landmarks, unanimous)


def _render(
    cfg: Config,
    image: np.ndarray,
    result: PoseResult,
    angles: dict[tuple[MeasurementName, Side], float],
    orientation: OrientationState,
    metrics: PipelineMetrics,
    source: FrameSource,
) -> np.ndarray:
    """Compose one output frame. Pure drawing - no decisions."""
    canvas = cv2.resize(image, (cfg.ui.window_width, cfg.ui.window_height))
    src_h, src_w = image.shape[:2]
    scale = (cfg.ui.window_width / src_w, cfg.ui.window_height / src_h)

    landmarks = result.landmarks
    if landmarks is not None and cfg.ui.draw_skeleton:
        draw_skeleton(canvas, landmarks, scale, cfg.validity.min_visibility)

    draw_measurements(canvas, angles, orientation)
    draw_orientation(canvas, orientation)
    if cfg.ui.show_health_panel:
        draw_health(canvas, metrics.snapshot(), source.description)

    if not source.is_healthy:
        draw_banner(canvas, "camera stopped delivering frames - waiting")
    elif landmarks is None:
        draw_banner(canvas, "no person detected")
    return canvas


def _report(metrics: PipelineMetrics, source: FrameSource) -> None:
    """Final timings, so a headless run still reports its result."""
    s = metrics.snapshot()
    log.info("--- pipeline ---")
    log.info("  frames processed : %6.0f  (%.0f without a person)",
             s["frames_processed"], s["frames_without_person"])
    log.info("  displayed FPS    : %6.1f", s["displayed_fps"])
    log.info("  inference  p50/p95: %6.1f / %.1f ms",
             s["inference_p50_ms"], s["inference_p95_ms"])
    log.info("  render     p50/p95: %6.1f / %.1f ms",
             s["render_p50_ms"], s["render_p95_ms"])
    log.info("  end-to-end p50/p95: %6.1f / %.1f ms",
             s["end_to_end_p50_ms"], s["end_to_end_p95_ms"])
    log.info("  source drops     : %6.0f (%.1f%%)",
             s["source_drops"], 100.0 * getattr(source, "drop_rate", 0.0))
