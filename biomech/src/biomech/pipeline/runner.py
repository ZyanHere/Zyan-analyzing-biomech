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

from ..biomechanics.angles import compute_angles
from ..capture.camera import CameraSource
from ..capture.source import FrameSource
from ..capture.video import VideoSource
from ..config import Config
from ..errors import CaptureError, SourceExhaustedError
from ..filtering.one_euro import LandmarkFilter
from ..inference.pose import PoseEstimator
from ..metrics.timing import PipelineMetrics
from ..ui.layout import FrameComposer, canvas_size
from ..validity.anterior import AnteriorTracker
from ..validity.orientation import PlaneGate, yaw_degrees
from ..validity.rules import ValidityInputs, ValidityJudge
from ..validity.signals import BoneLengthTracker

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
    anterior = AnteriorTracker()
    bones = BoneLengthTracker(cfg.validity.bone_history_frames)
    judge = ValidityJudge(cfg.validity)
    smoother = LandmarkFilter(cfg.filtering)
    show_ui = not args.no_ui

    if show_ui:
        # Resizable, so the window manager does any scaling on the display side
        # rather than costing numpy work on the inference thread.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, cfg.ui.window_width, cfg.ui.window_height)

    deadline = time.perf_counter() + args.duration if args.duration else None
    log.info("running: %s -> %s", source.description, estimator.description)

    try:
        _loop(cfg, source, estimator, metrics, gate, anterior, bones, judge,
              smoother, show_ui, deadline)
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
    anterior: AnteriorTracker,
    bones: BoneLengthTracker,
    judge: ValidityJudge,
    smoother: LandmarkFilter,
    show_ui: bool,
    deadline: float | None,
) -> None:
    """Take the newest frame, infer, draw, measure. Repeat."""
    composer: FrameComposer | None = None
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
        marks = result.landmarks
        yaw = yaw_degrees(marks.world_xyz) if marks else float("nan")
        orientation = gate.update(yaw)

        # Every trust signal reads RAW landmarks, before any smoothing: a filter
        # removes exactly the variation these checks look for (F27).
        facing = anterior.update(marks.image_xy) if marks else None
        if marks:
            bones.update(marks)

        # Two signals from here on. Smoothed landmarks produce the displayed
        # angles; RAW landmarks feed validity above, because a filter removes
        # exactly the variation those checks look for (F27).
        if marks is None:
            smoother.reset()
            angles = {}
        else:
            angles = compute_angles(smoother.apply(marks, frame.capture_ts), facing)
        verdicts = judge.judge_all(
            ValidityInputs(marks, orientation, bones, anterior.is_established)
        )
        metrics.biomech_ms.add((time.perf_counter() - started) * 1000.0)

        if show_ui:
            started = time.perf_counter()
            if composer is None:
                composer = FrameComposer(*frame.image.shape[1::-1])
                cv2.resizeWindow(WINDOW, *canvas_size(*frame.image.shape[1::-1]))
            canvas = composer.compose(
                image=frame.image,
                landmarks=marks,
                angles=angles,
                verdicts=verdicts,
                orientation=orientation,
                stats=metrics.snapshot(),
                source_note=source.description,
                facing_note=anterior.confidence_note,
                min_visibility=cfg.validity.min_visibility,
                draw_pose=cfg.ui.draw_skeleton,
                banner=_banner_for(source, marks),
            )
            cv2.imshow(WINDOW, canvas)
            metrics.render_ms.add((time.perf_counter() - started) * 1000.0)
            if (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS:
                raise KeyboardInterrupt

        metrics.source_drops = getattr(source, "drops", 0)
        metrics.frame_displayed(frame.capture_ts)


def _banner_for(source: FrameSource, marks: object) -> str:
    """The one condition, if any, the user needs to act on right now."""
    if not source.is_healthy:
        return "camera stopped delivering frames - waiting"
    if marks is None:
        return "no person detected"
    return ""


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
