"""The composition root.

Owns: wiring, the main loop, lifecycle and shutdown. The only module that knows
all the others.
Does NOT own: any domain logic. Every decision here is about *when* things
happen, never what an angle means or whether it can be trusted.

Sequential by design. VIDEO-mode tracking assumes a single contiguous stream,
and splitting frames across workers breaks it - knee jump rose from 5.31 to
9.32 degrees in testing (FINDINGS.md F36). Only capture runs on its own thread,
so a slow consumer gets a fresh frame rather than a backlog.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

import cv2

from ..biomechanics.angles import compute_angles
from ..config import Config
from ..errors import SourceExhaustedError
from ..filtering.one_euro import LandmarkFilter
from ..metrics.report import build_report, write_report
from ..metrics.timing import PipelineMetrics
from ..recording.writer import SessionRecorder
from ..types import Frame, PoseResult
from ..ui.layout import FrameComposer, canvas_size
from ..validity.anterior import AnteriorTracker
from ..validity.orientation import PlaneGate, yaw_degrees
from ..validity.rules import ValidityInputs, ValidityJudge
from ..validity.signals import BoneLengthTracker
from .provider import PoseProvider, open_provider

log = logging.getLogger(__name__)

WINDOW = "biomech"
_QUIT_KEYS = (ord("q"), 27)  # q or Escape
_IDLE_SLEEP_S = 0.001


@dataclass
class Pipeline:
    """Everything that carries state across frames.

    Bundled rather than passed individually: the loop needs nine collaborators,
    and a nine-parameter signature hides which of them are stateful.
    """

    cfg: Config
    provider: PoseProvider
    metrics: PipelineMetrics
    gate: PlaneGate = field(init=False)
    anterior: AnteriorTracker = field(init=False)
    bones: BoneLengthTracker = field(init=False)
    judge: ValidityJudge = field(init=False)
    smoother: LandmarkFilter = field(init=False)
    composer: FrameComposer | None = None
    recorder: SessionRecorder | None = None

    def __post_init__(self) -> None:
        self.gate = PlaneGate(self.cfg.orientation)
        self.anterior = AnteriorTracker()
        self.bones = BoneLengthTracker(self.cfg.validity.bone_history_frames)
        self.judge = ValidityJudge(self.cfg.validity)
        self.smoother = LandmarkFilter(self.cfg.filtering)


def run(cfg: Config, args: argparse.Namespace) -> int:
    """Run the pipeline until the source ends or the user stops it."""
    pipeline = Pipeline(
        cfg=cfg,
        provider=open_provider(cfg, args),
        metrics=PipelineMetrics(cfg.metrics),
    )
    if args.record_landmarks or args.record_video:
        pipeline.recorder = SessionRecorder(cfg, args.record_landmarks, args.record_video)

    show_ui = not args.no_ui
    if show_ui:
        # Resizable, so the window manager does any scaling on the display side
        # rather than costing numpy work on the inference thread.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, cfg.ui.window_width, cfg.ui.window_height)

    deadline = time.perf_counter() + args.duration if args.duration else None
    log.info("running: %s", pipeline.provider.description)

    try:
        _loop(pipeline, show_ui, deadline)
    except SourceExhaustedError as end:
        log.info("%s", end)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        if pipeline.recorder is not None:
            pipeline.recorder.close()
        pipeline.provider.close()
        if show_ui:
            cv2.destroyAllWindows()
        _report(pipeline)
        if args.metrics_out:
            _write_metrics(pipeline, args)
    return 0


def _write_metrics(pipeline: Pipeline, args: argparse.Namespace) -> None:
    """Persist the run, so a documented number is never a retyped one."""
    write_report(
        args.metrics_out,
        build_report(
            cfg=pipeline.cfg,
            metrics=pipeline.metrics,
            source=pipeline.provider.description,
            label=args.label or args.source,
            latency_kind=(
                "end_to_end" if pipeline.provider.timestamps_are_live else "processing"
            ),
        ),
    )


def _loop(pipeline: Pipeline, show_ui: bool, deadline: float | None) -> None:
    """Take the next pose, measure it, judge it, draw it. Repeat."""
    while deadline is None or time.perf_counter() < deadline:
        pose = pipeline.provider.next_pose()
        if pose is None:
            # A live camera between frames. Yielding keeps this loop from
            # spinning a core while waiting on hardware.
            time.sleep(_IDLE_SLEEP_S)
            continue

        read_at = time.perf_counter()
        frame, result = pose
        pipeline.metrics.inference_ms.add(result.inference_ms)
        if not result.has_person:
            pipeline.metrics.frames_without_person += 1
        if pipeline.recorder is not None:
            pipeline.recorder.write(frame, result)

        started = time.perf_counter()
        angles, verdicts, orientation = _measure(pipeline, frame, result)
        pipeline.metrics.biomech_ms.add((time.perf_counter() - started) * 1000.0)

        if show_ui and _draw(pipeline, frame, result, angles, verdicts, orientation):
            raise KeyboardInterrupt

        pipeline.metrics.source_drops = pipeline.provider.drops
        # For a live source this is true end-to-end latency, from the moment of
        # capture. For a replay there was no capture, so it measures processing
        # latency from the moment the record was read - a different quantity,
        # labelled as such in the report.
        pipeline.metrics.frame_displayed(
            frame.capture_ts if pipeline.provider.timestamps_are_live else read_at
        )


def _measure(pipeline: Pipeline, frame: Frame, result: PoseResult) -> tuple[dict, dict, object]:
    """Angles, verdicts and orientation for one frame.

    Two signals leave here. RAW landmarks feed the validity checks, because a
    filter removes exactly the variation they look for (F27). SMOOTHED
    landmarks feed the angles, because smoothing positions keeps the skeleton
    geometrically consistent in a way that smoothing an output angle cannot.
    """
    marks = result.landmarks
    orientation = pipeline.gate.update(
        yaw_degrees(marks.world_xyz) if marks else float("nan")
    )

    facing = pipeline.anterior.update(marks.image_xy) if marks else None
    if marks is not None:
        pipeline.bones.update(marks)
        angles = compute_angles(pipeline.smoother.apply(marks, frame.capture_ts), facing)
    else:
        pipeline.smoother.reset()
        angles = {}

    verdicts = pipeline.judge.judge_all(
        ValidityInputs(marks, orientation, pipeline.bones, pipeline.anterior.is_established)
    )
    return angles, verdicts, orientation


def _draw(
    pipeline: Pipeline, frame: Frame, result: PoseResult,
    angles: dict, verdicts: dict, orientation: object,
) -> bool:
    """Render one frame. Returns True if the user asked to quit."""
    started = time.perf_counter()
    if pipeline.composer is None:
        width, height = frame.size
        pipeline.composer = FrameComposer(width, height)
        cv2.resizeWindow(WINDOW, *canvas_size(width, height))

    canvas = pipeline.composer.compose(
        image=frame.image,
        landmarks=result.landmarks,
        angles=angles,
        verdicts=verdicts,
        orientation=orientation,
        stats=pipeline.metrics.snapshot(),
        source_note=pipeline.provider.description,
        facing_note=pipeline.anterior.confidence_note,
        min_visibility=pipeline.cfg.validity.min_visibility,
        draw_pose=pipeline.cfg.ui.draw_skeleton,
        banner=_banner_for(pipeline.provider, result),
    )
    cv2.imshow(WINDOW, canvas)
    pipeline.metrics.render_ms.add((time.perf_counter() - started) * 1000.0)
    return (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS


def _banner_for(provider: PoseProvider, result: PoseResult) -> str:
    """The one condition, if any, the user needs to act on right now."""
    if not provider.is_healthy:
        return "camera stopped delivering frames - waiting"
    if not result.has_person:
        return "no person detected"
    return ""


def _report(pipeline: Pipeline) -> None:
    """Final timings, so a headless run still reports its result."""
    s = pipeline.metrics.snapshot()
    log.info("--- pipeline ---")
    log.info("  frames processed : %6.0f  (%.0f without a person)",
             s["frames_processed"], s["frames_without_person"])
    log.info("  displayed FPS    : %6.1f", s["displayed_fps"])
    if pipeline.provider.runs_model:
        log.info("  inference  mean/p50/p95: %5.1f / %.1f / %.1f ms",
                 s["inference_mean_ms"], s["inference_p50_ms"], s["inference_p95_ms"])
    else:
        log.info("  inference         :   replayed, model not run")
    log.info("  biomech    p50    : %6.2f ms", s["biomech_p50_ms"])
    log.info("  render     p50/p95: %6.1f / %.1f ms",
             s["render_p50_ms"], s["render_p95_ms"])
    label = "end-to-end" if pipeline.provider.timestamps_are_live else "processing"
    log.info("  %-10s mean/p50/p95: %5.1f / %.1f / %.1f ms", label,
             s["end_to_end_mean_ms"], s["end_to_end_p50_ms"], s["end_to_end_p95_ms"])
    log.info("  source drops     : %6.0f", s["source_drops"])
