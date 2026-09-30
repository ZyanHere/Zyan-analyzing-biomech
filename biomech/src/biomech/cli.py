"""Command-line argument parsing.

Owns: the CLI surface, and translating it into a Config.
Does NOT own: any behaviour those arguments select. It parses and returns.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config
from .errors import ConfigError

_DESCRIPTION = """Real-time biomechanical analysis from one monocular camera.

Twelve joint angles: elbow, knee, shoulder flexion, shoulder abduction, hip and
ankle, both sides. At most ten can be valid at once - abduction needs you facing
the camera, the other five need you side-on, and one camera cannot show both.
"""


def build_parser() -> argparse.ArgumentParser:
    """The complete CLI surface."""
    p = argparse.ArgumentParser(
        prog="biomech",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("source")
    src.add_argument(
        "--source",
        choices=("camera", "video", "landmarks"),
        default="camera",
        help="where frames come from. 'video' replays an mp4 through the model; "
        "'landmarks' replays a recorded JSONL and skips the model entirely.",
    )
    src.add_argument("--path", type=Path, help="file for --source video or landmarks")
    src.add_argument("--device", type=int, default=0, help="camera index (default 0)")

    mdl = p.add_argument_group("model")
    mdl.add_argument(
        "--model",
        choices=("lite", "full", "heavy"),
        default="full",
        help="pose model variant (default full; lite is ~3x noisier at the knee)",
    )
    mdl.add_argument(
        "--running-mode",
        choices=("video", "image"),
        default="video",
        help="video tracks between frames and is 2.4x faster; image is stateless",
    )

    rec = p.add_argument_group("recording")
    rec.add_argument(
        "--record-landmarks",
        type=Path,
        metavar="PATH",
        help="write landmarks and timestamps as JSONL for later replay",
    )
    rec.add_argument(
        "--record-video",
        type=Path,
        metavar="PATH",
        help="ALSO write the raw video. This captures the subject; off by default. "
        "Needed only to compare pose models later, since a landmark stream is "
        "already one model's output.",
    )

    out = p.add_argument_group("output")
    out.add_argument("--metrics-out", type=Path, help="write timing results as JSON at exit")
    out.add_argument("--no-ui", action="store_true", help="run headless (benchmarking)")
    out.add_argument("--duration", type=float, help="stop after N seconds")
    out.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    out.add_argument(
        "--no-filter",
        action="store_true",
        help="disable temporal smoothing. Useful for measuring raw jitter, and "
        "for reading true peak range of motion, which smoothing shaves by ~10%%.",
    )

    diag = p.add_argument_group("diagnostics")
    diag.add_argument(
        "--probe",
        action="store_true",
        help="measure what the camera actually delivers and exit. Reports UNIQUE "
        "frames per second, which is not the same as read() calls per second.",
    )
    diag.add_argument(
        "--bench-inference",
        type=Path,
        metavar="VIDEO",
        help="replay a video through the model alone and report its latency, "
        "with no camera and nothing downstream. Reproducible, and it measures "
        "the machine rather than the webcam.",
    )

    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a validated Config from parsed arguments."""
    base = Config()
    cfg = replace(
        base,
        capture=replace(base.capture, device_index=args.device),
        model=replace(base.model, variant=args.model, running_mode=args.running_mode),
        ui=replace(base.ui, show_health_panel=not args.no_ui),
        filtering=replace(base.filtering, filter_landmarks=not args.no_filter),
    )
    cfg.validate()
    return cfg


def validate_args(args: argparse.Namespace) -> None:
    """Reject argument combinations that cannot work.

    Checked here rather than at use, so a mistake fails in the first second
    instead of after the model has spent twelve seconds loading.
    """
    if args.source in ("video", "landmarks"):
        if args.path is None:
            raise ConfigError(
                f"--source {args.source} needs --path pointing at the file to replay."
            )
        if not args.path.exists():
            raise ConfigError(f"No such file: {args.path}")

    if args.source == "landmarks" and args.record_video:
        raise ConfigError(
            "--record-video cannot be used with --source landmarks: a landmark "
            "recording contains no images to write."
        )

    if args.source == "camera" and args.path:
        raise ConfigError("--path applies to --source video or landmarks, not camera.")
