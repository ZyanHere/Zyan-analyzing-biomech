"""Where a pose comes from.

Owns: the abstraction that makes camera, video and landmark replay
interchangeable to everything downstream.
Does NOT own: capture, inference, or any measurement.

The three sources do not enter at the same place. Camera and video produce
*images* and must run the model; a landmark recording is already the model's
output and enters after it. Rather than sprinkle that distinction through the
loop, both are expressed as "give me the next pose":

    camera / video  ->  Frame  ->  PoseEstimator  ->  PoseResult
    landmarks       ->                                PoseResult

Downstream code - filtering, angles, validity, display - is then identical in
all three modes, which is what makes a replayed result comparable to a live one.
"""

from __future__ import annotations

import argparse
from typing import Protocol

from ..capture.camera import CameraSource
from ..capture.landmarks import LandmarkSource
from ..capture.video import VideoSource
from ..config import Config
from ..errors import CaptureError
from ..inference.pose import PoseEstimator
from ..types import Frame, PoseResult


class PoseProvider(Protocol):
    """Anything that can produce the next pose for the pipeline."""

    def next_pose(self) -> tuple[Frame, PoseResult] | None:
        """The next frame and its pose, or None if none is ready yet.

        Raises:
            SourceExhaustedError: a finite source has ended.
        """
        ...

    def close(self) -> None: ...

    @property
    def description(self) -> str: ...

    @property
    def is_healthy(self) -> bool: ...

    @property
    def drops(self) -> int: ...

    @property
    def runs_model(self) -> bool:
        """Whether this provider exercises the pose model.

        False for landmark replay, which is why it cannot be used to compare
        models - a distinction worth being explicit about rather than implied.
        """
        ...

    @property
    def timestamps_are_live(self) -> bool:
        """Whether capture_ts is on the wall clock.

        False for landmark replay, where the timestamps are relative to a
        recording made at some other time. End-to-end latency measured against
        them is meaningless - the difference came out as 127,089,632 ms - so
        the runner measures processing latency instead.

        The recorded timestamps are still used for the filter's dt, because
        replay must reproduce the original cadence to be comparable with live.
        """
        ...


class LiveProvider:
    """Frames from a camera or video file, run through the model."""

    def __init__(self, source: CameraSource | VideoSource, estimator: PoseEstimator) -> None:
        self._source = source
        self._estimator = estimator

    def next_pose(self) -> tuple[Frame, PoseResult] | None:
        frame = self._source.next_frame()
        if frame is None:
            return None
        return frame, self._estimator.estimate(frame)

    def close(self) -> None:
        self._estimator.close()
        self._source.close()

    @property
    def description(self) -> str:
        return f"{self._source.description} -> {self._estimator.description}"

    @property
    def is_healthy(self) -> bool:
        return self._source.is_healthy

    @property
    def drops(self) -> int:
        return self._source.drops

    @property
    def runs_model(self) -> bool:
        return True

    @property
    def timestamps_are_live(self) -> bool:
        return True


class ReplayProvider:
    """Poses read straight from a recording, with no model involved."""

    def __init__(self, source: LandmarkSource) -> None:
        self._source = source

    def next_pose(self) -> tuple[Frame, PoseResult] | None:
        return self._source.next_pose()

    def close(self) -> None:
        self._source.close()

    @property
    def description(self) -> str:
        return f"{self._source.description} (model not run)"

    @property
    def is_healthy(self) -> bool:
        return True

    @property
    def drops(self) -> int:
        return 0

    @property
    def runs_model(self) -> bool:
        return False

    @property
    def timestamps_are_live(self) -> bool:
        return False


def open_provider(cfg: Config, args: argparse.Namespace) -> PoseProvider:
    """Build the provider the arguments select."""
    if args.source == "camera":
        return LiveProvider(CameraSource(cfg.capture), PoseEstimator(cfg.model))
    if args.source == "video":
        return LiveProvider(VideoSource(args.path), PoseEstimator(cfg.model))
    if args.source == "landmarks":
        return ReplayProvider(LandmarkSource(args.path))
    raise CaptureError(
        f"Unknown source {args.source!r}. Expected camera, video or landmarks."
    )
