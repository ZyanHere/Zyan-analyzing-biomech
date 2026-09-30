"""Video file replay.

Owns: reading frames from an mp4 in order.
Does NOT own: timing policy, or anything downstream.

This is the source that can compare pose models, because it replays the same
*images* through whichever model is configured. A landmark recording cannot do
that - its landmarks are already one model's output (FINDINGS.md F16, F24).

Deliberately single-threaded and unthrottled. There is no device to fall behind
and no freshness to preserve: every frame matters, and reading as fast as
possible is what makes throughput measurable independently of a camera that
caps at 30 FPS.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from ..errors import CaptureError, SourceExhaustedError
from ..types import Frame

log = logging.getLogger(__name__)


class VideoSource:
    """Replays an mp4, one frame per call, in capture order."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seq = 0
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise CaptureError(
                f"Could not open video {path}. Is it a readable mp4?"
            )
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("replaying %s (%d frames)", path.name, self._total)

    def next_frame(self) -> Frame | None:
        """The next frame in order.

        Raises:
            SourceExhaustedError: the file ended. Terminal, not a fault.
        """
        ok, image = self._cap.read()
        if not ok or image is None:
            raise SourceExhaustedError(
                f"End of {self._path.name} after {self._seq} frames."
            )
        self._seq += 1
        return Frame(image=image, capture_ts=time.perf_counter(), seq=self._seq)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def description(self) -> str:
        return f"video {self._path.name} ({self._total} frames)"

    @property
    def is_healthy(self) -> bool:
        """Always true: a file does not degrade, it ends."""
        return True

    @property
    def drops(self) -> int:
        """Always zero: every frame is delivered, none is discarded."""
        return 0

    @property
    def drop_rate(self) -> float:
        return 0.0
