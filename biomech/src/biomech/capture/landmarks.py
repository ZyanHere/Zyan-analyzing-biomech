"""Replaying a recorded landmark stream.

Owns: reading a JSONL session back into the types the pipeline consumes.
Does NOT own: writing them (see recording/writer.py), or any model.

This source enters the pipeline **after** inference. It cannot exercise the
pose model - its landmarks are already one model's output - so it cannot be
used to compare models. What it can do is make everything downstream
deterministic: the filter comparison, the convention checks and every accuracy
analysis in the investigation ran on a stream like this, because two filters
cannot be compared on a live human who moves differently each time.

It also reads the investigation's own fixture format, which stored normalised
coordinates rather than pixels. Detecting that from the header - or from the
data when the header predates the format - lets a reviewer replay the recorded
sessions without re-recording anything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..errors import CaptureError, SourceExhaustedError
from ..types import LANDMARK_COUNT, Frame, LandmarkSet, PoseResult

log = logging.getLogger(__name__)

# Normalised coordinates are fractions of the frame, so they never exceed ~1.
# Pixels on any real frame are far larger. The gap is wide enough that this
# is a safe discriminator when a file has no header to say which it is.
_NORMALISED_CEILING = 2.0
_DEFAULT_SIZE = (640, 480)


def _frame_size(record: dict) -> tuple[int, int]:
    """Frame dimensions from a record, tolerating both formats.

    The application writes "width"/"height". The investigation's fixtures carry
    neither, and use "w" for the world landmarks - so a value under that key is
    only a width if it is actually a number.
    """
    width = record.get("width", record.get("w"))
    height = record.get("height", record.get("h"))
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        return int(width), int(height)
    return _DEFAULT_SIZE


class LandmarkSource:
    """Yields recorded poses, one per call, in capture order."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: list[dict] = []
        self._index = 0
        self._size = _DEFAULT_SIZE
        self._normalised = False
        self._config: dict | None = None
        self._load()

    def _load(self) -> None:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CaptureError(f"Could not read {self._path}: {exc}") from exc

        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if "format" in record:
                self._config = record.get("config")
                self._normalised = record.get("coordinates") == "normalised"
                continue
            self._records.append(record)

        if not self._records:
            raise CaptureError(f"{self._path} contains no landmark records.")

        self._detect_layout()
        log.info(
            "replaying %s (%d records, %s coordinates)",
            self._path.name, len(self._records),
            "normalised" if self._normalised else "pixel",
        )

    def _detect_layout(self) -> None:
        """Work out the coordinate convention and frame size from the data.

        The investigation's fixtures predate the header, so this falls back to
        inspecting the values rather than refusing to read them.
        """
        first = next((r for r in self._records if self._points_of(r) is not None), None)
        if first is None:
            return
        self._size = _frame_size(first)
        points = self._points_of(first)
        assert points is not None
        if float(np.max(np.abs(np.asarray(points, dtype=float)[:, :2]))) <= _NORMALISED_CEILING:
            self._normalised = True

    @staticmethod
    def _points_of(record: dict) -> list | None:
        """Image-space landmarks under either key."""
        return record.get("xy") or record.get("lm")

    def next_pose(self) -> tuple[Frame, PoseResult]:
        """The next recorded frame and its pose.

        Raises:
            SourceExhaustedError: the recording ended. Terminal, not a fault.
        """
        if self._index >= len(self._records):
            raise SourceExhaustedError(
                f"End of {self._path.name} after {len(self._records)} records."
            )
        record = self._records[self._index]
        self._index += 1

        # No image exists in a landmark recording, so a blank canvas stands in.
        # Everything downstream of inference is unaffected; only the video pane
        # is empty, which is honest about what was recorded.
        blank = np.zeros((self._size[1], self._size[0], 3), np.uint8)
        frame = Frame(image=blank, capture_ts=float(record.get("t", 0.0)),
                      seq=int(record.get("seq", self._index)))
        return frame, PoseResult(
            landmarks=self._landmarks_of(record),
            capture_ts=frame.capture_ts,
            seq=frame.seq,
            inference_ms=float(record.get("infer_ms", 0.0)),
        )

    def _landmarks_of(self, record: dict) -> LandmarkSet | None:
        points = self._points_of(record)
        if not record.get("found", points is not None) or points is None:
            return None

        raw = np.asarray(points, dtype=float)
        image_xy = raw[:, :2].copy()
        if self._normalised:
            image_xy[:, 0] *= self._size[0]
            image_xy[:, 1] *= self._size[1]

        # The investigation format carried visibility as a fourth column; the
        # application's own format keeps it in a separate key.
        if "vis" in record:
            visibility = np.asarray(record["vis"], dtype=float)
        elif raw.shape[1] >= 4:
            visibility = raw[:, 3].copy()
        else:
            visibility = np.ones(LANDMARK_COUNT)

        world = record.get("xyz") or record.get("w")
        world_xyz = (
            np.asarray(world, dtype=float)
            if isinstance(world, list)
            else np.zeros((LANDMARK_COUNT, 3))
        )
        return LandmarkSet(image_xy=image_xy, world_xyz=world_xyz, visibility=visibility)

    def close(self) -> None:
        self._records = []

    @property
    def description(self) -> str:
        return f"landmarks {self._path.name} ({len(self._records)} records)"

    @property
    def recorded_config(self) -> dict | None:
        """The configuration the recording was made with, if it carried one."""
        return self._config

    @property
    def is_healthy(self) -> bool:
        return True

    @property
    def drops(self) -> int:
        return 0

    @property
    def drop_rate(self) -> float:
        return 0.0
