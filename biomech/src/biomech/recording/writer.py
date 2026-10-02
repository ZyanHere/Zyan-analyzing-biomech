"""Recording a session for later replay.

Owns: the on-disk format, and writing it.
Does NOT own: replaying it (see capture/landmarks.py).

Two levels, because they answer different questions:

  landmarks (JSONL)  small, always safe to keep, and enough to exercise
                     everything downstream of the model - filtering, angles,
                     validity, display. This is what makes the filter and
                     convention comparisons reproducible.
  video (mp4)        the only thing that can compare pose MODELS, because it
                     replays identical *images*. A landmark stream cannot: its
                     landmarks are already one model's output (F16, F24).

Video is off by default and logged loudly when enabled, because it captures the
subject. Landmarks are coordinates rather than imagery.

Every recording carries the configuration that produced it, so a replay is
self-describing: a result is only meaningful alongside the settings that
produced it, and every threshold in this system came from an experiment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..types import Frame, PoseResult

log = logging.getLogger(__name__)

FORMAT_VERSION = 1


class SessionRecorder:
    """Writes landmarks, and optionally video, for later replay."""

    def __init__(
        self,
        config: Config,
        landmarks_path: Path | None,
        video_path: Path | None,
        fps_hint: float = 30.0,
    ) -> None:
        self._landmarks_path = landmarks_path
        self._video_path = video_path
        self._fps_hint = fps_hint
        self._handle = None
        self._writer: cv2.VideoWriter | None = None
        self._frames = 0

        if landmarks_path is not None:
            landmarks_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = landmarks_path.open("w", encoding="utf-8")
            self._write_header(config)
            log.info("recording landmarks -> %s", landmarks_path)

        if video_path is not None:
            log.warning(
                "recording VIDEO -> %s. This captures images of the subject; "
                "landmark recording alone is enough for everything except "
                "comparing pose models.",
                video_path,
            )

    def _write_header(self, config: Config) -> None:
        """First line describes the recording, so a replay needs no companion."""
        assert self._handle is not None
        header = {
            "format": FORMAT_VERSION,
            "coordinates": "pixels",
            "config": config.to_dict(),
        }
        self._handle.write(json.dumps(header) + "\n")

    def write(self, frame: Frame, result: PoseResult) -> None:
        """Record one frame's outcome."""
        self._frames += 1
        if self._writer is None and self._video_path is not None:
            self._open_video(frame)
        if self._writer is not None:
            self._writer.write(frame.image)
        if self._handle is not None:
            self._handle.write(json.dumps(self._as_record(frame, result)) + "\n")

    def _open_video(self, frame: Frame) -> None:
        """Opened on the first frame, when the true size is finally known."""
        assert self._video_path is not None
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self._video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps_hint,
            frame.size,
        )

    @staticmethod
    def _as_record(frame: Frame, result: PoseResult) -> dict[str, object]:
        record: dict[str, object] = {
            "seq": frame.seq,
            "t": round(frame.capture_ts, 6),
            # Spelled out, not "w"/"h": the investigation's fixture format
            # uses "w" for world landmarks, and a one-letter key that means
            # two different things in two formats is a bug waiting to happen.
            "width": frame.size[0],
            "height": frame.size[1],
            "found": result.has_person,
            "infer_ms": round(result.inference_ms, 3),
        }
        marks = result.landmarks
        if marks is not None:
            record["xy"] = np.round(marks.image_xy, 2).tolist()
            record["xyz"] = np.round(marks.world_xyz, 5).tolist()
            record["vis"] = np.round(marks.visibility, 4).tolist()
        return record

    def close(self) -> None:
        """Flush and release. Safe to call twice."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            log.info("wrote %d records -> %s", self._frames, self._landmarks_path)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
