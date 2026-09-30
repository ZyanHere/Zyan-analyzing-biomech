"""Pose estimation: one frame in, 33 landmarks out.

Owns: model lifetime, VIDEO-mode tracking state, landmark extraction, and the
timing of the model call in isolation.
Does NOT own: what any landmark means anatomically. This module knows nothing
about elbows.

VIDEO mode is the default and the reason this class is single-threaded.
`detect_for_video` carries tracking state between frames: it locates the person
once and then follows the landmarks, which is 2.4x faster than re-detecting
every frame and cuts frame-to-frame knee jump by 82% (FINDINGS.md F36).

That tracking assumes a single contiguous stream of monotonically increasing
timestamps. Splitting frames across two workers gives each detector every
second frame and breaks it - knee jump rose from 5.31 to 9.32 degrees in
testing. So the correct parallelism here is none, and this class must not be
shared between threads.
"""

from __future__ import annotations

import logging
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ..config import ModelConfig
from ..errors import ModelError
from ..paths import model_download_url, model_path
from ..types import LANDMARK_COUNT, Frame, LandmarkSet, PoseResult

log = logging.getLogger(__name__)

_RUNNING_MODES = {
    "video": vision.RunningMode.VIDEO,
    "image": vision.RunningMode.IMAGE,
}


class MonotonicMillis:
    """Turns capture timestamps into the strictly increasing ms VIDEO mode needs.

    Separate from the estimator so it can be tested without loading a model -
    it is the subtlest code in this module and the failure is silent: MediaPipe
    rejects a repeated timestamp, and two frames can easily land inside the same
    millisecond at 30 FPS.

    Measured from the first frame rather than from zero, so the model's notion
    of elapsed time matches reality rather than starting at an arbitrary offset.
    """

    def __init__(self) -> None:
        self._first_ts: float | None = None
        self._last_ms = -1

    def __call__(self, capture_ts: float) -> int:
        if self._first_ts is None:
            self._first_ts = capture_ts
        ms = int((capture_ts - self._first_ts) * 1000.0)
        # Forced forward rather than repeated: a duplicate is rejected, and a
        # frame that arrives out of order must not rewind the model's clock.
        ms = max(ms, self._last_ms + 1)
        self._last_ms = ms
        return ms


class PoseEstimator:
    """Wraps MediaPipe's PoseLandmarker. One instance, one thread."""

    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._path = model_path(cfg.variant)
        self._require_model_file()
        self._detector = self._create_detector()
        self._timestamps = MonotonicMillis()

    def _require_model_file(self) -> None:
        """Fail with the download command rather than a library stack trace."""
        if self._path.exists():
            return
        raise ModelError(
            f"Model file not found: {self._path}\n"
            f"Download it with:\n"
            f'  curl -L -o "{self._path}" \\\n'
            f"    {model_download_url(self._cfg.variant)}"
        )

    def _create_detector(self) -> vision.PoseLandmarker:
        """Build the detector, logging the one-time cost.

        First construction initialises the XNNPACK delegate and took ~12 s on
        the test machine (FINDINGS.md F4). Logged so a frozen window during
        startup is explained rather than mysterious.
        """
        mode = _RUNNING_MODES.get(self._cfg.running_mode)
        if mode is None:
            raise ModelError(
                f"Unknown running mode {self._cfg.running_mode!r}. "
                f"Expected one of: {', '.join(_RUNNING_MODES)}."
            )

        log.info("loading %s model (first load can take ~12s)...", self._cfg.variant)
        started = time.perf_counter()
        try:
            detector = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self._path)),
                    running_mode=mode,
                    num_poses=1,
                )
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as our own type
            raise ModelError(f"Could not load {self._path}: {exc}") from exc

        log.info(
            "model ready in %.1fs (%s, %s mode)",
            time.perf_counter() - started,
            self._cfg.variant,
            self._cfg.running_mode,
        )
        return detector

    def estimate(self, frame: Frame) -> PoseResult:
        """Run the model on one frame.

        The returned `inference_ms` times the model call alone, not the colour
        conversion or the wrapping, because that is the number the assignment
        asks for separately from end-to-end latency.
        """
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        started = time.perf_counter()
        if self._cfg.running_mode == "video":
            result = self._detector.detect_for_video(image, self._timestamps(frame.capture_ts))
        else:
            result = self._detector.detect(image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return PoseResult(
            landmarks=self._to_landmark_set(result, frame),
            capture_ts=frame.capture_ts,
            seq=frame.seq,
            inference_ms=elapsed_ms,
        )

    @staticmethod
    def _to_landmark_set(result: object, frame: Frame) -> LandmarkSet | None:
        """Convert MediaPipe's output, or None when no person was found.

        None is returned rather than a zero-filled array so that downstream code
        cannot mistake absence for a person standing at the image origin.
        """
        pose = getattr(result, "pose_landmarks", None)
        if not pose:
            return None

        width, height = frame.size
        landmarks = pose[0]
        image_xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
        visibility = np.array([lm.visibility for lm in landmarks], dtype=np.float64)

        world = getattr(result, "pose_world_landmarks", None)
        world_xyz = (
            np.array([[lm.x, lm.y, lm.z] for lm in world[0]], dtype=np.float64)
            if world
            else np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
        )

        return LandmarkSet(image_xy=image_xy, world_xyz=world_xyz, visibility=visibility)

    def close(self) -> None:
        """Release the model. Safe to call twice."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    @property
    def description(self) -> str:
        return f"pose_landmarker_{self._cfg.variant} ({self._cfg.running_mode} mode)"
