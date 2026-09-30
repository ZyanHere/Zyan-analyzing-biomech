"""Live camera capture.

Owns: the device, its settings, the capture thread, and frame freshness.
Does NOT own: what a frame contains, or any model.

Three settings here were each worth a measurable amount, and all three are
easy to get wrong silently:

  DirectShow      MSMF returns the same frame up to four times - 74%
                  duplicates - so counting read() calls measures the loop
                  rather than the camera (FINDINGS.md F26).
  manual exposure auto-exposure lengthens exposure time indoors, costing 75%
                  of the frame rate AND making a motionless elbow read
                  +/- 23 deg instead of 1.5 (F7). Not a brightness setting:
                  exposure *time* is what blurs the landmarks.
  640x480         1280x720 caps at 10 FPS on the test camera regardless of
                  exposure (F5).

The device is asked for all three and then checked, because a camera that
silently ignores a request is the common case, not the exception.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

from ..config import CaptureConfig
from ..errors import CaptureError
from ..types import Frame
from .slot import LatestFrameSlot

log = logging.getLogger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}

# Consecutive failed reads before the source is called unhealthy. At ~30 FPS
# this is about a third of a second - long enough to ignore a hiccup, short
# enough that the user sees the banner while still wondering what happened.
_UNHEALTHY_AFTER_FAILURES = 10


class CameraSource:
    """A live camera, read on its own thread, newest frame wins."""

    def __init__(self, cfg: CaptureConfig) -> None:
        self._cfg = cfg
        self._slot = LatestFrameSlot()
        self._seq = 0
        self._consecutive_failures = 0
        self._stop = threading.Event()
        self._cap = self._open()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        """Open the device and verify it honoured the request."""
        backend = _BACKENDS.get(self._cfg.backend)
        if backend is None:
            raise CaptureError(
                f"Unknown capture backend {self._cfg.backend!r}. "
                f"Expected one of: {', '.join(_BACKENDS)}."
            )

        cap = cv2.VideoCapture(self._cfg.device_index, backend)
        if not cap.isOpened():
            raise CaptureError(
                f"Camera {self._cfg.device_index} could not be opened. "
                "Is another application using it, or is camera access blocked? "
                "Try a different index with --device N."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._apply_exposure(cap)
        self._verify_resolution(cap)
        return cap

    def _apply_exposure(self, cap: cv2.VideoCapture) -> None:
        """Set manual exposure, and say so loudly if the device refused.

        A device that ignores this reverts to the auto-exposure behaviour that
        costs 10x accuracy (F7). The warning is the only defence, because the
        symptom - a noisy elbow reading - looks like a model problem.
        """
        if not self._cfg.use_manual_exposure:
            return
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual on DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, self._cfg.exposure_log2)

        actual = cap.get(cv2.CAP_PROP_EXPOSURE)
        if abs(actual - self._cfg.exposure_log2) > 0.5:
            log.warning(
                "Camera did not accept manual exposure: asked for %.1f, reports %.1f. "
                "Auto-exposure costs ~75%% of the frame rate and roughly 10x the "
                "angle jitter (FINDINGS.md F7). Measurements will be degraded.",
                self._cfg.exposure_log2,
                actual,
            )
        else:
            log.info("exposure fixed at 2^%.1f s", actual)

    def _verify_resolution(self, cap: cv2.VideoCapture) -> None:
        """Refuse a resolution we did not ask for.

        Silently accepting a substitute would invalidate every timing number,
        since frame rate on this hardware is resolution-dependent (F5).
        """
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (self._cfg.width, self._cfg.height):
            cap.release()
            raise CaptureError(
                f"Camera {self._cfg.device_index} gave {w}x{h} when asked for "
                f"{self._cfg.width}x{self._cfg.height}. Frame rate on this hardware "
                "depends on resolution, so a substitute would invalidate the timing "
                "results. Choose a supported mode in config.py."
            )
        log.info("camera %d open at %dx%d via %s", self._cfg.device_index, w, h, self._cfg.backend)

    def _loop(self) -> None:
        """Read continuously, publishing only the newest frame."""
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                self._consecutive_failures += 1
                if self._consecutive_failures == _UNHEALTHY_AFTER_FAILURES:
                    log.warning("camera stopped delivering frames")
                time.sleep(0.005)
                continue

            if self._consecutive_failures >= _UNHEALTHY_AFTER_FAILURES:
                log.info("camera recovered")
            self._consecutive_failures = 0
            self._seq += 1
            # Copied before publishing: the standard is that a queued frame is
            # never written again by the producer, enforced rather than assumed.
            self._slot.put(Frame(image=image.copy(), capture_ts=time.perf_counter(),
                                 seq=self._seq))

    def next_frame(self) -> Frame | None:
        return self._slot.take()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def description(self) -> str:
        return (
            f"camera {self._cfg.device_index} {self._cfg.width}x{self._cfg.height} "
            f"{self._cfg.backend} exposure 2^{self._cfg.exposure_log2:.0f}"
        )

    @property
    def is_healthy(self) -> bool:
        return self._consecutive_failures < _UNHEALTHY_AFTER_FAILURES

    @property
    def drops(self) -> int:
        return self._slot.drops

    @property
    def drop_rate(self) -> float:
        return self._slot.drop_rate
