"""Temporal smoothing of landmark positions.

Owns: the One Euro filter and its time handling.
Does NOT own: which landmarks matter, or what they mean.

Applied to **positions**, before angles are computed. Measured 40-45% better
than smoothing the resulting angle during real movement, and equivalent at rest
(FINDINGS.md F27): smoothing positions keeps the skeleton geometrically
consistent, which smoothing an output angle cannot repair after the fact.

One Euro rather than a moving average because a fixed constant forces a choice
between jitter and lag, and loses both ways (F15):

    EMA alpha=0.15   best elbow jitter, but 240 ms of lag - seven frames
                     and on the knee it was WORSE on both axes at once
    One Euro         equal or better jitter at 0-48 ms

It adapts its cutoff to observed velocity: heavy smoothing when a joint is
still, and out of the way when it moves. Being dt-aware also means dropped
frames degrade it gracefully, where a fixed-alpha EMA silently changes meaning
whenever the frame rate varies.

**Peaks must be read from the unfiltered signal.** Every filter shaves extremes;
One Euro retains 88-89% of movement range, so a filtered signal under-reports
maximum range of motion by about 10%. Display smoothed; measure peaks raw.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import FilterConfig
from ..types import LandmarkSet


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """Low-pass smoothing factor for a given cutoff and timestep."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


class OneEuroArrayFilter:
    """One Euro applied element-wise to an array of any shape.

    Vectorised rather than one filter object per coordinate: 66 scalar filters
    would be 66 Python-level updates per frame for arithmetic numpy does in one.
    The algorithm is identical; only the loop is gone.
    """

    def __init__(self, cfg: FilterConfig) -> None:
        self._cfg = cfg
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None
        self._last_ts: float | None = None

    def reset(self) -> None:
        """Forget all history. The next sample passes through unchanged."""
        self._value = None
        self._derivative = None
        self._last_ts = None

    def apply(self, sample: np.ndarray, timestamp_s: float) -> np.ndarray:
        """Smooth one sample, using capture time to derive the timestep.

        Time comes from the frame's capture timestamp, never from a wall clock
        read here: wall-clock would fold queueing delay into the velocity
        estimate and make the smoothing depend on system load.
        """
        if self._value is None or self._last_ts is None:
            self._value = sample.copy()
            self._derivative = np.zeros_like(sample)
            self._last_ts = timestamp_s
            return sample

        dt = timestamp_s - self._last_ts
        if dt <= 0.0 or dt > self._cfg.dt_max_s:
            # Either time went backwards, or the gap is long enough that the
            # previous sample is no longer evidence about this one. Restarting
            # is honest; interpolating across it would invent motion.
            self.reset()
            return self.apply(sample, timestamp_s)

        dt = max(dt, self._cfg.dt_min_s)
        self._last_ts = timestamp_s

        raw_derivative = (sample - self._value) / dt
        a_d = _alpha(self._cfg.d_cutoff_hz, dt)
        self._derivative = a_d * raw_derivative + (1.0 - a_d) * self._derivative

        # The adaptive part: the faster the point is moving, the higher the
        # cutoff, and so the less it is smoothed.
        cutoff = self._cfg.min_cutoff_hz + self._cfg.beta * np.abs(self._derivative)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        self._value = a * sample + (1.0 - a) * self._value
        return self._value.copy()


class LandmarkFilter:
    """Smooths a whole landmark set, leaving the original untouched.

    Returns a new LandmarkSet rather than mutating: the raw landmarks are still
    needed by the validity layer, which must judge the data rather than the
    smoother's opinion of it.
    """

    def __init__(self, cfg: FilterConfig) -> None:
        self._enabled = cfg.filter_landmarks
        self._image = OneEuroArrayFilter(cfg)
        self._world = OneEuroArrayFilter(cfg)

    def apply(self, landmarks: LandmarkSet, capture_ts: float) -> LandmarkSet:
        if not self._enabled:
            return landmarks
        return LandmarkSet(
            image_xy=self._image.apply(landmarks.image_xy, capture_ts),
            world_xyz=self._world.apply(landmarks.world_xyz, capture_ts),
            visibility=landmarks.visibility,
        )

    def reset(self) -> None:
        """Called when tracking is lost, so a reappearing subject does not get
        smoothed toward where they used to be."""
        self._image.reset()
        self._world.reset()
