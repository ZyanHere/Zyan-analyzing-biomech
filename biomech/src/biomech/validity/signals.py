"""The individual trust signals.

Owns: computing each signal, and nothing about how they combine.
Does NOT own: the veto policy (see rules.py), or any angle.

Every signal here runs on **raw** landmarks, before smoothing. Filtering exists
to remove exactly the variation these checks look for - One Euro cuts
frame-to-frame jump by 40-45% (FINDINGS.md F27), which would suppress the bone
length variance that flags an unstable reconstruction. Judging trust from a
smoothed signal asks whether the smoother did its job, not whether the data was
sound.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..biomechanics.angles import CHAIN, REQUIRED_LANDMARKS
from ..types import (
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    LandmarkSet,
    MeasurementName,
    Side,
)

# Segments whose length must not change. Used for the variance check (per side,
# vetoing) and the asymmetry check (bilateral, advisory only).
SEGMENTS: dict[str, tuple[str, str]] = {
    "upper_arm": ("shoulder", "elbow"),
    "forearm": ("elbow", "wrist"),
    "femur": ("hip", "knee"),
    "shank": ("knee", "ankle"),
}

# Which segments each measurement depends on, for the variance veto.
MEASUREMENT_SEGMENTS: dict[MeasurementName, tuple[str, ...]] = {
    MeasurementName.ELBOW_FLEXION: ("upper_arm", "forearm"),
    MeasurementName.KNEE_FLEXION: ("femur", "shank"),
    MeasurementName.SHOULDER_FLEXION: ("upper_arm",),
    MeasurementName.SHOULDER_ABDUCTION: ("upper_arm",),
    MeasurementName.HIP_FLEXION: ("femur",),
    MeasurementName.ANKLE_ANGLE: ("shank",),
}

# The anatomical frame is built from these. Without them there is no body-
# relative reference and every signed measurement is meaningless.
FRAME_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)


def min_visibility(landmarks: LandmarkSet, name: MeasurementName, side: Side) -> float:
    """Lowest confidence across this measurement's own landmark chain.

    Per measurement, never global: leg visibility stayed at 0.95 while an arm
    was hidden behind the back, so a single overall number would have missed it
    entirely (FINDINGS.md F30).
    """
    chain = CHAIN[side]
    indices = tuple(chain[part] for part in REQUIRED_LANDMARKS[name])
    return landmarks.min_visibility(indices)


def frame_visibility(landmarks: LandmarkSet) -> float:
    """Lowest confidence across the shoulders and hips."""
    return landmarks.min_visibility(FRAME_LANDMARKS)


class BoneLengthTracker:
    """Rolling history of segment lengths, per side.

    A rigid body has constant segment lengths, so variation is reconstruction
    error - measurable with no ground truth, no calibration and no cooperation
    from the subject (F22).
    """

    def __init__(self, window: int) -> None:
        self._window = window
        self._history: dict[tuple[str, Side], deque[float]] = {}

    def update(self, landmarks: LandmarkSet) -> None:
        """Record this frame's segment lengths."""
        for side in (Side.LEFT, Side.RIGHT):
            chain = CHAIN[side]
            for segment, (proximal, distal) in SEGMENTS.items():
                length = float(
                    np.linalg.norm(
                        landmarks.image_xy[chain[proximal]] - landmarks.image_xy[chain[distal]]
                    )
                )
                key = (segment, side)
                if key not in self._history:
                    self._history[key] = deque(maxlen=self._window)
                self._history[key].append(length)

    def variance_pct(self, segment: str, side: Side) -> float:
        """Coefficient of variation for one segment, as a percentage.

        NaN until there is enough history to mean anything. A veto signal that
        cannot yet be computed must block rather than silently pass, so the
        caller treats NaN as "unknown", not "fine".
        """
        samples = self._history.get((segment, side))
        if samples is None or len(samples) < max(5, self._window // 4):
            return float("nan")
        values = np.fromiter(samples, dtype=float)
        mean = float(values.mean())
        return 100.0 * float(values.std()) / mean if mean > 1e-9 else float("nan")

    def worst_variance_pct(self, name: MeasurementName, side: Side) -> float:
        """The least stable segment this measurement depends on."""
        values = [self.variance_pct(seg, side) for seg in MEASUREMENT_SEGMENTS[name]]
        if any(np.isnan(v) for v in values):
            return float("nan")
        return max(values)

    def asymmetry_pct(self, segment: str) -> float:
        """Left/right difference for one segment, as a percentage.

        **Advisory only.** It is bilateral, so it cannot be computed when one
        side is hidden - and rejecting a perfectly visible left knee because the
        right leg is occluded would be stricter than the brief requires. It also
        false-positives on camera elevation: 14.1% asymmetry on a measurement
        that was accurate to 0.2 degrees, because a low camera foreshortens the
        near leg differently from the far one (F37).

        Returns NaN when either side lacks history.
        """
        left = self._history.get((segment, Side.LEFT))
        right = self._history.get((segment, Side.RIGHT))
        if not left or not right:
            return float("nan")
        mean_left, mean_right = float(np.mean(left)), float(np.mean(right))
        if mean_right < 1e-9:
            return float("nan")
        return 100.0 * abs(mean_left - mean_right) / mean_right

    def worst_asymmetry_pct(self, name: MeasurementName) -> float:
        values = [self.asymmetry_pct(seg) for seg in MEASUREMENT_SEGMENTS[name]]
        finite = [v for v in values if not np.isnan(v)]
        return max(finite) if finite else float("nan")
