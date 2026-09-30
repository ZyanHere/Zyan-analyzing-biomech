"""The data types that cross module boundaries.

Owns: the shape of everything passed between stages, and the landmark index map.
Does NOT own: how any of it is produced, judged or displayed.

All types are frozen. A queued frame that a producer can still write to is a
race condition waiting for load, and immutability makes that impossible rather
than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# MediaPipe BlazePose landmark indices. Named here once so no other module
# contains a bare integer index.
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

LANDMARK_COUNT = 33


class Side(Enum):
    """Which side of the body a measurement belongs to."""

    LEFT = "L"
    RIGHT = "R"


class Plane(Enum):
    """The anatomical plane a measurement is defined in.

    This determines which way the subject must face for the measurement to be
    valid, and the two are mutually exclusive: see FINDINGS.md F34.
    """

    SAGITTAL = "sagittal"
    FRONTAL = "frontal"


class MeasurementName(Enum):
    """The six mandatory measurement types.

    Twelve measurements in total, since each is computed for both sides.
    """

    ELBOW_FLEXION = "elbow_flexion"
    KNEE_FLEXION = "knee_flexion"
    SHOULDER_FLEXION = "shoulder_flexion"
    SHOULDER_ABDUCTION = "shoulder_abduction"
    HIP_FLEXION = "hip_flexion"
    ANKLE_ANGLE = "ankle_angle"


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured image with the metadata needed to time it.

    `capture_ts` travels with the frame all the way to display. That single
    field is what makes honest end-to-end latency measurable.
    """

    image: np.ndarray
    capture_ts: float
    seq: int

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        h, w = self.image.shape[:2]
        return w, h


@dataclass(frozen=True, slots=True)
class LandmarkSet:
    """One person's landmarks for one frame.

    image_xy:   (33, 2) pixel coordinates. The representation angles are
                computed from; see ARCHITECTURE section 7.
    world_xyz:  (33, 3) metric coordinates, hip-centred and camera-aligned.
                Used only for the orientation estimate, never for angle
                magnitudes, which it compresses by 23% (FINDINGS.md F28).
    visibility: (33,) the model's own confidence per landmark.
    """

    image_xy: np.ndarray
    world_xyz: np.ndarray
    visibility: np.ndarray

    def min_visibility(self, indices: tuple[int, ...]) -> float:
        """Lowest confidence across a specific landmark chain.

        Evaluated per measurement rather than globally: leg visibility stayed
        at 0.95 while an arm was hidden behind the back (FINDINGS.md F30).
        """
        return float(np.min(self.visibility[list(indices)]))


@dataclass(frozen=True, slots=True)
class PoseResult:
    """The outcome of running the model on one frame.

    "No person detected" is an explicit state, never an empty landmark array,
    so downstream code cannot mistake absence for a pose at the origin.
    """

    landmarks: LandmarkSet | None
    capture_ts: float
    seq: int
    inference_ms: float

    @property
    def has_person(self) -> bool:
        return self.landmarks is not None


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether a measurement is believable, and if not, why not.

    The reason is part of the type because the UI must show it. A bare boolean
    would force the display layer to guess.
    """

    is_valid: bool
    reason: str = ""
    advisory: str = ""

    @classmethod
    def ok(cls, advisory: str = "") -> Verdict:
        return cls(is_valid=True, advisory=advisory)

    @classmethod
    def rejected(cls, reason: str) -> Verdict:
        return cls(is_valid=False, reason=reason)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One joint angle, in the goniometry chart's convention.

    `value_deg` is signed where the chart names two directions: positive is
    flexion, abduction or dorsiflexion; negative is extension, adduction or
    plantarflexion. It is never displayed as a negative number - see
    `biomechanics.conventions.present` - because the chart lists the two
    directions as separate named ranges.
    """

    name: MeasurementName
    side: Side
    value_deg: float
    verdict: Verdict

    @property
    def key(self) -> str:
        """Stable identifier, e.g. 'elbow_flexion_L'."""
        return f"{self.name.value}_{self.side.value}"
