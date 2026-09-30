"""The twelve joint angles, in the goniometry chart's conventions.

Owns: every angle computation, in both representations.
Does NOT own: whether a result is trustworthy (validity/), smoothing
(filtering/), or display (ui/). Angles are computed unconditionally; deciding
what to believe is a separate question asked by a separate module.

Two things a naive three-point implementation gets wrong:

  Shoulder flexion and abduction use the SAME three landmarks but different
  planes. One interior angle cannot produce both (FINDINGS.md F17).

  Hinge joints and plane-projected joints need DIFFERENT implementations in 2D
  and 3D. The 3D form projects onto `anterior = cross(left, up)`, which
  degenerates to pure Z when every z is zero, so it can only ever return 0 or
  180 on image coordinates (F34). Elbow and knee, being interior angles of
  three points, are the same computation in any number of dimensions.

2D is the primary representation. 3D geometry is exact but the model's world
landmarks compress real movement by 23% - bend 150 degrees and it reports 125
(F28). 2D is wrong only when the plane is not presented to the camera, and that
condition is detectable at runtime, which the compression is not (F29).
"""

from __future__ import annotations

import numpy as np

from ..types import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    LandmarkSet,
    MeasurementName,
    Side,
)
from .frame import ImageFrame, SpatialFrame, image_frame, spatial_frame

_EPS = 1e-9

# Landmark indices per side, so no function below contains a bare integer.
CHAIN = {
    Side.LEFT: {
        "shoulder": LEFT_SHOULDER, "elbow": LEFT_ELBOW, "wrist": LEFT_WRIST,
        "hip": LEFT_HIP, "knee": LEFT_KNEE, "ankle": LEFT_ANKLE,
        "heel": LEFT_HEEL, "foot": LEFT_FOOT_INDEX,
    },
    Side.RIGHT: {
        "shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW, "wrist": RIGHT_WRIST,
        "hip": RIGHT_HIP, "knee": RIGHT_KNEE, "ankle": RIGHT_ANKLE,
        "heel": RIGHT_HEEL, "foot": RIGHT_FOOT_INDEX,
    },
}

# The landmarks each measurement depends on. Used by validity/ to evaluate
# visibility per measurement rather than globally - leg visibility stayed at
# 0.95 while an arm was hidden behind the back (FINDINGS.md F30).
REQUIRED_LANDMARKS: dict[MeasurementName, tuple[str, ...]] = {
    MeasurementName.ELBOW_FLEXION: ("shoulder", "elbow", "wrist"),
    MeasurementName.KNEE_FLEXION: ("hip", "knee", "ankle"),
    MeasurementName.SHOULDER_FLEXION: ("hip", "shoulder", "elbow"),
    MeasurementName.SHOULDER_ABDUCTION: ("hip", "shoulder", "elbow"),
    MeasurementName.HIP_FLEXION: ("shoulder", "hip", "knee"),
    MeasurementName.ANKLE_ANGLE: ("knee", "ankle", "heel", "foot"),
}


def interior_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Unsigned angle at b, in degrees. 180 means a straight limb.

    Dimension-agnostic: the same computation in 2D and 3D, which is why elbow
    and knee need only one implementation.
    """
    ba, bc = a - b, c - b
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na < _EPS or nc < _EPS:
        return float("nan")
    cosine = float(np.dot(ba, bc)) / (na * nc)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _signed_from_down(vec: np.ndarray, down: np.ndarray, toward: np.ndarray) -> float:
    """Angle of `vec` away from `down`, positive toward `toward`.

    Anatomical neutral - a limb hanging at rest, parallel to `down` - is
    exactly 0, which is what the chart requires. The raw interior angle would
    give 180 for the same pose.
    """
    return float(np.degrees(np.arctan2(float(np.dot(vec, toward)), float(np.dot(vec, down)))))


# --- hinge joints: one implementation serves both representations ------------


def elbow_flexion(points: np.ndarray, side: Side) -> float:
    """Chart: flexion 0-150. A straight arm is 0, not 180. Unsigned."""
    c = CHAIN[side]
    return 180.0 - interior_angle(
        points[c["shoulder"]], points[c["elbow"]], points[c["wrist"]]
    )


def knee_flexion(points: np.ndarray, side: Side) -> float:
    """Chart: flexion 0-135. A straight leg is 0. Unsigned."""
    c = CHAIN[side]
    return 180.0 - interior_angle(points[c["hip"]], points[c["knee"]], points[c["ankle"]])


# --- plane-projected joints: 2D ----------------------------------------------


def shoulder_flexion_2d(points: np.ndarray, side: Side, frame: ImageFrame) -> float:
    """Sagittal. Positive is flexion (forward), negative extension (backward).

    Valid only while the subject is side-on, because that is when the image
    plane IS the sagittal plane (FINDINGS.md F20, F27).
    """
    c = CHAIN[side]
    humerus = points[c["elbow"]] - points[c["shoulder"]]
    return _signed_from_down(humerus, frame.down, frame.anterior)


def shoulder_abduction_2d(points: np.ndarray, side: Side, frame: ImageFrame) -> float:
    """Frontal. Positive is abduction (away from the midline).

    Requires the OPPOSITE orientation from every other measurement: the subject
    must face the camera. One fixed camera cannot serve both planes (F34).
    """
    c = CHAIN[side]
    humerus = points[c["elbow"]] - points[c["shoulder"]]
    return _signed_from_down(humerus, frame.down, frame.lateral(side))


def hip_flexion_2d(points: np.ndarray, side: Side, frame: ImageFrame) -> float:
    """Sagittal. Positive is flexion (forward), negative extension."""
    c = CHAIN[side]
    femur = points[c["knee"]] - points[c["hip"]]
    return _signed_from_down(femur, frame.down, frame.anterior)


def ankle_angle(points: np.ndarray, side: Side) -> float:
    """Chart: dorsiflexion 0-20, plantarflexion 0-50. Neutral is ~90, not 180.

    Positive is dorsiflexion (toes toward the shin), negative plantarflexion.

    The foot is defined heel to toe, not ankle to toe: the latter carries a
    constant -16.7 degree bias and 18% more noise, because the ankle joint sits
    above and forward of the heel and so is not on the foot's long axis (E3).
    """
    c = CHAIN[side]
    shank = points[c["knee"]] - points[c["ankle"]]
    foot = points[c["foot"]] - points[c["heel"]]
    na, nc = float(np.linalg.norm(shank)), float(np.linalg.norm(foot))
    if na < _EPS or nc < _EPS:
        return float("nan")
    cosine = float(np.dot(shank, foot)) / (na * nc)
    return 90.0 - float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# --- plane-projected joints: 3D ----------------------------------------------


def shoulder_flexion_3d(points: np.ndarray, side: Side, frame: SpatialFrame) -> float:
    """Sagittal, from metric world coordinates. Orientation-independent."""
    c = CHAIN[side]
    humerus = points[c["elbow"]] - points[c["shoulder"]]
    return _signed_from_down(humerus, -frame.up, frame.anterior)


def shoulder_abduction_3d(points: np.ndarray, side: Side, frame: SpatialFrame) -> float:
    """Frontal, from metric world coordinates."""
    c = CHAIN[side]
    humerus = points[c["elbow"]] - points[c["shoulder"]]
    lateral = frame.left if side is Side.LEFT else -frame.left
    return _signed_from_down(humerus, -frame.up, lateral)


def hip_flexion_3d(points: np.ndarray, side: Side, frame: SpatialFrame) -> float:
    """Sagittal, from metric world coordinates."""
    c = CHAIN[side]
    femur = points[c["knee"]] - points[c["hip"]]
    return _signed_from_down(femur, -frame.up, frame.anterior)


# --- the twelve -------------------------------------------------------------


def compute_angles(
    landmarks: LandmarkSet, anterior_sign: float | None
) -> dict[tuple[MeasurementName, Side], float]:
    """Every measurement the application reports, in image coordinates.

    Args:
        anterior_sign: which way the subject faces, from validity/orientation.
            None means it has not been established - the signed sagittal
            measurements then return NaN rather than a guess, because the wrong
            sign turns flexion into extension.

    Returns NaN for any measurement that cannot be computed. NaN is used rather
    than omission so the result always has twelve entries and the display never
    has to reason about missing keys.
    """
    points = landmarks.image_xy
    frame = image_frame(points, anterior_sign) if anterior_sign is not None else None
    out: dict[tuple[MeasurementName, Side], float] = {}

    for side in (Side.LEFT, Side.RIGHT):
        out[(MeasurementName.ELBOW_FLEXION, side)] = elbow_flexion(points, side)
        out[(MeasurementName.KNEE_FLEXION, side)] = knee_flexion(points, side)
        out[(MeasurementName.ANKLE_ANGLE, side)] = ankle_angle(points, side)

        if frame is None:
            nan = float("nan")
            out[(MeasurementName.SHOULDER_FLEXION, side)] = nan
            out[(MeasurementName.SHOULDER_ABDUCTION, side)] = nan
            out[(MeasurementName.HIP_FLEXION, side)] = nan
            continue

        out[(MeasurementName.SHOULDER_FLEXION, side)] = shoulder_flexion_2d(points, side, frame)
        out[(MeasurementName.SHOULDER_ABDUCTION, side)] = shoulder_abduction_2d(
            points, side, frame
        )
        out[(MeasurementName.HIP_FLEXION, side)] = hip_flexion_2d(points, side, frame)

    return out


def compute_angles_3d(
    landmarks: LandmarkSet,
) -> dict[tuple[MeasurementName, Side], float]:
    """The same twelve from world coordinates, for cross-checking.

    Not the primary path - see the module docstring - but kept because the
    orientation estimate uses world coordinates and because comparing the two
    representations is how the 23% compression was found in the first place.
    """
    points = landmarks.world_xyz
    frame = spatial_frame(points)
    out: dict[tuple[MeasurementName, Side], float] = {}

    for side in (Side.LEFT, Side.RIGHT):
        out[(MeasurementName.ELBOW_FLEXION, side)] = elbow_flexion(points, side)
        out[(MeasurementName.KNEE_FLEXION, side)] = knee_flexion(points, side)
        out[(MeasurementName.ANKLE_ANGLE, side)] = ankle_angle(points, side)

        if frame is None:
            nan = float("nan")
            out[(MeasurementName.SHOULDER_FLEXION, side)] = nan
            out[(MeasurementName.SHOULDER_ABDUCTION, side)] = nan
            out[(MeasurementName.HIP_FLEXION, side)] = nan
            continue

        out[(MeasurementName.SHOULDER_FLEXION, side)] = shoulder_flexion_3d(points, side, frame)
        out[(MeasurementName.SHOULDER_ABDUCTION, side)] = shoulder_abduction_3d(
            points, side, frame
        )
        out[(MeasurementName.HIP_FLEXION, side)] = hip_flexion_3d(points, side, frame)

    return out
