"""A skeleton posed at exactly known joint angles.

Test infrastructure, not application code, which is why it lives here rather
than in src/. Its purpose is to make the angle mathematics falsifiable
independently of any model: if a measurement disagrees with the angle that was
used to build the pose, the arithmetic is wrong, and no pose estimator was
involved to blame.

Built in a clean right-handed frame - up = +Y, the subject's left = +X,
anterior = +Z - so that cross(left, up) = anterior, matching what the real
world frame was measured to do (FINDINGS.md F18).

Sagittal rotations use axis = -left, except the knee, which flexes backwards
and therefore uses +left. That asymmetry is anatomy, not a sign error.
"""

from __future__ import annotations

import numpy as np

from biomech.types import LANDMARK_COUNT, LandmarkSet

UP = np.array([0.0, 1.0, 0.0])
LEFT = np.array([1.0, 0.0, 0.0])
ANT = np.array([0.0, 0.0, 1.0])
DOWN = -UP

SEGMENTS = {
    "trunk": 0.50, "shoulder_half": 0.18, "hip_half": 0.15,
    "upper_arm": 0.30, "forearm": 0.26, "femur": 0.42, "shank": 0.42,
    "heel_back": 0.05, "foot_forward": 0.20, "foot_drop": 0.06,
}


def rotate(v: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    return (
        v * np.cos(theta)
        + np.cross(a, v) * np.sin(theta)
        + a * np.dot(a, v) * (1.0 - np.cos(theta))
    )


def build(
    *,
    shoulder_flexion: float = 0.0,
    shoulder_abduction: float = 0.0,
    elbow: float = 0.0,
    hip_flexion: float = 0.0,
    knee: float = 0.0,
    ankle: float = 0.0,
    side: str = "L",
    body_yaw: float = 0.0,
) -> np.ndarray:
    """Return (33, 3) world points for a pose with exactly these angles.

    Angles are in the chart's convention: shoulder_flexion positive forward,
    abduction positive away from the midline, knee positive flexed (heel back),
    ankle positive dorsiflexion.
    """
    lm = np.zeros((LANDMARK_COUNT, 3))
    sign = 1.0 if side == "L" else -1.0
    sagittal = -LEFT
    abduction_axis = ANT * sign

    mid_hip = np.zeros(3)
    mid_shoulder = mid_hip + UP * SEGMENTS["trunk"]
    lm[23] = mid_hip + LEFT * SEGMENTS["hip_half"]
    lm[24] = mid_hip - LEFT * SEGMENTS["hip_half"]
    lm[11] = mid_shoulder + LEFT * SEGMENTS["shoulder_half"]
    lm[12] = mid_shoulder - LEFT * SEGMENTS["shoulder_half"]
    lm[0] = mid_shoulder + UP * 0.25 + ANT * 0.05
    lm[7] = lm[0] - ANT * 0.10 + LEFT * 0.07
    lm[8] = lm[0] - ANT * 0.10 - LEFT * 0.07

    active = (11, 13, 15, 23, 25, 27, 29, 31) if side == "L" else (12, 14, 16, 24, 26, 28, 30, 32)
    sh, el, wr, hp, kn, an, hl, ft = active

    humerus = rotate(rotate(DOWN, sagittal, shoulder_flexion), abduction_axis, shoulder_abduction)
    lm[el] = lm[sh] + humerus * SEGMENTS["upper_arm"]
    lm[wr] = lm[el] + rotate(humerus, sagittal, elbow) * SEGMENTS["forearm"]

    femur = rotate(DOWN, sagittal, hip_flexion)
    lm[kn] = lm[hp] + femur * SEGMENTS["femur"]
    lm[an] = lm[kn] + rotate(femur, LEFT, knee) * SEGMENTS["shank"]

    # The foot hangs off the shank, so it inherits the knee and hip rotations.
    # Applying only the ankle angle would leave it in its global orientation
    # and fabricate a bias whenever the leg is bent.
    heel_offset = -ANT * SEGMENTS["heel_back"] - UP * SEGMENTS["foot_drop"]
    toe_offset = ANT * SEGMENTS["foot_forward"] - UP * SEGMENTS["foot_drop"]

    def to_world(offset: np.ndarray) -> np.ndarray:
        return rotate(rotate(rotate(offset, sagittal, ankle), LEFT, knee), sagittal, hip_flexion)

    lm[hl] = lm[an] + to_world(heel_offset)
    lm[ft] = lm[an] + to_world(toe_offset)

    _mirror_idle_limbs(lm, side, heel_offset, toe_offset)

    if body_yaw:
        lm = np.array([rotate(p, UP, body_yaw) for p in lm])
    return lm


def _mirror_idle_limbs(
    lm: np.ndarray, side: str, heel_offset: np.ndarray, toe_offset: np.ndarray
) -> None:
    """Place the unmeasured limbs at rest, so the anatomical frame is well conditioned."""
    idle = (12, 14, 16, 24, 26, 28, 30, 32) if side == "L" else (11, 13, 15, 23, 25, 27, 29, 31)
    sh, el, wr, hp, kn, an, hl, ft = idle
    lm[el] = lm[sh] + DOWN * SEGMENTS["upper_arm"]
    lm[wr] = lm[el] + DOWN * SEGMENTS["forearm"]
    lm[kn] = lm[hp] + DOWN * SEGMENTS["femur"]
    lm[an] = lm[kn] + DOWN * SEGMENTS["shank"]
    lm[hl] = lm[an] + heel_offset
    lm[ft] = lm[an] + toe_offset


def project(
    world: np.ndarray, focal: float = 600.0, distance: float = 3.0,
    width: int = 640, height: int = 480,
) -> np.ndarray:
    """Pinhole projection to pixels, with y growing downward as in an image."""
    out = np.zeros((LANDMARK_COUNT, 2))
    for i, p in enumerate(world):
        z = max(distance - p[2], 0.1)
        out[i] = [width / 2.0 + focal * p[0] / z, height / 2.0 - focal * p[1] / z]
    return out


def landmark_set(world: np.ndarray, visibility: float = 1.0) -> LandmarkSet:
    """Wrap a synthetic pose as the type the application consumes."""
    return LandmarkSet(
        image_xy=project(world),
        world_xyz=world,
        visibility=np.full(LANDMARK_COUNT, visibility),
    )
