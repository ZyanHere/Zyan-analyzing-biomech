"""The anatomical reference frame.

Owns: constructing body-relative axes from landmarks, in both 2D and 3D.
Does NOT own: any joint angle, and no judgement about trustworthiness.

The frame is rebuilt from the landmarks on **every frame**, never cached and
never taken from a fixed world axis. The model's world coordinates are
camera-aligned rather than body-locked: the hip vector swings 34 degrees mean
and 138 degrees maximum as the subject turns (FINDINGS.md F18). Projecting onto
a fixed world plane and calling it sagittal would therefore be wrong the moment
anyone moved.

Anterior direction is split deliberately. The *geometry* - what the three cues
say on this frame - lives here, because "which way is the body facing" is part
of constructing the frame. The *policy* for combining them over time lives in
validity/orientation.py, because deciding what to trust is a different question
from computing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import (
    LEFT_EAR,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Side,
)

_EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalise, returning the input unchanged if it is degenerate."""
    norm = float(np.linalg.norm(v))
    return v / norm if norm > _EPS else v


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """Body-relative axes in 3D (metric world coordinates).

    Right-handed and verified against recorded data: MediaPipe's world frame
    has up = -y, the subject's left = +x, anterior = -z (FINDINGS.md F18).
    Constructing from landmarks rather than assuming those signs means the code
    survives a change of model.
    """

    up: np.ndarray
    left: np.ndarray
    anterior: np.ndarray


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Body-relative axes within the image plane.

    `down` points from the shoulders toward the hips, so a limb hanging at rest
    is parallel to it and reads as 0 - the chart's anatomical neutral.

    `anterior` is the in-image direction the subject faces. It is supplied
    rather than inferred here, because getting it wrong inverts the sign of
    every sagittal measurement and turns flexion into extension.

    `left_axis` points toward the subject's own left, taken from the hips. It
    cannot be derived from `down` alone: facing the camera, the subject's left
    side appears on the image *right*, and which perpendicular corresponds to
    "their left" flips when they turn around. Reading it from the landmarks
    makes the frame correct from any viewpoint.
    """

    down: np.ndarray
    anterior: np.ndarray
    left_axis: np.ndarray

    def lateral(self, side: Side) -> np.ndarray:
        """Direction away from the midline for one side.

        Flipped for the right side so that positive abduction means "away from
        the body" on both sides, as the chart defines it.
        """
        return self.left_axis if side is Side.LEFT else -self.left_axis


def spatial_frame(world_xyz: np.ndarray) -> SpatialFrame | None:
    """Build the 3D anatomical frame, or None if the torso is degenerate."""
    mid_hip = (world_xyz[LEFT_HIP] + world_xyz[RIGHT_HIP]) / 2.0
    mid_shoulder = (world_xyz[LEFT_SHOULDER] + world_xyz[RIGHT_SHOULDER]) / 2.0

    up = _unit(mid_shoulder - mid_hip)
    raw_left = world_xyz[LEFT_HIP] - world_xyz[RIGHT_HIP]
    # Orthogonalised so the frame stays right-angled when the subject leans.
    left = _unit(raw_left - np.dot(raw_left, up) * up)
    if np.linalg.norm(up) < _EPS or np.linalg.norm(left) < _EPS:
        return None
    return SpatialFrame(up=up, left=left, anterior=_unit(np.cross(left, up)))


def image_frame(image_xy: np.ndarray, anterior_sign: float) -> ImageFrame | None:
    """Build the in-image frame given which way the subject faces.

    Args:
        anterior_sign: +1 or -1, selecting which perpendicular to `down` points
            forward. Produced by validity/orientation.py from the cues below.
    """
    mid_hip = (image_xy[LEFT_HIP] + image_xy[RIGHT_HIP]) / 2.0
    mid_shoulder = (image_xy[LEFT_SHOULDER] + image_xy[RIGHT_SHOULDER]) / 2.0

    down = mid_hip - mid_shoulder
    if float(np.linalg.norm(down)) < _EPS:
        return None
    down = _unit(down)
    perp = np.array([-down[1], down[0]])
    anterior = perp * float(np.sign(anterior_sign) or 1.0)

    # Taken from the hips, not from `down`: when the subject is side-on the
    # hips foreshorten to nearly nothing, so this degenerates - which is
    # correct, because abduction is not measurable in that orientation anyway.
    raw_left = image_xy[LEFT_HIP] - image_xy[RIGHT_HIP]
    left_axis = _unit(raw_left - np.dot(raw_left, down) * down)

    return ImageFrame(down=down, anterior=anterior, left_axis=left_axis)


def anterior_cues(image_xy: np.ndarray) -> tuple[float, float, float]:
    """Three independent votes on which in-image direction is forward.

    Returned separately rather than combined, because they fail in *correlated*
    ways: nose and ear are both head-based and degrade together with camera
    elevation, so a majority vote scored 0.51 where the best single cue scored
    1.00 (FINDINGS.md F35). Combining them is a policy decision that belongs
    with the code that can also remember previous frames.

    Each value is +1 or -1 against the same perpendicular.
    """
    mid_hip = (image_xy[LEFT_HIP] + image_xy[RIGHT_HIP]) / 2.0
    mid_shoulder = (image_xy[LEFT_SHOULDER] + image_xy[RIGHT_SHOULDER]) / 2.0
    down = _unit(mid_hip - mid_shoulder)
    perp = np.array([-down[1], down[0]])

    ear_mid = (image_xy[LEFT_EAR] + image_xy[RIGHT_EAR]) / 2.0
    toes = (image_xy[LEFT_FOOT_INDEX] - image_xy[LEFT_HEEL]) + (
        image_xy[RIGHT_FOOT_INDEX] - image_xy[RIGHT_HEEL]
    )

    return (
        _vote(perp, image_xy[NOSE] - mid_shoulder),
        _vote(perp, image_xy[NOSE] - ear_mid),
        _vote(perp, toes / 2.0),
    )


def _vote(perp: np.ndarray, reference: np.ndarray) -> float:
    return 1.0 if float(np.dot(perp, reference)) > 0.0 else -1.0
