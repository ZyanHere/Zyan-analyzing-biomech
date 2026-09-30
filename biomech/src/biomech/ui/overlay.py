"""Drawing onto the video frame itself.

Owns: the skeleton overlay and the warning banner.
Does NOT own: the text panels (panel.py), layout (layout.py), or any
computation. It receives values and draws them.

Skeleton colour encodes **landmark visibility**, not measurement validity.
Those are different questions: an elbow landmark can be perfectly visible while
shoulder flexion is invalid because the subject is facing the wrong way.
Colouring the skeleton by measurement validity would conflate the two and
mislead - so the skeleton answers "does the model see this joint" and the panel
answers "can this measurement be trusted".
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import LandmarkSet

# Drawn as (proximal, distal) pairs. Face landmarks carry no measurement and
# clutter the overlay, so they are omitted.
_BONES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 23), (12, 24), (23, 24),        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # arms
    (23, 25), (25, 27), (24, 26), (26, 28),        # legs
    (27, 29), (29, 31), (28, 30), (30, 32),        # feet
)

_CONFIDENT = (0, 220, 0)
_UNCERTAIN = (80, 170, 255)
_JOINT_EDGE = (20, 20, 20)
_FIRST_BODY_LANDMARK = 11


def draw_skeleton(image: np.ndarray, landmarks: LandmarkSet, min_visibility: float) -> None:
    """Draw bones and joints, coloured by the model's own confidence.

    Mutates `image` in place - the one place this module writes anything. It
    owns the pixels it was handed and nothing else.
    """
    points = [(int(x), int(y)) for x, y in landmarks.image_xy]

    for a, b in _BONES:
        confident = min(landmarks.visibility[a], landmarks.visibility[b]) >= min_visibility
        cv2.line(image, points[a], points[b],
                 _CONFIDENT if confident else _UNCERTAIN, 2, cv2.LINE_AA)

    for idx in range(_FIRST_BODY_LANDMARK, len(points)):
        confident = landmarks.visibility[idx] >= min_visibility
        cv2.circle(image, points[idx], 4, _CONFIDENT if confident else _UNCERTAIN, -1)
        cv2.circle(image, points[idx], 4, _JOINT_EDGE, 1, cv2.LINE_AA)


def draw_banner(image: np.ndarray, message: str) -> None:
    """A full-width strip for a condition the user must act on.

    Used when the source is unhealthy or nobody is in frame. A USB camera that
    stops delivering frames commonly recovers on its own, so the application
    says so and keeps running rather than exiting.
    """
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, height - 40), (width, height), (25, 25, 170), -1)
    cv2.putText(image, message, (14, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
