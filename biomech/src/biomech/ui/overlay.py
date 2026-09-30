"""Drawing onto the video frame.

Owns: the skeleton overlay and the health panel.
Does NOT own: any computation. It receives values and draws them; it never
decides whether a measurement is valid or what an angle is.

Phase 3 scope: enough to see that tracking works and what the pipeline costs.
The measurement panel arrives in phase 8.

Skeleton colour encodes **landmark visibility**, not measurement validity.
Those are different questions - an elbow landmark can be perfectly visible
while shoulder flexion is invalid because the subject faces the wrong way - and
conflating them in one colour would mislead.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import LandmarkSet

# Drawn as (proximal, distal) pairs. Face landmarks are omitted: they carry no
# measurement and clutter the overlay.
_BONES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 23), (12, 24), (23, 24),        # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # arms
    (23, 25), (25, 27), (24, 26), (26, 28),        # legs
    (27, 29), (29, 31), (28, 30), (30, 32),        # feet
)

_CONFIDENT = (0, 235, 0)
_UNCERTAIN = (0, 150, 255)
_PANEL_BG = (0, 0, 0)
_TEXT = (235, 235, 235)


def draw_skeleton(
    image: np.ndarray,
    landmarks: LandmarkSet,
    scale: tuple[float, float],
    min_visibility: float,
) -> None:
    """Draw bones and joints, coloured by the model's own confidence.

    Mutates `image` in place, which is the one place this module writes
    anything: it owns the pixels it was handed, and nothing else.
    """
    sx, sy = scale
    points = [(int(x * sx), int(y * sy)) for x, y in landmarks.image_xy]

    for a, b in _BONES:
        confident = min(landmarks.visibility[a], landmarks.visibility[b]) >= min_visibility
        cv2.line(image, points[a], points[b], _CONFIDENT if confident else _UNCERTAIN, 2)

    for idx, point in enumerate(points):
        if idx < 11:  # skip the face
            continue
        confident = landmarks.visibility[idx] >= min_visibility
        cv2.circle(image, point, 4, _CONFIDENT if confident else _UNCERTAIN, -1)


def draw_health(image: np.ndarray, stats: dict[str, float], source_note: str) -> None:
    """Draw the pipeline health panel.

    On screen rather than in a log because the pipeline's behaviour under load
    is part of the result, not a footnote to it.
    """
    lines = [
        f"{stats['displayed_fps']:5.1f} FPS displayed",
        f"infer   p50 {stats['inference_p50_ms']:5.1f}  p95 {stats['inference_p95_ms']:5.1f} ms",
        f"render  p50 {stats['render_p50_ms']:5.1f}  p95 {stats['render_p95_ms']:5.1f} ms",
        f"end-end p50 {stats['end_to_end_p50_ms']:5.1f}  p95 {stats['end_to_end_p95_ms']:5.1f} ms",
        f"drops {stats['source_drops']:.0f}   no person {stats['frames_without_person']:.0f}",
        source_note,
    ]
    height = 22 * len(lines) + 16
    cv2.rectangle(image, (0, 0), (430, height), _PANEL_BG, -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (12, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, _TEXT, 1, cv2.LINE_AA)


def draw_banner(image: np.ndarray, message: str) -> None:
    """A full-width warning strip, for conditions the user must act on.

    Used when the source is unhealthy: a USB camera that stops delivering
    frames commonly recovers on its own, so the application says so and keeps
    running rather than exiting.
    """
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, h - 54), (w, h), (30, 30, 190), -1)
    cv2.putText(image, message, (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
