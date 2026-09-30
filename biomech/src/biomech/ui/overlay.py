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

import math

import cv2
import numpy as np

from ..biomechanics.conventions import REQUIRED_PLANE, present
from ..types import LandmarkSet, MeasurementName, Plane, Side
from ..validity.orientation import OrientationState

# Fixed display order: never reordered, so a value stays in the same place as
# the subject moves and the eye can find it without reading.
_ROWS: tuple[tuple[MeasurementName, Side], ...] = tuple(
    (name, side)
    for name in (
        MeasurementName.ELBOW_FLEXION,
        MeasurementName.KNEE_FLEXION,
        MeasurementName.SHOULDER_FLEXION,
        MeasurementName.SHOULDER_ABDUCTION,
        MeasurementName.HIP_FLEXION,
        MeasurementName.ANKLE_ANGLE,
    )
    for side in (Side.LEFT, Side.RIGHT)
)

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
_OUT_OF_RANGE = (80, 170, 255)

# What the subject should do to make a measurement available again.
_PLANE_HINT = {
    Plane.SAGITTAL: "-- turn side-on",
    Plane.FRONTAL: "-- face the camera",
}


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


def draw_measurements(
    image: np.ndarray,
    angles: dict[tuple[MeasurementName, Side], float],
    orientation: OrientationState | None = None,
) -> None:
    """List every measurement, always, in a fixed order.

    All twelve are shown whether or not they currently have a value. Rows that
    appear and vanish as the subject turns read as bugs; a permanent list makes
    the constraint visible instead of hiding it - and it is the direct
    implementation of the brief's requirement to indicate that state rather
    than display a misleading value.

    A row whose plane is not presented shows the instruction that would fix it,
    never a number. At most ten of the twelve can be live at once, because
    abduction's plane excludes the other five (FINDINGS.md F34).
    """
    h, w = image.shape[:2]
    x = w - 330
    cv2.rectangle(image, (x - 14, 0), (w, 34 * len(_ROWS) + 22), _PANEL_BG, -1)

    for i, (name, side) in enumerate(_ROWS):
        supported = orientation is None or orientation.supports(name)
        if supported:
            shown = present(name, angles.get((name, side), float("nan")))
            text = str(shown)
            colour = _TEXT if not math.isnan(shown.magnitude_deg) else _UNCERTAIN
            if not math.isnan(shown.magnitude_deg) and not shown.within_normal_range:
                colour = _OUT_OF_RANGE
        else:
            text = _PLANE_HINT[REQUIRED_PLANE[name]]
            colour = _UNCERTAIN

        label = f"{name.value.replace('_', ' ')} {side.value}"
        cv2.putText(image, label, (x, 26 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, _TEXT, 1, cv2.LINE_AA)
        cv2.putText(image, text, (x, 44 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)


def draw_orientation(image: np.ndarray, orientation: OrientationState) -> None:
    """Show the current plane and live yaw, so the constraint is learnable.

    The subject can see what turning does to the measurements, rather than
    discovering by trial that half of them stopped working.
    """
    h, w = image.shape[:2]
    colour = _CONFIDENT if orientation.plane is not None else _UNCERTAIN
    yaw = "--" if math.isnan(orientation.yaw_deg) else f"{orientation.yaw_deg:.0f} deg"
    cv2.rectangle(image, (0, h - 46), (430, h), _PANEL_BG, -1)
    cv2.putText(image, f"rotation {yaw}  |  {orientation.guidance}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)


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
