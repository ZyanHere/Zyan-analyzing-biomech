"""The text panels: measurements, health, orientation.

Owns: rendering values into pixel regions.
Does NOT own: any computation, and no decision about whether a value is valid -
it receives verdicts and displays them.

The measurement panel always shows all twelve rows in a fixed order. Rows that
appear and vanish as the subject turns read as bugs; a permanent list with a
reason against each blank makes the constraint visible and learnable. That is
the direct implementation of the brief's requirement to indicate that state
rather than display a misleading value.

At most ten of the twelve can be live at once: abduction's plane excludes the
other five, so the list never goes fully green, and that is the honest picture
(FINDINGS.md F34).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..biomechanics.conventions import present
from ..types import MeasurementName, Side, Verdict
from ..validity.orientation import OrientationState

PANEL_WIDTH = 360
HEALTH_HEIGHT = 96

_BG = (18, 18, 18)
_RULE = (55, 55, 55)
_LABEL = (150, 150, 150)
_VALUE = (240, 240, 240)
_MUTED = (110, 120, 135)
_WARN = (80, 170, 255)
_GOOD = (0, 220, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

ROWS: tuple[tuple[MeasurementName, Side], ...] = tuple(
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

_SHORT = {
    MeasurementName.ELBOW_FLEXION: "elbow",
    MeasurementName.KNEE_FLEXION: "knee",
    MeasurementName.SHOULDER_FLEXION: "shoulder flex",
    MeasurementName.SHOULDER_ABDUCTION: "shoulder abd",
    MeasurementName.HIP_FLEXION: "hip",
    MeasurementName.ANKLE_ANGLE: "ankle",
}


def measurement_panel(
    panel: np.ndarray,
    angles: dict[tuple[MeasurementName, Side], float],
    verdicts: dict[tuple[MeasurementName, Side], Verdict],
) -> None:
    """Render all twelve rows into an existing view.

    Draws into a caller-owned buffer rather than allocating: at 30 FPS a fresh
    panel every frame is allocation and copy work on the same thread as
    inference, and it measurably cost more than it saved.
    """
    height = panel.shape[0]
    panel[:] = _BG
    cv2.putText(panel, "MEASUREMENTS", (14, 24), _FONT, 0.46, _LABEL, 1, cv2.LINE_AA)
    cv2.line(panel, (14, 34), (PANEL_WIDTH - 14, 34), _RULE, 1)

    row_height = max(26, (height - 48) // len(ROWS))
    for i, key in enumerate(ROWS):
        top = 48 + i * row_height
        _draw_row(panel, top, key, angles.get(key, float("nan")), verdicts.get(key))


def _draw_row(
    panel: np.ndarray,
    top: int,
    key: tuple[MeasurementName, Side],
    value_deg: float,
    verdict: Verdict | None,
) -> None:
    """One measurement: its name, and either a value or the reason there isn't one."""
    name, side = key
    label = f"{_SHORT[name]} {side.value}"
    cv2.putText(panel, label, (14, top + 12), _FONT, 0.42, _LABEL, 1, cv2.LINE_AA)

    if verdict is not None and not verdict.is_valid:
        # Never a bare blank. The reason is the requirement, not a courtesy.
        cv2.putText(panel, verdict.reason, (14, top + 30), _FONT, 0.44, _MUTED, 1, cv2.LINE_AA)
        return

    shown = present(name, value_deg)
    colour = _VALUE
    if math.isnan(shown.magnitude_deg):
        colour = _MUTED
    elif not shown.within_normal_range:
        colour = _WARN
    cv2.putText(panel, str(shown), (14, top + 30), _FONT, 0.52, colour, 1, cv2.LINE_AA)

    if verdict is not None and verdict.advisory:
        # An advisory accompanies a value; it never replaces one.
        cv2.putText(panel, verdict.advisory, (196, top + 30), _FONT, 0.38, _WARN, 1, cv2.LINE_AA)


def health_panel(panel: np.ndarray, stats: dict[str, float], source_note: str,
                 orientation: OrientationState, facing_note: str) -> None:
    """Pipeline state, on screen rather than in a log.

    How the pipeline behaves under load is part of the result, not a footnote:
    a frame rate that holds only when nothing is moving is worth knowing about
    while it is happening.
    """
    width = panel.shape[1]
    panel[:] = _BG
    fps = stats["displayed_fps"]
    fps_colour = _GOOD if fps >= 29.0 else _WARN

    cv2.putText(panel, f"{fps:4.1f} FPS", (14, 28), _FONT, 0.62, fps_colour, 1, cv2.LINE_AA)
    cv2.putText(panel,
                f"infer {stats['inference_p50_ms']:.0f}/{stats['inference_p95_ms']:.0f}"
                f"   biomech {stats['biomech_p50_ms']:.1f}"
                f"   render {stats['render_p50_ms']:.1f}"
                f"   e2e {stats['end_to_end_p50_ms']:.0f}/{stats['end_to_end_p95_ms']:.0f} ms",
                (130, 28), _FONT, 0.42, _LABEL, 1, cv2.LINE_AA)

    yaw = "--" if math.isnan(orientation.yaw_deg) else f"{orientation.yaw_deg:.0f} deg"
    plane_colour = _GOOD if orientation.plane is not None else _WARN
    cv2.putText(panel, f"rotation {yaw}  |  {orientation.guidance}",
                (14, 56), _FONT, 0.46, plane_colour, 1, cv2.LINE_AA)
    status = (f"{facing_note}   drops {stats['source_drops']:.0f}"
              f"   no person {stats['frames_without_person']:.0f}")
    cv2.putText(panel, status, (14, 80), _FONT, 0.40, _LABEL, 1, cv2.LINE_AA)

    # Right-aligned beside the status line, measured rather than estimated. The
    # canvas is composed at the video's NATIVE width - 640 px, not the size the
    # window is scaled to - so there is far less room here than the window
    # suggests, and a per-character estimate silently overlapped the two for
    # every frame of a recorded session.
    _draw_right_aligned(panel, source_note, status, width)


def _text_width(text: str, scale: float) -> int:
    """Rendered width in pixels, from the font rather than a guess."""
    return cv2.getTextSize(text, _FONT, scale, 1)[0][0]


def _elide(text: str, available: int, scale: float) -> str:
    """Trim from the FRONT until it fits, keeping the informative tail.

    The tail is the model and running mode; the head is the camera geometry,
    which is already in the startup log. Dropping the end would discard the
    part worth reading on screen.
    """
    if _text_width(text, scale) <= available:
        return text
    for cut in range(1, len(text)):
        candidate = ".." + text[cut:]
        if _text_width(candidate, scale) <= available:
            return candidate
    return ""


def _draw_right_aligned(panel: np.ndarray, text: str, beside: str, width: int) -> None:
    """Draw `text` at the right edge of the status row, never over `beside`."""
    gap = 12
    left_end = 14 + _text_width(beside, 0.40)
    available = width - 14 - left_end - gap
    if available <= 0:
        return
    shown = _elide(text, available, 0.38)
    if not shown:
        return
    cv2.putText(panel, shown, (width - 14 - _text_width(shown, 0.38), 80),
                _FONT, 0.38, _MUTED, 1, cv2.LINE_AA)
