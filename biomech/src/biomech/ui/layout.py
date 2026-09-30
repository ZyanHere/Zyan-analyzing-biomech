"""Composing the output frame.

Owns: where each element sits, and the buffer it is drawn into.
Does NOT own: what any element contains.

Two decisions here are about cost rather than appearance, because rendering
shares a thread with inference (FINDINGS.md F36) and every millisecond spent
here is taken directly from it:

  native resolution   An earlier version resized 640x480 to 1100x825 and drew
                      text on the enlarged canvas - ~907k pixels of work per
                      frame before any drawing, to produce something the window
                      manager scales for free.
  one allocation      The canvas is allocated once and drawn into. Building
                      fresh panels and hstacking them each frame measured
                      *slower* than the resize it replaced: 3.6 -> 4.8 ms.

    +-------------------------+----------------+
    |                         |                |
    |      video + skeleton   |  measurements  |
    |                         |  (12 rows)     |
    +-------------------------+                |
    |   health / orientation  |                |
    +-------------------------+----------------+
"""

from __future__ import annotations

import numpy as np

from ..types import LandmarkSet, MeasurementName, Side, Verdict
from ..validity.orientation import OrientationState
from .overlay import draw_banner, draw_skeleton
from .panel import HEALTH_HEIGHT, PANEL_WIDTH, health_panel, measurement_panel


def canvas_size(video_width: int, video_height: int) -> tuple[int, int]:
    """The composed frame's dimensions, for sizing the window at startup."""
    return video_width + PANEL_WIDTH, video_height + HEALTH_HEIGHT


class FrameComposer:
    """Builds output frames into a reused buffer.

    Holds the canvas for the lifetime of the run. The three regions are numpy
    views, so writing to them writes to the canvas with no copying.
    """

    def __init__(self, video_width: int, video_height: int) -> None:
        width, height = canvas_size(video_width, video_height)
        self._canvas = np.zeros((height, width, 3), np.uint8)
        self._video = self._canvas[:video_height, :video_width]
        self._health = self._canvas[video_height:, :video_width]
        self._panel = self._canvas[:, video_width:]

    def compose(
        self,
        image: np.ndarray,
        landmarks: LandmarkSet | None,
        angles: dict[tuple[MeasurementName, Side], float],
        verdicts: dict[tuple[MeasurementName, Side], Verdict],
        orientation: OrientationState,
        stats: dict[str, float],
        source_note: str,
        facing_note: str,
        min_visibility: float,
        draw_pose: bool = True,
        banner: str = "",
    ) -> np.ndarray:
        """Draw one frame. The returned array is the reused canvas.

        The caller must not retain it across frames - it is overwritten in
        place, which is the point.
        """
        # Copied in rather than drawn on: the source frame belongs to the
        # pipeline and may also be handed to the recorder.
        self._video[:] = image
        if landmarks is not None and draw_pose:
            draw_skeleton(self._video, landmarks, min_visibility)
        if banner:
            draw_banner(self._video, banner)

        health_panel(self._health, stats, source_note, orientation, facing_note)
        measurement_panel(self._panel, angles, verdicts)
        return self._canvas
