"""Which way the subject is facing, decided over time rather than per frame.

Owns: the policy for combining the three facing cues.
Does NOT own: the cues themselves, which are geometry and live in
biomechanics/frame.py.

This sets the SIGN of every signed sagittal measurement, so getting it wrong
turns flexion into extension - a factor-of-two error that looks entirely
plausible on screen and would never be noticed.

No single cue is reliable, and **majority voting makes it worse**. Consistency
within each recorded phase (FINDINGS.md F35):

    phase       nose   ear->nose   heel->toe   majority   unanimous-hold
    cam_high    0.67      0.54        1.00       0.51         1.00
    cam_low     0.81      0.73        1.00       0.90         1.00
    seated      1.00      1.00        0.73       1.00         1.00
    sh_180      0.55      1.00        0.79       0.80         1.00

Nose and ear are both head-based and fail together with camera elevation,
outvoting the foot cue that remains correct. Majority voting assumes
independent failures; these are correlated. So ambiguous frames are **discarded
rather than averaged**, and the previous confident answer is held.
"""

from __future__ import annotations

import numpy as np

from ..biomechanics.frame import anterior_cues


class AnteriorTracker:
    """Holds the facing direction, updating only when all three cues agree."""

    __slots__ = ("_direction", "_unanimous_frames", "_ambiguous_frames")

    def __init__(self) -> None:
        self._direction: float | None = None
        self._unanimous_frames = 0
        self._ambiguous_frames = 0

    def update(self, image_xy: np.ndarray) -> float | None:
        """Feed one frame's landmarks; get the current facing direction.

        Returns +1 or -1 once established, or None if it never has been - in
        which case the signed sagittal measurements must be withheld rather
        than guessed. That happens in 100% of frames when the legs are out of
        view and 46% at long range (F35), so it is a real state, not an edge
        case.
        """
        nose, ear, foot = anterior_cues(image_xy)
        if nose == ear == foot:
            self._direction = nose
            self._unanimous_frames += 1
        else:
            self._ambiguous_frames += 1
        return self._direction

    @property
    def is_established(self) -> bool:
        return self._direction is not None

    @property
    def confidence_note(self) -> str:
        """One line for the health panel, so a held value is visible as held."""
        if self._direction is None:
            return "facing unknown"
        total = self._unanimous_frames + self._ambiguous_frames
        agreed = 100.0 * self._unanimous_frames / total if total else 0.0
        return f"facing {'fwd' if self._direction > 0 else 'rev'} ({agreed:.0f}% agreed)"
