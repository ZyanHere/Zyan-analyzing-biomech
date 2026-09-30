"""Composing the trust signals into a verdict per measurement.

Owns: which signals may reject a measurement, and the reason given when one
does.
Does NOT own: computing the signals (signals.py) or any angle.

The composition rule that matters:

    a VETO-capable signal that cannot be computed  ->  blocks the measurement
    an ADVISORY signal that cannot be computed     ->  is omitted

That asymmetry is the whole design. Bone-length asymmetry is bilateral, so it
is uncomputable whenever one side is hidden - and the brief says to compute each
side whenever *that side's* landmarks are reliable. Giving it a veto would
reject a perfectly visible left knee because the right leg was occluded, which
is stricter than required. Measured across eleven recorded failure phases it
also adds no unique coverage and false-positives on camera elevation, so it
warns rather than rejects (FINDINGS.md F37).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..biomechanics.conventions import REQUIRED_PLANE
from ..config import ValidityConfig
from ..types import LandmarkSet, MeasurementName, Plane, Side, Verdict
from .orientation import OrientationState
from .signals import BoneLengthTracker, frame_visibility, min_visibility


@dataclass(frozen=True, slots=True)
class ValidityInputs:
    """Everything a verdict depends on, gathered once per frame.

    Passed as one object so the judge cannot quietly acquire a dependency: if
    it needs something new, it has to appear here first.
    """

    landmarks: LandmarkSet | None
    orientation: OrientationState
    bones: BoneLengthTracker
    anterior_known: bool


class ValidityJudge:
    """Decides, per measurement, whether to report a number or a reason."""

    def __init__(self, cfg: ValidityConfig) -> None:
        self._cfg = cfg

    def judge(
        self, inputs: ValidityInputs, name: MeasurementName, side: Side
    ) -> Verdict:
        """The verdict for one measurement, with the reason if rejected.

        Checks run cheapest-first and in the order a user can act on: there is
        no point saying "left leg occluded" to someone who is not in frame.
        """
        if inputs.landmarks is None:
            return Verdict.rejected("no person detected")

        if not inputs.orientation.supports(name):
            # Names what THIS measurement needs, not what the subject is
            # currently doing. "side-on: flexion measurable" is a status;
            # "face the camera" is something a user can act on.
            return Verdict.rejected(_PLANE_INSTRUCTION[REQUIRED_PLANE[name]])

        if _is_signed(name):
            # Only the signed measurements use the anatomical frame. Elbow and
            # knee are interior angles of three points, so a hip landmark being
            # uncertain has no bearing on them - rejecting them for it would be
            # stricter than the geometry requires.
            if frame_visibility(inputs.landmarks) < self._cfg.min_visibility:
                return Verdict.rejected("torso not fully visible")
            if not inputs.anterior_known:
                return Verdict.rejected("facing direction unknown")

        visibility = min_visibility(inputs.landmarks, name, side)
        if visibility < self._cfg.min_visibility:
            return Verdict.rejected(f"{_limb_of(name)} not clearly visible")

        variance = inputs.bones.worst_variance_pct(name, side)
        if math.isnan(variance):
            return Verdict.rejected("gathering data")
        if variance > self._cfg.max_bone_cv_pct:
            return Verdict.rejected("unstable tracking")

        return Verdict.ok(advisory=self._advisory(inputs, name))

    def _advisory(self, inputs: ValidityInputs, name: MeasurementName) -> str:
        """A warning that accompanies a value rather than replacing it."""
        asymmetry = inputs.bones.worst_asymmetry_pct(name)
        if math.isnan(asymmetry):
            return ""
        if asymmetry > self._cfg.advisory_bone_asymmetry_pct:
            return f"L/R mismatch {asymmetry:.0f}%"
        return ""

    def judge_all(self, inputs: ValidityInputs) -> dict[tuple[MeasurementName, Side], Verdict]:
        """Verdicts for all twelve, so the display never reasons about gaps."""
        return {
            (name, side): self.judge(inputs, name, side)
            for name in MeasurementName
            for side in (Side.LEFT, Side.RIGHT)
        }


# Measurements whose sign depends on knowing which way the subject faces.
# Hinge joints are unsigned interior angles and need no facing direction.
_SIGNED = frozenset(
    {
        MeasurementName.SHOULDER_FLEXION,
        MeasurementName.SHOULDER_ABDUCTION,
        MeasurementName.HIP_FLEXION,
        MeasurementName.ANKLE_ANGLE,
    }
)

# What the subject must do for a measurement in each plane to become valid.
_PLANE_INSTRUCTION = {
    Plane.SAGITTAL: "turn side-on",
    Plane.FRONTAL: "face the camera",
}

_LIMB = {
    MeasurementName.ELBOW_FLEXION: "arm",
    MeasurementName.SHOULDER_FLEXION: "arm",
    MeasurementName.SHOULDER_ABDUCTION: "arm",
    MeasurementName.KNEE_FLEXION: "leg",
    MeasurementName.HIP_FLEXION: "leg",
    MeasurementName.ANKLE_ANGLE: "foot",
}


def _is_signed(name: MeasurementName) -> bool:
    return name in _SIGNED


def _limb_of(name: MeasurementName) -> str:
    return _LIMB[name]
