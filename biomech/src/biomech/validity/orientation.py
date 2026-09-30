"""Which anatomical plane the subject is presenting to the camera.

Owns: the yaw estimate, the plane bands, and the hysteresis that stops the UI
flickering at a threshold.
Does NOT own: any angle. It reports which measurements *could* be valid, never
what they are.

One fixed camera cannot show all twelve measurements at once. Abduction lives
in the frontal plane and needs the subject facing the camera; the other five
are sagittal and need them side-on. Measured error against body rotation
(FINDINGS.md F34), where 0 is facing and 90 is side-on:

    elbow / knee / shoulder flex / hip / ankle   54-80 deg wrong at 0,  exact at 90
    shoulder abduction                            exact at 0,  90 deg wrong at 90

That is geometry, not a tuning parameter. No filter or model change affects it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..biomechanics.conventions import REQUIRED_PLANE
from ..config import OrientationConfig
from ..types import LEFT_SHOULDER, RIGHT_SHOULDER, MeasurementName, Plane


def yaw_degrees(world_xyz: np.ndarray) -> float:
    """Body rotation from the world-space shoulder axis. 0 facing, 90 side-on.

    No calibration, no reference, no running maximum - which is the point.
    Two earlier estimators were measured and discarded (F35, F36):

        raw shoulder pixels    conflates rotation with distance: 0 to 39.7 deg
                               of phantom rotation from the subject simply
                               walking backwards
        shoulder/trunk ratio   fixes distance but not posture: 50 deg spread at
                               one orientation, because raising an arm moves
                               both terms at once

    This measures a *direction* rather than a length, so neither distance nor
    posture disturbs it: 1.1-6.2 deg across six phases known to be face-on,
    62.3-87.7 deg across five known to be side-on.

    Using 3D here does not contradict choosing 2D for angles. The 23% gain
    compression that disqualified 3D (F28) distorts limb angle *magnitudes*;
    this uses only the direction of the trunk's shoulder axis.
    """
    v = world_xyz[LEFT_SHOULDER] - world_xyz[RIGHT_SHOULDER]
    if float(np.linalg.norm(v)) < 1e-9:
        return float("nan")
    return float(math.degrees(math.atan2(abs(float(v[2])), abs(float(v[0])))))


@dataclass(frozen=True, slots=True)
class OrientationState:
    """What the subject is currently presenting."""

    yaw_deg: float
    plane: Plane | None
    settling: bool

    def supports(self, name: MeasurementName) -> bool:
        """Whether this orientation makes that measurement geometrically valid."""
        return self.plane is not None and REQUIRED_PLANE[name] is self.plane

    @property
    def guidance(self) -> str:
        """What the subject should do, phrased as an instruction."""
        if math.isnan(self.yaw_deg):
            return "no person"
        if self.plane is Plane.SAGITTAL:
            return "side-on: flexion measurable"
        if self.plane is Plane.FRONTAL:
            return "facing: abduction measurable"
        return "turn side-on, or face the camera"


class PlaneGate:
    """Classifies yaw into a plane, with hysteresis.

    A bare threshold chatters. Frame-to-frame yaw noise is about 1 degree, and
    real postures sit at 44-46 degrees - directly on the boundary - which
    produced up to three state flips inside a single recorded phase. On screen
    that is a measurement blinking between a value and "turn side-on".

    Two mechanisms, both needed (F36):
      margin  a state is only left once yaw clears its threshold by 5 degrees,
              so noise around the boundary cannot cross back and forth
      hold    a change must persist 5 frames (~0.25 s) before the UI acts on it

    Together these gave zero flips across every recorded phase.
    """

    def __init__(self, cfg: OrientationConfig) -> None:
        self._cfg = cfg
        self._state: Plane | None = None
        self._candidate: Plane | None = None
        self._streak = 0

    def update(self, yaw_deg: float) -> OrientationState:
        """Feed one frame's yaw, get the settled orientation."""
        if math.isnan(yaw_deg):
            return OrientationState(yaw_deg=yaw_deg, plane=self._state, settling=True)

        observed = self._classify(yaw_deg)
        if observed is self._state:
            self._candidate, self._streak = None, 0
        else:
            self._candidate, self._streak = (
                (observed, self._streak + 1)
                if observed is self._candidate
                else (observed, 1)
            )
            if self._streak >= self._cfg.hysteresis_frames:
                self._state, self._candidate, self._streak = observed, None, 0

        return OrientationState(
            yaw_deg=yaw_deg, plane=self._state, settling=self._streak > 0
        )

    def _classify(self, yaw_deg: float) -> Plane | None:
        """Which plane this yaw indicates, biased toward keeping the current one.

        The margin is applied only when leaving a state, never when entering
        one, so the bands are sticky rather than merely shifted.
        """
        margin = self._cfg.hysteresis_deg
        if self._state is Plane.SAGITTAL and yaw_deg >= self._cfg.sagittal_min_yaw_deg - margin:
            return Plane.SAGITTAL
        if self._state is Plane.FRONTAL and yaw_deg <= self._cfg.frontal_max_yaw_deg + margin:
            return Plane.FRONTAL

        if yaw_deg >= self._cfg.sagittal_min_yaw_deg:
            return Plane.SAGITTAL
        if yaw_deg <= self._cfg.frontal_max_yaw_deg:
            return Plane.FRONTAL
        return None
