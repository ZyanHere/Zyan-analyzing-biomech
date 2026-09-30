"""Orientation: the yaw estimate and the hysteresis that stops it flickering."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomech.config import OrientationConfig
from biomech.types import MeasurementName, Plane
from biomech.validity.orientation import PlaneGate, yaw_degrees

from ..synthetic import build


def _yaw_of(body_yaw: float) -> float:
    return yaw_degrees(build(side="L", body_yaw=body_yaw))


class TestYawEstimate:
    def test_facing_the_camera_reads_near_zero(self) -> None:
        assert _yaw_of(0.0) == pytest.approx(0.0, abs=2.0)

    def test_side_on_reads_near_ninety(self) -> None:
        assert _yaw_of(90.0) == pytest.approx(90.0, abs=2.0)

    @pytest.mark.parametrize("truth", [0, 15, 30, 45, 60, 75, 90])
    def test_tracks_rotation_across_the_range(self, truth: float) -> None:
        assert _yaw_of(truth) == pytest.approx(truth, abs=2.0)

    def test_symmetric_about_facing(self) -> None:
        """Turning left or right presents the same plane, so the magnitude is
        what matters; the estimator must not distinguish them."""
        assert _yaw_of(60.0) == pytest.approx(_yaw_of(-60.0), abs=1.0)

    def test_unaffected_by_distance(self) -> None:
        """The failure that disqualified the pixel-based estimator: it reported
        0 to 39.7 deg of phantom rotation from the subject moving (F35)."""
        near = build(side="L", body_yaw=55.0)
        far = near * 0.5  # same pose, half the apparent size
        assert yaw_degrees(near) == pytest.approx(yaw_degrees(far), abs=0.5)

    def test_unaffected_by_posture(self) -> None:
        """The failure that disqualified the shoulder/trunk ratio: raising an
        arm moved both terms and produced 50 deg of spread (F36)."""
        arms_down = yaw_degrees(build(side="L", body_yaw=70.0))
        arm_up = yaw_degrees(build(shoulder_flexion=160.0, side="L", body_yaw=70.0))
        seated = yaw_degrees(build(hip_flexion=90.0, knee=90.0, side="L", body_yaw=70.0))
        assert arm_up == pytest.approx(arms_down, abs=2.0)
        assert seated == pytest.approx(arms_down, abs=2.0)

    def test_degenerate_shoulders_give_nan(self) -> None:
        points = np.zeros((33, 3))
        assert math.isnan(yaw_degrees(points))


class TestPlaneGate:
    def _gate(self, **kw: float) -> PlaneGate:
        return PlaneGate(OrientationConfig(**kw))  # type: ignore[arg-type]

    def _settle(self, gate: PlaneGate, yaw: float, frames: int = 10) -> Plane | None:
        for _ in range(frames):
            state = gate.update(yaw)
        return state.plane

    def test_side_on_selects_the_sagittal_plane(self) -> None:
        assert self._settle(self._gate(), 80.0) is Plane.SAGITTAL

    def test_facing_selects_the_frontal_plane(self) -> None:
        assert self._settle(self._gate(), 5.0) is Plane.FRONTAL

    def test_the_gap_between_bands_supports_neither(self) -> None:
        """Deliberate: between 30 and 45 degrees both planes are too far off
        for their measurements to be trusted."""
        assert self._settle(self._gate(), 38.0) is None

    def test_a_change_must_persist_before_it_is_acted_on(self) -> None:
        gate = self._gate()
        self._settle(gate, 80.0)
        for _ in range(4):  # one short of the 5-frame hold
            state = gate.update(5.0)
        assert state.plane is Plane.SAGITTAL, "switched before the hold elapsed"
        state = gate.update(5.0)
        assert state.plane is Plane.FRONTAL

    def test_noise_at_the_boundary_produces_no_flicker(self) -> None:
        """The failure this exists to prevent: yaw noise is ~1 deg sd and real
        postures sit at 44-46 deg, which flipped a bare threshold three times
        inside one recorded phase (F36)."""
        gate = self._gate()
        self._settle(gate, 80.0)

        flips = 0
        previous = Plane.SAGITTAL
        for yaw in (44.8, 45.2, 44.9, 45.1, 44.7, 45.3, 44.6, 45.4) * 4:
            plane = gate.update(yaw).plane
            if plane is not previous:
                flips += 1
                previous = plane
        assert flips == 0, f"gate flickered {flips} times around the threshold"

    def test_a_genuine_turn_still_switches(self) -> None:
        """Hysteresis must resist noise without resisting the user."""
        gate = self._gate()
        self._settle(gate, 80.0)
        assert self._settle(gate, 3.0) is Plane.FRONTAL

    def test_state_is_held_while_no_person_is_detected(self) -> None:
        """A dropout should not reset the orientation to unknown - the subject
        has not moved just because the model lost them for a frame."""
        gate = self._gate()
        self._settle(gate, 80.0)
        state = gate.update(float("nan"))
        assert state.plane is Plane.SAGITTAL
        assert state.settling


class TestMeasurementSupport:
    def test_sagittal_supports_flexion_but_not_abduction(self) -> None:
        gate = PlaneGate(OrientationConfig())
        for _ in range(10):
            state = gate.update(80.0)
        assert state.supports(MeasurementName.ELBOW_FLEXION)
        assert state.supports(MeasurementName.HIP_FLEXION)
        assert not state.supports(MeasurementName.SHOULDER_ABDUCTION)

    def test_frontal_supports_abduction_but_not_flexion(self) -> None:
        """The whole constraint in one assertion: one camera, two planes, and
        they are mutually exclusive (F34)."""
        gate = PlaneGate(OrientationConfig())
        for _ in range(10):
            state = gate.update(5.0)
        assert state.supports(MeasurementName.SHOULDER_ABDUCTION)
        assert not state.supports(MeasurementName.ELBOW_FLEXION)

    def test_guidance_tells_the_subject_what_to_do(self) -> None:
        gate = PlaneGate(OrientationConfig())
        for _ in range(10):
            state = gate.update(38.0)
        assert "turn" in state.guidance.lower()
