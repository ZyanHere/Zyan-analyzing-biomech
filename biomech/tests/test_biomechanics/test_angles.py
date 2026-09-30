"""Exact-truth validation of the angle mathematics.

The pose is built by forward kinematics from known angles, so any disagreement
is arithmetic, with no model involved to blame. This is what lets model error
be attributed honestly later: everything measured here is ours.

Requirement: 0.000000 degrees. Not "close enough".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomech.biomechanics.angles import (
    compute_angles,
    compute_angles_3d,
    elbow_flexion,
    interior_angle,
    knee_flexion,
)
from biomech.biomechanics.frame import anterior_cues, image_frame, spatial_frame
from biomech.types import MeasurementName, Side

from ..synthetic import build, landmark_set, project

EXACT = 1e-6

# Both directions where the chart names two, since extension, adduction and
# plantarflexion are separate named ranges rather than a sign convention.
SWEEPS: dict[MeasurementName, tuple[str, list[float]]] = {
    MeasurementName.ELBOW_FLEXION: ("elbow", [0, 15, 30, 60, 90, 120, 150]),
    MeasurementName.KNEE_FLEXION: ("knee", [0, 15, 30, 60, 90, 120, 135]),
    MeasurementName.SHOULDER_FLEXION: ("shoulder_flexion", [-60, -30, 0, 30, 90, 150, 180]),
    MeasurementName.SHOULDER_ABDUCTION: ("shoulder_abduction", [-45, -20, 0, 30, 90, 180]),
    MeasurementName.HIP_FLEXION: ("hip_flexion", [-30, -15, 0, 20, 60, 90, 120]),
    MeasurementName.ANKLE_ANGLE: ("ankle", [-50, -30, 0, 10, 20]),
}


def _anterior(points_2d: np.ndarray) -> float:
    """The unanimous anterior sign, as the application would obtain it."""
    cues = anterior_cues(points_2d)
    assert cues[0] == cues[1] == cues[2], "synthetic pose should be unambiguous"
    return cues[0]


class TestInteriorAngle:
    def test_straight_limb_is_180(self) -> None:
        a, b, c = np.array([0.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
        assert interior_angle(a, b, c) == pytest.approx(180.0)

    def test_right_angle(self) -> None:
        a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert interior_angle(a, b, c) == pytest.approx(90.0)

    def test_coincident_points_give_nan_not_a_number(self) -> None:
        """A degenerate chain has no angle. Returning 0 would look measured."""
        p = np.array([1.0, 1.0])
        assert math.isnan(interior_angle(p, p, p))


class TestChartConventions:
    def test_straight_arm_reads_zero_not_180(self) -> None:
        """The example the brief gives explicitly."""
        world = build(elbow=0.0, side="L")
        assert elbow_flexion(world, Side.LEFT) == pytest.approx(0.0, abs=EXACT)

    def test_straight_leg_reads_zero(self) -> None:
        world = build(knee=0.0, side="L")
        assert knee_flexion(world, Side.LEFT) == pytest.approx(0.0, abs=EXACT)

    def test_every_measurement_is_zero_at_anatomical_neutral(self) -> None:
        """Standing at rest is 0 for all twelve - that is what neutral means."""
        marks = landmark_set(build(side="L", body_yaw=90.0))
        angles = compute_angles(marks, _anterior(marks.image_xy))
        for (name, side), value in angles.items():
            assert value == pytest.approx(0.0, abs=1e-6), f"{name.value} {side.value}"


class TestThreeDimensionalExactness:
    """3D is orientation-independent, so it must recover every angle exactly."""

    @pytest.mark.parametrize("name", list(SWEEPS))
    @pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
    def test_recovers_known_angles(self, name: MeasurementName, side: Side) -> None:
        arg, values = SWEEPS[name]
        for truth in values:
            marks = landmark_set(build(**{arg: truth}, side=side.name.title()[0]))
            got = compute_angles_3d(marks)[(name, side)]
            assert got == pytest.approx(truth, abs=EXACT), (
                f"{name.value} {side.value} at {truth} deg read {got}"
            )

    @pytest.mark.parametrize("yaw", [0, 30, 60, 90, 135, 180])
    def test_body_rotation_changes_nothing(self, yaw: float) -> None:
        """The frame is body-relative, so turning must not move the number."""
        for name, (arg, values) in SWEEPS.items():
            truth = values[-1]
            marks = landmark_set(build(**{arg: truth}, side="L", body_yaw=yaw))
            got = compute_angles_3d(marks)[(name, Side.LEFT)]
            assert got == pytest.approx(truth, abs=1e-5), f"{name.value} at yaw {yaw}"


class TestTwoDimensionalExactness:
    """2D is exact only when the plane of motion faces the camera.

    Sagittal measurements therefore need the subject side-on; abduction needs
    them face-on. The two are mutually exclusive (FINDINGS.md F34).
    """

    SAGITTAL = [
        MeasurementName.ELBOW_FLEXION,
        MeasurementName.KNEE_FLEXION,
        MeasurementName.SHOULDER_FLEXION,
        MeasurementName.HIP_FLEXION,
        MeasurementName.ANKLE_ANGLE,
    ]

    @pytest.mark.parametrize("name", SAGITTAL)
    def test_sagittal_measurements_are_exact_side_on(self, name: MeasurementName) -> None:
        arg, values = SWEEPS[name]
        for truth in values:
            world = build(**{arg: truth}, side="L", body_yaw=90.0)
            marks = landmark_set(world)
            got = compute_angles(marks, _anterior(marks.image_xy))[(name, Side.LEFT)]
            assert got == pytest.approx(truth, abs=1e-4), (
                f"{name.value} at {truth} deg read {got} side-on"
            )

    def test_abduction_is_exact_face_on(self) -> None:
        arg, values = SWEEPS[MeasurementName.SHOULDER_ABDUCTION]
        for truth in values:
            world = build(**{arg: truth}, side="L", body_yaw=0.0)
            marks = landmark_set(world)
            got = compute_angles(marks, _anterior(marks.image_xy))[
                (MeasurementName.SHOULDER_ABDUCTION, Side.LEFT)
            ]
            assert got == pytest.approx(truth, abs=1e-4)

    def test_sagittal_measurement_is_badly_wrong_facing_the_camera(self) -> None:
        """Not a bug - the reason the orientation gate exists.

        Facing the camera, the sagittal plane is perpendicular to the image
        plane, so forward motion projects onto almost nothing (F20, F27).
        """
        world = build(elbow=60.0, side="L", body_yaw=0.0)
        marks = landmark_set(world)
        got = compute_angles(marks, _anterior(marks.image_xy))[
            (MeasurementName.ELBOW_FLEXION, Side.LEFT)
        ]
        assert abs(got - 60.0) > 20.0, "expected large projection error facing the camera"


class TestSignsAndDegenerateCases:
    def test_positive_is_flexion_negative_is_extension(self) -> None:
        for arg, name in (
            ("shoulder_flexion", MeasurementName.SHOULDER_FLEXION),
            ("hip_flexion", MeasurementName.HIP_FLEXION),
        ):
            forward = compute_angles_3d(landmark_set(build(**{arg: 45.0}, side="L")))
            backward = compute_angles_3d(landmark_set(build(**{arg: -25.0}, side="L")))
            assert forward[(name, Side.LEFT)] > 0
            assert backward[(name, Side.LEFT)] < 0

    def test_dorsiflexion_positive_plantarflexion_negative(self) -> None:
        up = compute_angles_3d(landmark_set(build(ankle=15.0, side="L")))
        down = compute_angles_3d(landmark_set(build(ankle=-40.0, side="L")))
        assert up[(MeasurementName.ANKLE_ANGLE, Side.LEFT)] > 0
        assert down[(MeasurementName.ANKLE_ANGLE, Side.LEFT)] < 0

    def test_unknown_anterior_yields_nan_for_signed_measurements(self) -> None:
        """A guessed sign would turn flexion into extension - a 2x error that
        looks entirely plausible on screen."""
        marks = landmark_set(build(shoulder_flexion=90.0, side="L", body_yaw=90.0))
        angles = compute_angles(marks, anterior_sign=None)
        assert math.isnan(angles[(MeasurementName.SHOULDER_FLEXION, Side.LEFT)])
        assert math.isnan(angles[(MeasurementName.HIP_FLEXION, Side.LEFT)])
        # Hinge joints need no anterior direction, so they still report.
        assert not math.isnan(angles[(MeasurementName.ELBOW_FLEXION, Side.LEFT)])

    def test_all_twelve_measurements_are_always_present(self) -> None:
        """NaN rather than omission, so the display never reasons about
        missing keys."""
        marks = landmark_set(build(side="L", body_yaw=90.0))
        assert len(compute_angles(marks, 1.0)) == 12
        assert len(compute_angles(marks, None)) == 12

    def test_collapsed_torso_gives_no_frame(self) -> None:
        points = np.zeros((33, 3))
        assert spatial_frame(points) is None
        assert image_frame(np.zeros((33, 2)), 1.0) is None


class TestAnteriorCues:
    def test_all_three_cues_agree_on_a_clean_side_on_pose(self) -> None:
        points = project(build(side="L", body_yaw=90.0))
        cues = anterior_cues(points)
        assert cues[0] == cues[1] == cues[2]

    def test_cues_flip_when_the_subject_turns_around(self) -> None:
        facing = anterior_cues(project(build(side="L", body_yaw=90.0)))
        away = anterior_cues(project(build(side="L", body_yaw=270.0)))
        assert facing[0] == -away[0]
