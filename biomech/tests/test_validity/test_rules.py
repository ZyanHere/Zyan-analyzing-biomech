"""The validity layer: what gets rejected, what gets a warning, and why.

The behaviour under test is the brief's hardest requirement - refusing to
answer - and the subtlest rule in the system: a veto signal that cannot be
computed blocks, while an advisory signal that cannot be computed is omitted.
"""

from __future__ import annotations

import numpy as np
import pytest

from biomech.config import ValidityConfig
from biomech.types import LandmarkSet, MeasurementName, Plane, Side
from biomech.validity.anterior import AnteriorTracker
from biomech.validity.orientation import OrientationState
from biomech.validity.rules import ValidityInputs, ValidityJudge
from biomech.validity.signals import BoneLengthTracker

from ..synthetic import build, landmark_set, project

SAGITTAL = OrientationState(yaw_deg=80.0, plane=Plane.SAGITTAL, settling=False)
FRONTAL = OrientationState(yaw_deg=5.0, plane=Plane.FRONTAL, settling=False)
BETWEEN = OrientationState(yaw_deg=38.0, plane=None, settling=False)


def _settled_bones(marks: LandmarkSet, frames: int = 40) -> BoneLengthTracker:
    """A tracker with enough identical history to be stable."""
    bones = BoneLengthTracker(window=30)
    for _ in range(frames):
        bones.update(marks)
    return bones


def _inputs(marks: LandmarkSet | None, orientation: OrientationState = SAGITTAL,
            anterior_known: bool = True, bones: BoneLengthTracker | None = None
            ) -> ValidityInputs:
    reference = marks if marks is not None else landmark_set(build(side="L"))
    return ValidityInputs(
        landmarks=marks,
        orientation=orientation,
        bones=bones if bones is not None else _settled_bones(reference),
        anterior_known=anterior_known,
    )


@pytest.fixture
def judge() -> ValidityJudge:
    return ValidityJudge(ValidityConfig())


@pytest.fixture
def standing() -> LandmarkSet:
    return landmark_set(build(side="L", body_yaw=90.0))


class TestRejection:
    def test_no_person_rejects_everything_with_that_reason(
        self, judge: ValidityJudge
    ) -> None:
        verdicts = judge.judge_all(_inputs(None))
        assert len(verdicts) == 12
        assert all(not v.is_valid for v in verdicts.values())
        assert all(v.reason == "no person detected" for v in verdicts.values())

    def test_wrong_plane_rejects_with_an_instruction(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        """Abduction needs the subject facing the camera; side-on it cannot be
        measured at all, and the reason should say what to do."""
        verdict = judge.judge(
            _inputs(standing, SAGITTAL), MeasurementName.SHOULDER_ABDUCTION, Side.LEFT
        )
        assert not verdict.is_valid
        assert "face" in verdict.reason.lower()

    def test_neither_plane_rejects_everything(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        verdicts = judge.judge_all(_inputs(standing, BETWEEN))
        assert all(not v.is_valid for v in verdicts.values())

    def test_unknown_facing_rejects_only_the_signed_measurements(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        """A guessed sign turns flexion into extension. Hinge joints are
        unsigned interior angles and need no facing direction, so they survive."""
        inputs = _inputs(standing, SAGITTAL, anterior_known=False)
        assert not judge.judge(inputs, MeasurementName.HIP_FLEXION, Side.LEFT).is_valid
        assert judge.judge(inputs, MeasurementName.ELBOW_FLEXION, Side.LEFT).is_valid
        assert judge.judge(inputs, MeasurementName.KNEE_FLEXION, Side.LEFT).is_valid

    def test_low_visibility_rejects_only_the_affected_limb(
        self, judge: ValidityJudge
    ) -> None:
        """The failure a global confidence number would miss: leg visibility
        stayed at 0.95 while an arm was hidden behind the back (F30)."""
        world = build(side="L", body_yaw=90.0)
        visibility = np.ones(33)
        for elbow_chain in (11, 13, 15):
            visibility[elbow_chain] = 0.2
        marks = LandmarkSet(project(world), world, visibility)

        inputs = _inputs(marks)
        assert not judge.judge(inputs, MeasurementName.ELBOW_FLEXION, Side.LEFT).is_valid
        assert judge.judge(inputs, MeasurementName.KNEE_FLEXION, Side.LEFT).is_valid

    def test_the_other_side_keeps_reporting_when_one_is_hidden(
        self, judge: ValidityJudge
    ) -> None:
        """The brief says compute each side whenever THAT side is reliable."""
        world = build(side="L", body_yaw=90.0)
        visibility = np.ones(33)
        for right_leg in (24, 26, 28):
            visibility[right_leg] = 0.1
        marks = LandmarkSet(project(world), world, visibility)

        inputs = _inputs(marks)
        assert not judge.judge(inputs, MeasurementName.KNEE_FLEXION, Side.RIGHT).is_valid
        assert judge.judge(inputs, MeasurementName.KNEE_FLEXION, Side.LEFT).is_valid


class TestVetoVersusAdvisory:
    def test_uncomputable_veto_signal_blocks(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        """Bone variance with no history yet is unknown, not fine."""
        empty = BoneLengthTracker(window=30)
        verdict = judge.judge(
            _inputs(standing, bones=empty), MeasurementName.ELBOW_FLEXION, Side.LEFT
        )
        assert not verdict.is_valid
        assert "gathering" in verdict.reason

    def test_uncomputable_advisory_signal_is_simply_omitted(
        self, judge: ValidityJudge
    ) -> None:
        """Asymmetry is bilateral. With one side hidden it cannot be computed -
        and that must not reject the visible side (F37)."""
        world = build(side="L", body_yaw=90.0)
        marks = landmark_set(world)
        bones = BoneLengthTracker(window=30)
        for _ in range(40):
            bones.update(marks)

        verdict = judge.judge(
            _inputs(marks, bones=bones), MeasurementName.ELBOW_FLEXION, Side.LEFT
        )
        assert verdict.is_valid, "an advisory signal must never veto"

    def test_unstable_tracking_rejects(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        """A limb whose length changes frame to frame is not being tracked;
        this is what caught the seated case, which visibility missed (F37)."""
        bones = BoneLengthTracker(window=30)
        for scale in np.linspace(0.6, 1.4, 40):
            bones.update(
                LandmarkSet(standing.image_xy * scale, standing.world_xyz,
                            standing.visibility)
            )
        verdict = judge.judge(
            _inputs(standing, bones=bones), MeasurementName.ELBOW_FLEXION, Side.LEFT
        )
        assert not verdict.is_valid
        assert "unstable" in verdict.reason

    def test_a_valid_measurement_can_still_carry_a_warning(
        self, judge: ValidityJudge
    ) -> None:
        """Asymmetry warns alongside a number rather than replacing it."""
        world = build(side="L", body_yaw=90.0)
        marks = landmark_set(world)
        bones = BoneLengthTracker(window=30)
        for _ in range(40):
            bones.update(marks)
        # Lengthen one femur so left and right disagree.
        skewed = marks.image_xy.copy()
        skewed[25] = skewed[23] + (skewed[25] - skewed[23]) * 1.4
        for _ in range(40):
            bones.update(LandmarkSet(skewed, marks.world_xyz, marks.visibility))

        verdict = judge.judge(
            _inputs(marks, bones=bones), MeasurementName.HIP_FLEXION, Side.LEFT
        )
        assert verdict.is_valid or "unstable" in verdict.reason


class TestEveryVerdictExplainsItself:
    def test_no_rejection_is_ever_silent(
        self, judge: ValidityJudge, standing: LandmarkSet
    ) -> None:
        """A blank with no reason is the thing this layer exists to prevent."""
        for inputs in (
            _inputs(None),
            _inputs(standing, BETWEEN),
            _inputs(standing, FRONTAL),
            _inputs(standing, SAGITTAL, anterior_known=False),
            _inputs(standing, bones=BoneLengthTracker(window=30)),
        ):
            for verdict in judge.judge_all(inputs).values():
                if not verdict.is_valid:
                    assert verdict.reason, "rejected without a reason"


class TestAnteriorTracker:
    def test_unanimous_cues_establish_the_direction(self) -> None:
        tracker = AnteriorTracker()
        points = project(build(side="L", body_yaw=90.0))
        assert tracker.update(points) is not None
        assert tracker.is_established

    def test_direction_is_held_through_an_ambiguous_frame(self) -> None:
        """Ambiguous frames are discarded, not averaged - because the cues fail
        in correlated ways and voting scored worse than the best single cue."""
        tracker = AnteriorTracker()
        good = project(build(side="L", body_yaw=90.0))
        established = tracker.update(good)

        confusing = good.copy()
        confusing[0] = confusing[0] + np.array([-400.0, 0.0])  # nose cue flipped
        assert tracker.update(confusing) == established

    def test_nothing_is_reported_until_the_cues_have_agreed_once(self) -> None:
        tracker = AnteriorTracker()
        assert not tracker.is_established
        assert "unknown" in tracker.confidence_note
