"""The display contract.

Two guarantees are tested rather than the pixels: every measurement is always
present, and no blank is ever unexplained. Those are requirements from the
brief, not styling.
"""

from __future__ import annotations

import numpy as np
import pytest

from biomech.biomechanics.conventions import present
from biomech.types import MeasurementName, Plane, Side, Verdict
from biomech.ui.layout import FrameComposer, canvas_size
from biomech.ui.panel import (
    HEALTH_HEIGHT,
    PANEL_WIDTH,
    ROWS,
    health_panel,
    measurement_panel,
)
from biomech.validity.orientation import OrientationState

from ..synthetic import build, landmark_set

SAGITTAL = OrientationState(yaw_deg=80.0, plane=Plane.SAGITTAL, settling=False)

STATS = {
    "displayed_fps": 30.5, "inference_p50_ms": 22.0, "inference_p95_ms": 25.0,
    "biomech_p50_ms": 0.8, "render_p50_ms": 2.2, "render_p95_ms": 3.0,
    "end_to_end_p50_ms": 26.0, "end_to_end_p95_ms": 29.0,
    "frames_processed": 100.0, "frames_without_person": 0.0, "source_drops": 4.0,
}


class TestAlwaysTwelveRows:
    def test_every_measurement_has_a_row(self) -> None:
        assert len(ROWS) == 12

    def test_row_order_is_fixed(self) -> None:
        """A value must stay in the same place as the subject moves, so the eye
        can find it without reading."""
        assert tuple(ROWS) == ROWS
        assert ROWS[0] == (MeasurementName.ELBOW_FLEXION, Side.LEFT)
        assert ROWS[-1] == (MeasurementName.ANKLE_ANGLE, Side.RIGHT)

    def test_every_measurement_name_appears_exactly_twice(self) -> None:
        for name in MeasurementName:
            assert sum(1 for n, _ in ROWS if n is name) == 2


class TestNoUnexplainedBlank:
    def test_a_rejected_measurement_shows_its_reason(self) -> None:
        """The brief's requirement: indicate that state rather than display a
        misleading value. A bare blank indicates nothing."""
        verdicts = {key: Verdict.rejected("turn side-on") for key in ROWS}
        panel = np.zeros((600, PANEL_WIDTH, 3), np.uint8)
        measurement_panel(panel, {}, verdicts)
        assert panel.std() > 0, "panel rendered nothing at all"

    def test_every_rejection_reason_is_non_empty(self) -> None:
        """Enforced at the type level would be better; this at least catches a
        verdict constructed without one."""
        for reason in ("no person detected", "turn side-on", "arm not clearly visible"):
            assert Verdict.rejected(reason).reason

    def test_a_valid_measurement_with_an_advisory_still_shows_the_value(self) -> None:
        """An advisory accompanies a number; it never replaces one."""
        verdicts = {key: Verdict.ok(advisory="L/R mismatch 12%") for key in ROWS}
        angles = dict.fromkeys(ROWS, 45.0)
        panel = np.zeros((600, PANEL_WIDTH, 3), np.uint8)
        measurement_panel(panel, angles, verdicts)
        assert panel.std() > 0


class TestPresentation:
    def test_negative_values_are_named_not_signed(self) -> None:
        """The chart lists the two directions as separate named ranges, so a
        minus sign is not a quantity a clinician would recognise."""
        shown = present(MeasurementName.HIP_FLEXION, -30.0)
        assert shown.label == "extension"
        assert shown.magnitude_deg == 30.0
        assert "-" not in str(shown)

    def test_no_data_renders_as_a_dash(self) -> None:
        assert str(present(MeasurementName.ELBOW_FLEXION, float("nan"))) == "--"

    def test_out_of_range_is_flagged(self) -> None:
        assert not present(MeasurementName.ELBOW_FLEXION, 175.0).within_normal_range
        assert present(MeasurementName.ELBOW_FLEXION, 150.0).within_normal_range


class TestLayout:
    def test_canvas_is_built_at_native_video_size(self) -> None:
        """Not upscaled in numpy: the window manager scales for free, and this
        thread also runs inference."""
        width, height = canvas_size(640, 480)
        assert width == 640 + PANEL_WIDTH
        assert height > 480

    def test_compose_produces_a_frame_of_the_expected_shape(self) -> None:
        marks = landmark_set(build(side="L", body_yaw=90.0))
        image = np.zeros((480, 640, 3), np.uint8)
        canvas = FrameComposer(640, 480).compose(
            image=image, landmarks=marks, angles={}, verdicts={},
            orientation=SAGITTAL, stats=STATS, source_note="test",
            facing_note="facing fwd", min_visibility=0.55,
        )
        expected_w, expected_h = canvas_size(640, 480)
        assert canvas.shape == (expected_h, expected_w, 3)

    def test_compose_does_not_mutate_the_source_frame(self) -> None:
        """The frame belongs to the pipeline; drawing on it would corrupt the
        recording path that shares it."""
        marks = landmark_set(build(side="L", body_yaw=90.0))
        image = np.zeros((480, 640, 3), np.uint8)
        FrameComposer(640, 480).compose(
            image=image, landmarks=marks, angles={}, verdicts={},
            orientation=SAGITTAL, stats=STATS, source_note="t",
            facing_note="f", min_visibility=0.55,
        )
        assert image.sum() == 0, "source frame was drawn on"

    @pytest.mark.parametrize("fps", [12.0, 30.5])
    def test_health_panel_renders_at_any_frame_rate(self, fps: float) -> None:
        stats = {**STATS, "displayed_fps": fps}
        panel = np.zeros((HEALTH_HEIGHT, 640, 3), np.uint8)
        health_panel(panel, stats, "camera 0", SAGITTAL, "facing fwd")
        assert panel.std() > 0
