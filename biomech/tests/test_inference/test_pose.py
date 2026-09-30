"""Inference: timestamps, model loading, and landmark extraction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from biomech.config import ModelConfig
from biomech.errors import ModelError
from biomech.inference.pose import MonotonicMillis, PoseEstimator
from biomech.paths import model_path
from biomech.types import LANDMARK_COUNT

VIDEO = (
    Path(__file__).resolve().parents[2]
    / "research" / "fixtures" / "sessions" / "sess_squat.mp4"
)
needs_model = pytest.mark.skipif(
    not model_path("full").exists(), reason="model is gitignored; see README to download"
)
needs_video = pytest.mark.skipif(
    not VIDEO.exists(), reason="recorded video is gitignored; run a session first"
)


class TestMonotonicMillis:
    """VIDEO mode rejects a repeated timestamp, and the failure is silent."""

    def test_starts_at_zero_on_the_first_frame(self) -> None:
        ts = MonotonicMillis()
        assert ts(1234.5) == 0

    def test_measures_elapsed_time_from_the_first_frame(self) -> None:
        ts = MonotonicMillis()
        ts(100.0)
        assert ts(100.25) == 250

    def test_two_frames_in_one_millisecond_do_not_repeat(self) -> None:
        """At 30 FPS this is rare; under replay it is routine."""
        ts = MonotonicMillis()
        first = ts(50.0)
        second = ts(50.0001)
        assert second > first, "a repeated timestamp would be rejected by MediaPipe"

    def test_an_out_of_order_frame_does_not_rewind_the_clock(self) -> None:
        ts = MonotonicMillis()
        ts(10.0)
        forward = ts(10.5)
        backward = ts(10.1)
        assert backward > forward

    def test_stays_strictly_increasing_over_a_burst(self) -> None:
        from itertools import pairwise

        ts = MonotonicMillis()
        values = [ts(100.0 + i * 0.0001) for i in range(200)]
        assert all(b > a for a, b in pairwise(values))


class TestModelLoading:
    def test_missing_model_gives_the_download_command(self) -> None:
        """A library stack trace would not tell the user what to do."""
        cfg = ModelConfig(variant="nonexistent")
        with pytest.raises(ModelError) as exc:
            PoseEstimator(cfg)
        message = str(exc.value)
        assert "curl" in message
        assert "storage.googleapis.com" in message

    @needs_model
    def test_unknown_running_mode_names_the_valid_ones(self) -> None:
        cfg = replace(ModelConfig(), running_mode="interpretive_dance")
        with pytest.raises(ModelError, match="video, image"):
            PoseEstimator(cfg)


@needs_model
@needs_video
class TestEstimation:
    """End to end over recorded frames. No camera required."""

    def test_returns_thirty_three_landmarks(self) -> None:
        from biomech.capture.video import VideoSource

        source, estimator = VideoSource(VIDEO), PoseEstimator(ModelConfig())
        try:
            result = estimator.estimate(source.next_frame())
            assert result.has_person
            assert result.landmarks is not None
            assert result.landmarks.image_xy.shape == (LANDMARK_COUNT, 2)
            assert result.landmarks.world_xyz.shape == (LANDMARK_COUNT, 3)
            assert result.landmarks.visibility.shape == (LANDMARK_COUNT,)
        finally:
            estimator.close()
            source.close()

    def test_image_coordinates_are_pixels_not_fractions(self) -> None:
        """Angles are computed in pixels; normalised coordinates would make
        every measurement depend on the aspect ratio."""
        from biomech.capture.video import VideoSource

        source, estimator = VideoSource(VIDEO), PoseEstimator(ModelConfig())
        try:
            marks = estimator.estimate(source.next_frame()).landmarks
            assert marks is not None
            assert marks.image_xy.max() > 2.0, "looks normalised, expected pixels"
        finally:
            estimator.close()
            source.close()

    def test_inference_time_is_recorded_and_plausible(self) -> None:
        from biomech.capture.video import VideoSource

        source, estimator = VideoSource(VIDEO), PoseEstimator(ModelConfig())
        try:
            for _ in range(3):  # discard warm-up
                estimator.estimate(source.next_frame())
            result = estimator.estimate(source.next_frame())
            assert 1.0 < result.inference_ms < 500.0
        finally:
            estimator.close()
            source.close()

    def test_frame_identity_is_carried_through(self) -> None:
        """capture_ts must survive to the far end, or end-to-end latency is
        unmeasurable."""
        from biomech.capture.video import VideoSource

        source, estimator = VideoSource(VIDEO), PoseEstimator(ModelConfig())
        try:
            frame = source.next_frame()
            result = estimator.estimate(frame)
            assert result.seq == frame.seq
            assert result.capture_ts == frame.capture_ts
        finally:
            estimator.close()
            source.close()

    def test_close_is_safe_to_call_twice(self) -> None:
        estimator = PoseEstimator(ModelConfig())
        estimator.close()
        estimator.close()
