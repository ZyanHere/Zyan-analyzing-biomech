"""Recording and replay: the determinism the investigation relied on.

Two filters cannot be compared on a live human, because the human moves
differently each time. Every comparison in this project ran on a recording, so
replay producing identical output is not a nice property - it is the property
that made the findings possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biomech.capture.landmarks import LandmarkSource
from biomech.config import Config
from biomech.errors import CaptureError, SourceExhaustedError
from biomech.recording.writer import SessionRecorder
from biomech.types import LANDMARK_COUNT, Frame, PoseResult

from ..synthetic import build, landmark_set

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "research" / "fixtures" / "sessions" / "session.jsonl"
)
needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason="recorded session is gitignored"
)


def _frame(seq: int, size: tuple[int, int] = (640, 480)) -> Frame:
    return Frame(image=np.zeros((size[1], size[0], 3), np.uint8),
                 capture_ts=seq / 30.0, seq=seq)


def _result(seq: int) -> PoseResult:
    marks = landmark_set(build(elbow=float(seq), side="L", body_yaw=90.0))
    return PoseResult(landmarks=marks, capture_ts=seq / 30.0, seq=seq, inference_ms=22.0)


class TestRoundTrip:
    def test_what_is_written_is_what_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "session.jsonl"
        recorder = SessionRecorder(Config(), path, None)
        written = [(_frame(i), _result(i)) for i in range(1, 6)]
        for frame, result in written:
            recorder.write(frame, result)
        recorder.close()

        source = LandmarkSource(path)
        for original_frame, original_result in written:
            frame, result = source.next_pose()
            assert frame.seq == original_frame.seq
            assert result.landmarks is not None
            assert original_result.landmarks is not None
            np.testing.assert_allclose(
                result.landmarks.image_xy, original_result.landmarks.image_xy, atol=0.01
            )

    def test_the_recording_carries_its_own_configuration(self, tmp_path: Path) -> None:
        """A result is only meaningful with the settings that produced it, and
        every threshold here came from an experiment."""
        path = tmp_path / "session.jsonl"
        recorder = SessionRecorder(Config(), path, None)
        recorder.write(_frame(1), _result(1))
        recorder.close()

        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["config"]["validity"]["min_visibility"] == 0.55
        assert LandmarkSource(path).recorded_config is not None

    def test_frames_without_a_person_survive_the_round_trip(self, tmp_path: Path) -> None:
        """Absence must stay distinguishable from a person at the origin."""
        path = tmp_path / "session.jsonl"
        recorder = SessionRecorder(Config(), path, None)
        recorder.write(_frame(1), PoseResult(None, 0.03, 1, 20.0))
        recorder.close()

        _, result = LandmarkSource(path).next_pose()
        assert not result.has_person
        assert result.landmarks is None

    def test_end_of_recording_raises_rather_than_returning_none(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "session.jsonl"
        recorder = SessionRecorder(Config(), path, None)
        recorder.write(_frame(1), _result(1))
        recorder.close()

        source = LandmarkSource(path)
        source.next_pose()
        with pytest.raises(SourceExhaustedError, match="End of"):
            source.next_pose()


class TestLegacyFormat:
    """The investigation's fixtures predate the header, so they must still load.

    Otherwise every finding they support stops being reproducible by anyone who
    clones the repository.
    """

    @needs_fixture
    def test_the_investigation_fixture_replays(self) -> None:
        source = LandmarkSource(FIXTURE)
        frame, result = source.next_pose()
        assert result.landmarks is not None
        assert result.landmarks.image_xy.shape == (LANDMARK_COUNT, 2)

    @needs_fixture
    def test_normalised_coordinates_are_scaled_to_pixels(self) -> None:
        """The old format stored fractions of the frame; angles are computed in
        pixels, so they must be converted on the way in."""
        _, result = LandmarkSource(FIXTURE).next_pose()
        assert result.landmarks is not None
        assert result.landmarks.image_xy.max() > 2.0

    @needs_fixture
    def test_world_landmarks_survive_the_key_collision(self) -> None:
        """The old format uses "w" for world landmarks; the new one used it for
        frame width. A one-letter key meaning two things is a bug waiting."""
        _, result = LandmarkSource(FIXTURE).next_pose()
        assert result.landmarks is not None
        assert result.landmarks.world_xyz.shape == (LANDMARK_COUNT, 3)
        assert not np.allclose(result.landmarks.world_xyz, 0.0)


class TestDeterminism:
    def test_replaying_twice_gives_identical_landmarks(self, tmp_path: Path) -> None:
        """The property every comparison in the investigation depended on."""
        path = tmp_path / "session.jsonl"
        recorder = SessionRecorder(Config(), path, None)
        for i in range(1, 11):
            recorder.write(_frame(i), _result(i))
        recorder.close()

        def read_all() -> list[np.ndarray]:
            source, out = LandmarkSource(path), []
            for _ in range(10):
                _, result = source.next_pose()
                assert result.landmarks is not None
                out.append(result.landmarks.image_xy)
            return out

        for first, second in zip(read_all(), read_all(), strict=True):
            np.testing.assert_array_equal(first, second)


class TestFailures:
    def test_missing_file_fails_clearly(self) -> None:
        with pytest.raises(CaptureError, match="Could not read"):
            LandmarkSource(Path("no_such_recording.jsonl"))

    def test_empty_file_is_rejected_rather_than_replayed_as_nothing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(CaptureError, match="no landmark records"):
            LandmarkSource(path)

    def test_recorder_close_is_safe_to_call_twice(self, tmp_path: Path) -> None:
        recorder = SessionRecorder(Config(), tmp_path / "s.jsonl", None)
        recorder.close()
        recorder.close()
