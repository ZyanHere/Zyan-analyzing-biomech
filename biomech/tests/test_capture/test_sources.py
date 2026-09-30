"""Source contract and video replay.

No live camera here: tests must run on a machine with no webcam, which is also
what lets a reviewer run them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomech.capture.camera import CameraSource
from biomech.capture.source import FrameSource
from biomech.capture.video import VideoSource
from biomech.config import CaptureConfig
from biomech.errors import CaptureError, SourceExhaustedError

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "research" / "fixtures" / "sessions" / "sess_still.mp4"
)
needs_fixture = pytest.mark.skipif(
    not FIXTURE.exists(), reason="recorded video is gitignored; run a session first"
)


def _contract() -> set[str]:
    """The public members FrameSource requires.

    Derived from the Protocol rather than listed here, so this test cannot
    drift from the interface it is checking.
    """
    return {name for name in vars(FrameSource) if not name.startswith("_")}


@pytest.mark.parametrize("cls", [CameraSource, VideoSource], ids=["camera", "video"])
def test_source_satisfies_the_contract(cls: type) -> None:
    """Structural check only - no device is opened, no file read.

    `issubclass` cannot be used here: a runtime_checkable Protocol with
    non-method members (description, is_healthy are properties) rejects it.
    Checking the members directly tests the same thing and names what is
    missing when it fails.
    """
    missing = _contract() - set(dir(cls))
    assert not missing, f"{cls.__name__} does not implement: {sorted(missing)}"


def test_unknown_backend_names_the_valid_ones() -> None:
    cfg = CaptureConfig(backend="magic")
    with pytest.raises(CaptureError, match="dshow"):
        CameraSource(cfg)


def test_missing_video_file_fails_clearly() -> None:
    with pytest.raises(CaptureError, match="Could not open video"):
        VideoSource(Path("no_such_file.mp4"))


@needs_fixture
def test_video_replays_frames_in_order() -> None:
    src = VideoSource(FIXTURE)
    try:
        seqs = [src.next_frame().seq for _ in range(5)]
    finally:
        src.close()
    assert seqs == [1, 2, 3, 4, 5]


@needs_fixture
def test_video_end_raises_rather_than_returning_none() -> None:
    """None means 'no frame yet, try again'; a finished file means 'stop'.

    Collapsing the two would make a finished video look like a dead camera.
    """
    src = VideoSource(FIXTURE)
    try:
        with pytest.raises(SourceExhaustedError, match="End of"):
            for _ in range(100_000):
                src.next_frame()
    finally:
        src.close()


@needs_fixture
def test_video_reports_no_drops() -> None:
    """A file delivers every frame; nothing is discarded for freshness."""
    src = VideoSource(FIXTURE)
    try:
        src.next_frame()
        assert src.drops == 0
        assert src.is_healthy
    finally:
        src.close()


@needs_fixture
def test_close_is_safe_to_call_twice() -> None:
    src = VideoSource(FIXTURE)
    src.close()
    src.close()
