"""The frame source interface.

Owns: the contract every source satisfies.
Does NOT own: any implementation. See camera.py, video.py, landmarks.py.

A Protocol rather than a base class: the three implementations share no code,
only a shape. Inheritance would add a layer that carries nothing.

The return contract distinguishes two states that need opposite responses:

    None                  no frame available *yet* - a live camera between
                          frames. The caller should try again.
    SourceExhaustedError  there will never be another frame - a file ended.
                          The caller should stop.

Collapsing these into one signal would make a finished video look like a dead
camera, and a slow camera look like a finished video.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Frame


@runtime_checkable
class FrameSource(Protocol):
    """Anything that can produce frames for the pipeline."""

    def next_frame(self) -> Frame | None:
        """The newest frame, or None if none is available yet.

        Raises:
            SourceExhaustedError: the source is finite and has ended.
            CaptureError: the source failed irrecoverably.
        """
        ...

    def close(self) -> None:
        """Release the device or file handle. Safe to call twice."""
        ...

    @property
    def description(self) -> str:
        """One line naming the source, for logs and the metrics report."""
        ...

    @property
    def is_healthy(self) -> bool:
        """False when the source is degraded but not yet dead.

        A camera that has stopped delivering frames is unhealthy while still
        open: the UI shows a banner rather than the application exiting, since
        a USB camera commonly recovers on its own.
        """
        ...
