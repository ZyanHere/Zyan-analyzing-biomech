"""A single-frame handoff between the capture thread and the consumer.

Owns: the newest-frame-wins policy and the drop count.
Does NOT own: capture, timing, or what a frame means.

This is a slot, not a queue, and that is the whole point. A queue would let a
slow consumer fall behind while frame rate still looked healthy - the display
would show smooth video of where the subject was a second ago. Discarding the
older frame keeps latency bounded by one frame instead of by queue depth.

Drops are counted rather than hidden: a high drop rate means the pipeline is
not keeping up, which is information the health panel should show even when
the output still looks smooth.
"""

from __future__ import annotations

import threading

from ..types import Frame


class LatestFrameSlot:
    """Thread-safe single-item handoff, newest wins."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._drops = 0
        self._accepted = 0

    def put(self, frame: Frame) -> bool:
        """Store a frame, discarding any the consumer has not taken.

        Returns:
            True if this overwrote an unconsumed frame, i.e. a drop occurred.
        """
        with self._lock:
            dropped = self._frame is not None
            if dropped:
                self._drops += 1
            self._frame = frame
            self._accepted += 1
            return dropped

    def take(self) -> Frame | None:
        """Remove and return the stored frame, or None if there is none.

        Clearing on take is deliberate: returning the same frame twice would
        let the pipeline process a stale image and report it as fresh work.
        """
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    @property
    def drops(self) -> int:
        """Frames discarded because the consumer had not taken the previous one."""
        with self._lock:
            return self._drops

    @property
    def accepted(self) -> int:
        """Frames written into the slot, including those later dropped."""
        with self._lock:
            return self._accepted

    @property
    def drop_rate(self) -> float:
        """Fraction of produced frames the consumer never saw, 0.0 to 1.0."""
        with self._lock:
            return self._drops / self._accepted if self._accepted else 0.0
