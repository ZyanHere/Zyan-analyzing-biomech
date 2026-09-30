"""The newest-frame-wins handoff.

The policy under test is the reason latency stays bounded: a queue would let a
slow consumer fall behind while frame rate still looked healthy.
"""

from __future__ import annotations

import threading

import numpy as np

from biomech.capture.slot import LatestFrameSlot
from biomech.types import Frame


def _frame(seq: int) -> Frame:
    return Frame(image=np.zeros((4, 4, 3), np.uint8), capture_ts=float(seq), seq=seq)


def test_take_returns_what_was_put() -> None:
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    taken = slot.take()
    assert taken is not None
    assert taken.seq == 1


def test_take_clears_the_slot() -> None:
    """Returning the same frame twice would let the pipeline process a stale
    image and report it as fresh work."""
    slot = LatestFrameSlot()
    slot.put(_frame(1))
    slot.take()
    assert slot.take() is None


def test_newest_frame_wins_and_the_older_one_is_counted_as_dropped() -> None:
    slot = LatestFrameSlot()
    assert slot.put(_frame(1)) is False
    assert slot.put(_frame(2)) is True

    taken = slot.take()
    assert taken is not None
    assert taken.seq == 2, "the consumer must see the newest frame, not the oldest"
    assert slot.drops == 1


def test_drop_rate_reflects_how_much_the_consumer_missed() -> None:
    slot = LatestFrameSlot()
    for i in range(10):
        slot.put(_frame(i))
    slot.take()
    assert slot.accepted == 10
    assert slot.drops == 9
    assert slot.drop_rate == 0.9


def test_drop_rate_is_zero_before_anything_is_produced() -> None:
    assert LatestFrameSlot().drop_rate == 0.0


def test_concurrent_writers_do_not_lose_the_drop_count() -> None:
    """The producer runs on its own thread; the counters must survive that."""
    slot = LatestFrameSlot()

    def produce() -> None:
        for i in range(200):
            slot.put(_frame(i))

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert slot.accepted == 800
    # Every put but the one still resident overwrote an unconsumed frame.
    assert slot.drops == 799
