"""Tests for flygym_stream.stream queues (PLAN.md §3 queue semantics).

FrameQueue: maxsize 1, drop-oldest, no backpressure, timeout -> None,
latest frame never lost. InputQueue: unbounded FIFO of InputMessage.
"""

from __future__ import annotations

from flygym_stream.protocol import Pause, Step
from flygym_stream.stream import FrameQueue, InputQueue


def test_frame_queue_put_get() -> None:
    q = FrameQueue()
    q.put(b"one")
    assert q.get(timeout=0.01) == b"one"
    assert q.get(timeout=0.01) is None  # empty


def test_frame_queue_timeout_returns_none() -> None:
    assert FrameQueue().get(timeout=0.0) is None
    assert FrameQueue().get(timeout=0.005) is None


def test_frame_queue_drop_oldest_under_overflow() -> None:
    q = FrameQueue()
    for frame in (b"old", b"newer", b"newest"):
        q.put(frame)
    assert q.get(timeout=0.01) == b"newest"  # oldest dropped, latest kept
    assert q.get(timeout=0.01) is None


def test_frame_queue_put_never_blocks_when_full() -> None:
    import time

    q = FrameQueue()
    q.put(b"stale")  # queue full; producer must not block on a slow client
    t0 = time.perf_counter()
    q.put(b"fresh")
    assert time.perf_counter() - t0 < 0.05
    assert q.get(timeout=0.01) == b"fresh"


def test_frame_queue_qsize() -> None:
    q = FrameQueue()
    assert len(q) == 0
    q.put(b"a")
    q.put(b"b")
    assert len(q) == 1


def test_input_queue_fifo_order() -> None:
    q = InputQueue()
    q.put(Pause())
    q.put(Step(n=3))
    assert q.drain() == [Pause(), Step(n=3)]
    assert q.drain() == []  # drained


def test_input_queue_drain_nonblocking_when_empty() -> None:
    assert InputQueue().drain() == []
