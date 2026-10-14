"""Queue transport between the sim thread and the WebSocket handler.

``FrameQueue`` holds at most one encoded JPEG: a slow client never blocks
the producer (no backpressure, no latency buildup — PLAN.md §3).
``InputQueue`` is an unbounded FIFO of validated ``InputMessage`` objects.
"""

from __future__ import annotations

import queue

from flygym_stream.protocol import InputMessage


class FrameQueue:
    """Latest-frame-wins queue (maxsize 1, drop-oldest on overflow).

    ``put`` never blocks: when the single slot is occupied (consumer slow),
    the stale frame is dropped and the new one takes its place.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=1)

    def put(self, jpeg: bytes) -> None:
        """Store the newest frame, dropping whatever was queued (if any)."""
        while True:
            try:
                self._q.put_nowait(jpeg)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()  # drop oldest
                except queue.Empty:
                    continue  # consumer took it; retry the put

    def get(self, timeout: float | None = None) -> bytes | None:
        """Wait up to ``timeout`` seconds for a frame; None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def __len__(self) -> int:
        return self._q.qsize()


class InputQueue:
    """Unbounded FIFO of validated control messages (PLAN.md §3)."""

    def __init__(self) -> None:
        self._q: queue.Queue[InputMessage] = queue.Queue()

    def put(self, msg: InputMessage) -> None:
        """Queue a message for the sim thread (never blocks)."""
        self._q.put(msg)

    def drain(self) -> list[InputMessage]:
        """Return all queued messages in FIFO order (non-blocking)."""
        drained: list[InputMessage] = []
        while True:
            try:
                drained.append(self._q.get_nowait())
            except queue.Empty:
                return drained
