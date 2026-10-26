"""Idle (no-viewer) sim semantics (2026-10 CPU fix).

With ``idle=True`` (used by ``serve``), SimThread does not step while
zero viewers are connected: no physics substeps, no frames, no render
work — the service idles at ~0 CPU. A viewer join resumes from a
rebased pacing anchor (no catch-up burst); a viewer leave freezes the
sim again. An explicit user pause always survives a viewer cycle.
"""

from __future__ import annotations

import time

from flygym_stream.config import Config
from flygym_stream.simulation import IDLE_SLEEP_S, MAX_CATCHUP_STEPS, SimThread
from flygym_stream.stream import FrameQueue, InputQueue


def make_idle_sim(backend, renderer, fake_clock) -> SimThread:
    """A SimThread on deterministic time with the serve idle policy."""
    return SimThread(
        backend, renderer, FrameQueue(), InputQueue(),
        Config(width=320, height=240, output_fps=30),
        clock=fake_clock, sleep=fake_clock.sleep, idle=True,
    )


def test_idle_zero_viewers_frozen(fake_backend, fake_renderer, fake_clock) -> None:
    """No viewer: paused, and no stepping/frames no matter the wall time."""
    frames = FrameQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, InputQueue(),
                    Config(width=320, height=240, output_fps=30),
                    clock=fake_clock, sleep=fake_clock.sleep, idle=True)
    assert sim.paused
    fake_clock.t = 10.0  # plenty of wall clock while idle
    sim.tick()
    assert fake_backend.steps == 0
    assert frames.get_nowait() is None


def test_idle_viewer_join_resumes_rebased(fake_backend, fake_renderer,
                                          fake_clock) -> None:
    """After a long idle, the first tick steps only the wall-clock delta
    (pacing rebased on join), not a MAX_CATCHUP_STEPS fast-forward."""
    sim = make_idle_sim(fake_backend, fake_renderer, fake_clock)
    fake_clock.t = 3600.0  # an hour of idle
    sim.tick()
    assert fake_backend.steps == 0
    sim.viewer_joined()
    assert not sim.paused
    fake_clock.t = 3600.010
    sim.tick()
    assert fake_backend.steps == 10  # exactly the 10 ms of dt=0.001
    assert fake_backend.steps < MAX_CATCHUP_STEPS


def test_idle_viewer_leave_freezes(fake_backend, fake_renderer, fake_clock) -> None:
    """A leaving viewer returns the idle sim to the frozen state."""
    sim = make_idle_sim(fake_backend, fake_renderer, fake_clock)
    sim.viewer_joined()
    fake_clock.t = 0.010
    sim.tick()
    assert fake_backend.steps == 10
    sim.viewer_left()
    assert sim.paused
    fake_clock.t = 1.0
    sim.tick()
    assert fake_backend.steps == 10  # frozen again


def test_idle_user_pause_survives_viewer_cycle(fake_backend, fake_renderer,
                                               fake_clock) -> None:
    """An explicit user pause is not lifted by a later viewer join."""
    sim = make_idle_sim(fake_backend, fake_renderer, fake_clock)
    sim.viewer_joined()
    sim.pause()
    sim.viewer_left()
    sim.viewer_joined()
    assert sim.paused
    fake_clock.t = 0.1
    sim.tick()
    assert fake_backend.steps == 0
    sim.resume()
    assert not sim.paused


def test_idle_step_n_works_with_no_viewer(fake_backend, fake_renderer,
                                          fake_clock) -> None:
    """step(n) is honored while idle-paused (forced frame included)."""
    frames = FrameQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, InputQueue(),
                    Config(width=320, height=240, output_fps=30),
                    clock=fake_clock, sleep=fake_clock.sleep, idle=True)
    sim.step(3)  # paused (idle) -> allowed
    assert fake_backend.steps == 3
    assert frames.get_nowait() == b"JPEG-1"


def test_idle_loop_sleeps_slowly_while_frozen(fake_backend, fake_renderer) -> None:
    """The real loop thread wakes at IDLE_SLEEP_S (not dt) while frozen:
    no ~10k it/s sleep(dt) spin burning CPU with no viewer."""
    sleeps: list[float] = []

    def record(seconds: float) -> None:
        sleeps.append(seconds)
        time.sleep(seconds)

    sim = SimThread(fake_backend, fake_renderer, FrameQueue(), InputQueue(),
                    Config(width=320, height=240, output_fps=30),
                    sleep=record, idle=True)
    sim.start()
    time.sleep(0.25)
    sim.stop()
    assert sleeps and all(s == IDLE_SLEEP_S for s in sleeps)
    assert fake_backend.steps == 0


def test_idle_false_keeps_legacy_behavior(fake_backend, fake_renderer,
                                          fake_clock) -> None:
    """idle=False (direct/test use): stepping ignores the viewer count."""
    sim = SimThread(fake_backend, fake_renderer, FrameQueue(), InputQueue(),
                    Config(width=320, height=240),
                    clock=fake_clock, sleep=fake_clock.sleep)
    assert not sim.paused  # no viewer, but idle is off
    fake_clock.t = 0.010
    sim.tick()
    assert fake_backend.steps == 10
