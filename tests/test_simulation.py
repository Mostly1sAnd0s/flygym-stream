"""Tests for flygym_stream.simulation (PLAN.md Phase 4).

Pacing/pause/step/input-order tests drive ``SimThread.tick`` directly with
a FakeClock (exact, race-free); one test exercises the real thread start/
stop. MuJoCoBackend tests use the real tiny model (CPU, no GL).
"""

from __future__ import annotations

import importlib.util
import time

import mujoco as mj
import numpy as np
import pytest

from flygym_stream.config import Config
from flygym_stream.protocol import (
    Actuator, ActuatorSet, Neutral, Orbit, Pause, Reset, Resume, Speed, Step,
)
from flygym_stream.scene import Scene
from flygym_stream.simulation import (
    MAX_CATCHUP_STEPS,
    MJWarpBackend,
    MuJoCoBackend,
    SimThread,
    make_backend,
)
from flygym_stream.stream import FrameQueue, InputQueue


def make_sim(backend, renderer, fake_clock, inputs=None) -> SimThread:
    """A SimThread on deterministic time (no real thread started)."""
    return SimThread(
        backend, renderer, FrameQueue(), inputs or InputQueue(),
        Config(width=320, height=240, playback_speed=1.0),
        clock=fake_clock, sleep=fake_clock.sleep,
    )


def test_tick_steps_track_wall_clock(fake_backend, fake_renderer, fake_clock) -> None:
    sim = make_sim(fake_backend, fake_renderer, fake_clock)
    sim.tick()
    assert fake_backend.steps == 0
    fake_clock.t = 0.010
    sim.tick()
    assert fake_backend.steps == 10
    assert fake_backend.time == pytest.approx(0.01)
    fake_clock.t = 0.015
    sim.tick()
    assert fake_backend.steps == 15


def test_tick_renders_only_when_due(fake_backend, fake_renderer, fake_clock) -> None:
    frames = FrameQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, InputQueue(),
                    Config(width=320, height=240, output_fps=30),
                    clock=fake_clock, sleep=fake_clock.sleep)
    fake_clock.t = 0.040  # 40 steps; first render due at sim time >= 1/30
    sim.tick()
    assert frames.get(timeout=0.01) == b"JPEG-1"
    fake_clock.t = 0.050  # not due again until 0.04 + 1/30
    sim.tick()
    assert frames.get(timeout=0.01) is None
    fake_clock.t = 0.080  # due again
    sim.tick()
    assert frames.get(timeout=0.01) == b"JPEG-2"


def test_tick_caps_catchup(fake_backend, fake_renderer, fake_clock) -> None:
    sim = make_sim(fake_backend, fake_renderer, fake_clock)
    fake_clock.t = 100.0  # huge backlog
    sim.tick()
    assert fake_backend.steps == MAX_CATCHUP_STEPS


def test_pause_blocks_stepping(fake_backend, fake_renderer, fake_clock) -> None:
    sim = make_sim(fake_backend, fake_renderer, fake_clock)
    fake_clock.t = 0.010
    sim.tick()
    assert fake_backend.steps == 10
    sim.pause()
    assert sim.paused
    fake_clock.t = 0.5
    sim.tick()
    assert fake_backend.steps == 10  # frozen


def test_step_n_advances_exactly_when_paused(fake_backend, fake_renderer, fake_clock) -> None:
    frames = FrameQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, InputQueue(),
                    Config(width=320, height=240),
                    clock=fake_clock, sleep=fake_clock.sleep)
    with pytest.raises(RuntimeError, match="paused"):
        sim.step(3)
    sim.pause()
    sim.step(7)
    assert fake_backend.steps == 7
    assert fake_backend.time == pytest.approx(7 * fake_backend.dt)
    assert frames.get(timeout=0.01) == b"JPEG-1"  # forced frame
    sim.step(3)
    assert fake_backend.steps == 10


def test_inputs_applied_in_order(fake_backend, fake_renderer, fake_clock) -> None:
    inputs = InputQueue()
    inputs.put(Actuator(index=0, value=0.5))
    inputs.put(ActuatorSet(values=(0.1,)))
    sim = make_sim(fake_backend, fake_renderer, fake_clock, inputs)
    sim.tick()
    assert fake_backend.inputs == [Actuator(index=0, value=0.5),
                                   ActuatorSet(values=(0.1,))]


def test_dispatch_loop_control_messages(fake_backend, fake_renderer, fake_clock) -> None:
    inputs = InputQueue()
    sim = make_sim(fake_backend, fake_renderer, fake_clock, inputs)
    inputs.put(Pause())
    fake_clock.t = 0.010
    sim.tick()
    assert sim.paused and fake_backend.steps == 0
    inputs.put(Speed(value=2.0))
    inputs.put(Resume())
    sim.tick()
    assert not sim.paused
    fake_clock.t = 0.025  # 0.015 s after rebase x 2 speed = 30 steps
    sim.tick()
    assert fake_backend.steps == 30
    inputs.put(Reset())
    sim.tick()
    assert fake_backend.time == 0.0


def test_set_speed_rebases_pacing(fake_backend, fake_renderer, fake_clock) -> None:
    sim = make_sim(fake_backend, fake_renderer, fake_clock)
    fake_clock.t = 0.010
    sim.tick()
    assert fake_backend.steps == 10
    fake_clock.t = 0.011
    sim.set_speed(2.0)
    fake_clock.t = 0.021  # 0.010 s wall x 2 = 0.020 sim
    sim.tick()
    assert fake_backend.steps == 30
    for bad in (0.0, -1.0, 101.0):
        with pytest.raises(ValueError):
            sim.set_speed(bad)


def test_thread_start_stop_clean(fake_backend, fake_renderer) -> None:
    frames = FrameQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, InputQueue(),
                    Config(width=320, height=240, output_fps=30))
    sim.start()
    time.sleep(0.2)
    sim.stop()
    assert not sim.running
    sim.stop()  # double stop is safe
    assert len(frames) >= 1  # frames flowed while running


def make_tiny_scene(tiny_mj_model) -> Scene:
    m, d = tiny_mj_model
    return Scene(m, d, actuator_ids=[0], target_site="target")


def test_mujoco_backend_step_and_time(tiny_mj_model) -> None:
    m, _ = tiny_mj_model
    b = MuJoCoBackend(make_tiny_scene(tiny_mj_model))
    assert b.name == "cpu" and b.n_actuators == 1
    assert b.dt == m.opt.timestep
    b.step(b.dt)
    assert b.time == pytest.approx(m.opt.timestep)
    with pytest.raises(ValueError):
        b.step(9.0)


def test_mujoco_backend_applies_inputs(tiny_mj_model) -> None:
    scene = make_tiny_scene(tiny_mj_model)
    b = MuJoCoBackend(scene)
    b.apply_input(Actuator(index=0, value=0.7))
    assert scene.mj_data.ctrl[0] == pytest.approx(0.7)
    b.apply_input(ActuatorSet(values=(0.3,)))
    assert scene.mj_data.ctrl[0] == pytest.approx(0.3)
    b.apply_input(Neutral())
    assert scene.mj_data.ctrl[0] == pytest.approx(0.0)  # no keyframe -> zeros
    b.apply_input(Orbit(azimuth=10.0, elevation=-5.0, distance=50.0))
    cam = mj.MjvCamera()
    scene.update_mjv_camera(cam)
    assert cam.azimuth == pytest.approx(10.0)
    with pytest.raises(ValueError):
        b.apply_input(Actuator(index=9, value=0.0))
    with pytest.raises(TypeError):
        b.apply_input(Pause())  # loop-control msgs are not sim inputs


def test_mujoco_backend_reset(tiny_mj_model) -> None:
    scene = make_tiny_scene(tiny_mj_model)
    b = MuJoCoBackend(scene)
    b.step(b.dt)
    b.apply_input(Actuator(index=0, value=1.0))
    b.reset()
    assert b.time == pytest.approx(0.0)
    assert scene.mj_data.ctrl[0] == pytest.approx(0.0)


def test_mjwarp_backend_missing_extra_raises(tiny_mj_model) -> None:
    if importlib.util.find_spec("mujoco_warp") is not None:
        pytest.skip("gpu extra is installed; error path not reachable")
    with pytest.raises(RuntimeError, match="gpu"):
        MJWarpBackend(make_tiny_scene(tiny_mj_model))


def test_make_backend_selection(tiny_mj_model) -> None:
    scene = make_tiny_scene(tiny_mj_model)
    assert isinstance(make_backend(scene, Config(backend="cpu")), MuJoCoBackend)
    if importlib.util.find_spec("mujoco_warp") is None:
        assert isinstance(make_backend(scene, Config(backend="auto")), MuJoCoBackend)
        with pytest.raises(RuntimeError, match="gpu"):
            make_backend(scene, Config(backend="gpu"))


# --- Phase 7 (M7.2): state writes via Backend.apply_state ---------------------

def _state_backend(tiny_mj_model):
    from flygym_stream.scene import Scene
    from flygym_stream.simulation import MuJoCoBackend
    m, d = tiny_mj_model
    return MuJoCoBackend(Scene(m, d, actuator_ids=[0], target_site="target"))


def test_set_state_writes_qpos_and_ctrl(tiny_mj_model) -> None:
    from flygym_stream.protocol import SetState
    b = _state_backend(tiny_mj_model)
    b.apply_state(SetState(qpos=(0.5,), qvel=(9.0,), ctrl=(1.5,)))
    assert b.time >= 0.0  # sanity
    assert b._scene.mj_data.qpos[0] == 0.5
    assert b._scene.mj_data.qvel[0] == 9.0
    assert b._scene.mj_data.ctrl[0] == 1.5


def test_set_state_rejects_wrong_lengths(tiny_mj_model) -> None:
    from flygym_stream.protocol import SetState
    b = _state_backend(tiny_mj_model)
    with pytest.raises(ValueError, match="length"):
        b.apply_state(SetState(qpos=(0.5, 0.0), qvel=None, ctrl=None))


def test_spawn_requires_free_root(tiny_mj_model) -> None:
    """The tiny model's root body is fixed: spawn must raise, not no-op
    silently (the client can see the world type in model-info, M7.4)."""
    from flygym_stream.protocol import Spawn
    b = _state_backend(tiny_mj_model)
    with pytest.raises(ValueError, match="free root"):
        b.apply_state(Spawn(pos=(1, 2, 3), quat=(1, 0, 0, 0)))


def test_param_whitelist_applies(tiny_mj_model) -> None:
    from flygym_stream.protocol import Param
    b = _state_backend(tiny_mj_model)
    m = b._scene.mj_model
    b.apply_state(Param(name="gravity", value=(0.0, 0.0, -1.0)))
    assert list(m.opt.gravity) == [0.0, 0.0, -1.0]
    b.apply_state(Param(name="timestep", value=0.002))
    assert m.opt.timestep == pytest.approx(0.002)
    b.apply_state(Param(name="friction", value=(0.3, 0.4, 0.5)))
    assert list(m.opt.o_friction[0:3]) == [0.3, 0.4, 0.5]
    with pytest.raises(ValueError, match="whitelist"):
        b.apply_state(Param(name="bogus", value=1.0))


def test_force_injects_into_xfrc_applied(tiny_mj_model) -> None:
    import mujoco as mj
    from flygym_stream.protocol import Force
    b = _state_backend(tiny_mj_model)
    scene = b._scene
    body = mj.mj_name2id(scene.mj_model, mj.mjtObj.mjOBJ_BODY, "body")
    b.apply_state(Force(body="body", force=(1.0, 2.0, 3.0)))
    adr = 6 * body
    flat = np.asarray(scene.mj_data.xfrc_applied).reshape(-1)
    assert list(flat[adr:adr + 3]) == [1.0, 2.0, 3.0]
    # body by id works too; unknown body raises
    b.apply_state(Force(body=body, force=(0.5, 0.0, 0.0)))
    assert flat[adr] == 0.5
    with pytest.raises(ValueError, match="body"):
        b.apply_state(Force(body="nope", force=(0, 0, 0)))


def test_keyframe_unknown_rejected(tiny_mj_model) -> None:
    """The tiny model has no keyframes: unknown names must raise."""
    from flygym_stream.protocol import Keyframe
    b = _state_backend(tiny_mj_model)
    with pytest.raises(ValueError, match="keyframe"):
        b.apply_state(Keyframe(name="neutral"))


def test_adhesion_requires_adhesion_actors(tiny_mj_model) -> None:
    """No adhesion actuators in the tiny model: must raise, not no-op."""
    from flygym_stream.protocol import Adhesion
    b = _state_backend(tiny_mj_model)
    with pytest.raises(ValueError, match="adhesion"):
        b.apply_state(Adhesion(leg="lf", state=1))
