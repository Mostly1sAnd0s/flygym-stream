"""Tests for the MJWarp GPU backend (PLAN.md Phase 6).

NoSlip-solver models are refused by mujoco-warp (no released version
implements it) — the default fly scene uses NoSlip, so the backend is a
fallback that works for NoSlip-free models; ``backend=auto`` falls back
to CPU. The parity test needs NVIDIA CUDA (warp), not EGL, but is marked
``gpu`` with the rest of the GPU-only tests.
"""

from __future__ import annotations

import builtins

import mujoco as mj
import numpy as np
import pytest

from flygym_stream.config import Config
from flygym_stream.protocol import Actuator, ActuatorSet, Neutral, Orbit
from flygym_stream.scene import Scene
from flygym_stream.simulation import MJWarpBackend, make_backend


def _scene(tiny_mj_model: tuple[mj.MjModel, mj.MjData]) -> Scene:
    m, d = tiny_mj_model
    return Scene(m, d, actuator_ids=[0], target_site="target")


def _hide_mjwarp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the 'gpu' extra is not installed."""
    real = builtins.__import__

    def boom(name: str, *args: object, **kwargs: object) -> object:
        if name == "mujoco_warp" or name.startswith("mujoco_warp."):
            raise ImportError("mocked missing extra")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)


def test_mjwarp_refuses_noslip_models(tiny_mj_model) -> None:
    """NoSlip models fail with an actionable error, before any CUDA work."""
    tiny_mj_model[0].opt.noslip_iterations = 3
    with pytest.raises(RuntimeError, match="NoSlip"):
        MJWarpBackend(_scene(tiny_mj_model))


def test_mjwarp_missing_extra_raises(
    tiny_mj_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hide_mjwarp(monkeypatch)
    with pytest.raises(RuntimeError, match="gpu.*extra|--backend cpu"):
        MJWarpBackend(_scene(tiny_mj_model))


@pytest.mark.gpu
def test_mjwarp_backend_steps_parity_with_cpu(tiny_mj_model) -> None:
    """GPU trajectory matches CPU Newton stepping for the same inputs."""
    gpu_scene = _scene(tiny_mj_model)
    backend = MJWarpBackend(gpu_scene)
    assert backend.name == "gpu"
    assert backend.n_actuators == 1

    # Independent CPU reference (the GPU backend mirrors into the scene's
    # MjData, so the fixture's own data cannot serve as the reference).
    from conftest import TINY_XML
    cpu_m = mj.MjModel.from_xml_string(TINY_XML)
    cpu_d = mj.MjData(cpu_m)
    mj.mj_forward(cpu_m, cpu_d)
    n_steps = 500
    for i in range(n_steps):
        if i == 100:
            # both ctrls set before the step that first consumes them
            backend.apply_input(Actuator(index=0, value=0.7))
            cpu_d.ctrl[0] = 0.7
        backend.step(backend.dt)
        mj.mj_step(cpu_m, cpu_d)

    # GPU time is float32: allow a few ulp of drift over 500 steps
    assert abs(backend.time - cpu_d.time) < 1e-4
    dq = np.abs(backend._d.qpos.numpy()[0] - cpu_d.qpos).max()
    dv = np.abs(backend._d.qvel.numpy()[0] - cpu_d.qvel).max()
    assert dq < 1e-3 and dv < 1e-3, (dq, dv)
    # the CPU mirror the renderer reads matches too
    assert np.abs(gpu_scene.mj_data.qpos - cpu_d.qpos).max() < 1e-6
    print(f"\nmjwarp parity: dq={dq:.2e} dv={dv:.2e} after {n_steps} steps")


@pytest.mark.gpu
def test_mjwarp_backend_reset_and_controls(tiny_mj_model) -> None:
    backend = MJWarpBackend(_scene(tiny_mj_model))
    for i in range(200):
        if i == 50:
            backend.apply_input(ActuatorSet(values=[0.5]))
        if i == 120:
            backend.apply_input(Neutral())
        backend.step(backend.dt)
    t_before = backend.time
    assert t_before > 0.1
    backend.apply_input(Orbit(azimuth=100.0, elevation=-5.0, distance=100.0))
    backend.reset()
    assert abs(backend.time) < 1e-9
    assert np.abs(backend._d.qpos.numpy()[0]).max() < 1e-6
    # neutral ctrl pushed back after reset
    assert np.allclose(backend._d.ctrl.numpy()[0], 0.0, atol=1e-6)


@pytest.mark.gpu
def test_make_backend_gpu_and_auto(tiny_mj_model) -> None:
    assert make_backend(_scene(tiny_mj_model), Config(backend="gpu")).name == "gpu"
    assert make_backend(_scene(tiny_mj_model), Config(backend="auto")).name == "gpu"
    assert make_backend(_scene(tiny_mj_model), Config(backend="cpu")).name == "cpu"


@pytest.mark.gpu
def test_mjwarp_fly_scene_refused() -> None:
    """The default fly scene (NoSlip) must be refused with a clear error."""
    from flygym_stream.scene import build_default_scene

    with pytest.raises(RuntimeError, match="NoSlip"):
        make_backend(build_default_scene(), Config(backend="gpu"))
