"""Tests for flygym_stream.scene (PLAN.md Phase 3).

Tiny-model tests run everywhere (no GL, no network). Full-fly tests are
marked ``network`` because a cold asset cache triggers the one-time S3
mesh fetch; on a warm cache they run offline.
"""

from __future__ import annotations

import pathlib

import mujoco as mj
import pytest

from flygym_stream import scene as scene_mod
from flygym_stream.scene import Scene, build_default_scene

SRC = pathlib.Path(scene_mod.__file__).parent


def make_tiny_scene(tiny_mj_model) -> Scene:
    """Wrap the tiny fixture model in a Scene (site 'target', no keyframe)."""
    m, d = tiny_mj_model
    return Scene(m, d, actuator_ids=[0], target_site="target")


def test_scene_step_advances_time(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    t0 = s.mj_data.time
    s.step()
    assert s.mj_data.time == pytest.approx(t0 + s.mj_model.opt.timestep)


def test_scene_set_actuators_writes_ctrl(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    s.set_actuators([0.5])
    assert s.mj_data.ctrl[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        s.set_actuators([0.1, 0.2])  # wrong length


def test_scene_reset_restores_state(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    s.set_actuators([0.9])
    s.mj_data.qvel[0] = 3.0
    s.mj_data.time = 7.0
    s.step()
    s.reset()
    assert s.mj_data.ctrl[0] == pytest.approx(0.0)
    assert s.mj_data.qvel[0] == pytest.approx(0.0)
    assert s.mj_data.time == pytest.approx(0.0)


def test_scene_set_orbit_camera_fills_mjv_camera(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    s.set_orbit_camera(azimuth=30.0, elevation=-15.0, distance_mm=100.0)
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    assert cam.type == mj.mjtCamera.mjCAMERA_FREE
    assert cam.azimuth == pytest.approx(30.0)
    assert cam.elevation == pytest.approx(-15.0)
    assert cam.distance == pytest.approx(0.1)  # MuJoCo is SI: meters
    expected_lookat = s.mj_data.site_xpos[
        mj.mj_name2id(s.mj_model, mj.mjtObj.mjOBJ_SITE, "target")
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(cam.lookat, expected_lookat))


def test_scene_orbit_camera_follows_moving_target(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    s.set_actuators([1.0])
    s.step()
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    site_id = mj.mj_name2id(s.mj_model, mj.mjtObj.mjOBJ_SITE, "target")
    expected = s.mj_data.site_xpos[site_id]
    assert all(abs(a - b) < 1e-9 for a, b in zip(cam.lookat, expected))


def test_scene_clamps_orbit_elevation_and_distance(tiny_mj_model) -> None:
    s = make_tiny_scene(tiny_mj_model)
    s.set_orbit_camera(azimuth=0.0, elevation=120.0, distance_mm=1.0)
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    assert cam.elevation < 90.0
    assert cam.distance >= scene_mod.MIN_ORBIT_DISTANCE_M


def test_scene_unknown_target_site_rejected(tiny_mj_model) -> None:
    m, d = tiny_mj_model
    with pytest.raises(ValueError):
        Scene(m, d, actuator_ids=[0], target_site="nope")


@pytest.fixture(scope="module")
def fly_scene() -> Scene:
    """The full default fly scene (may fetch S3 meshes on a cold cache)."""
    return build_default_scene()


@pytest.mark.network
def test_build_default_scene_compiles(fly_scene: Scene) -> None:
    m = fly_scene.mj_model
    assert m.nu > 0
    assert fly_scene.actuator_ids == list(range(m.nu))
    key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
    assert key >= 0
    assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "nmf") >= 0


@pytest.mark.network
def test_build_default_scene_neutral_reset_is_stable(fly_scene: Scene) -> None:
    neutral_qpos = fly_scene.mj_data.qpos.copy()
    for _ in range(50):
        fly_scene.step()
    assert not all(abs(a - b) < 1e-6 for a, b in zip(fly_scene.mj_data.qpos, neutral_qpos))
    fly_scene.reset()
    assert all(abs(a - b) < 1e-6 for a, b in zip(fly_scene.mj_data.qpos, neutral_qpos))


def test_model_is_built_only_in_scene_module() -> None:
    """DRY guard: the fly model is constructed in scene.py and nowhere else."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "scene.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "NeuroMechFly(" in text or "FlatGroundWorld(" in text:
            offenders.append(path.name)
    assert not offenders, f"model construction leaked into: {offenders}"
