"""Tests for flygym_stream.scene (PLAN.md Phase 3).

Tiny-model tests run everywhere (no GL, no network). Full-fly tests are
marked ``network`` because a cold asset cache triggers the one-time S3
mesh fetch; on a warm cache they run offline.
"""

from __future__ import annotations

import math
import pathlib

import mujoco as mj
import numpy as np
import pytest

from flygym_stream import scene as scene_mod
from flygym_stream.scene import Scene, build_default_scene

SRC = pathlib.Path(scene_mod.__file__).parent


def make_tiny_scene(tiny_mj_model) -> Scene:
    """Wrap the tiny fixture model in a Scene (site 'target', no keyframe)."""
    m, d = tiny_mj_model
    return Scene(m, d, actuator_ids=[0], target_site="target")


def make_free_tiny_scene(free_tiny_mj_model) -> Scene:
    """Wrap the free-root tiny model (site 'target', no keyframe)."""
    m, d = free_tiny_mj_model
    return Scene(m, d, actuator_ids=[0], target_site="target")


def quat_angle(q0, q1) -> float:
    """Angle (rad) between two quaternions (sign-independent)."""
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


# --- Rest anchor (default rest state: the fly holds its reference pose) ---


def test_rest_anchor_holds_root_pose(free_tiny_mj_model) -> None:
    """Default rest anchor: the free root's orientation is held (its
    angular velocity is zeroed every substep) while the actuated
    appendage stays dynamic. Without the anchor the actuator's reaction
    torque spins the root (ctrl target != joint value, like the fly's
    neutral pose driving its leg actuators)."""
    scene = make_free_tiny_scene(free_tiny_mj_model)
    assert scene.rest_anchor is True  # on by default
    scene.reset()
    scene.mj_data.ctrl[0] = 1.0  # actuator demand -> reaction torque
    quat0 = scene.mj_data.qpos[3:7].copy()
    hinge0 = float(scene.mj_data.qpos[7])
    for _ in range(300):
        scene.step()
    assert float(np.abs(scene.mj_data.qvel[3:6]).max()) < 1e-12  # mechanism
    assert abs(float(scene.mj_data.qpos[7]) - hinge0) > 0.01  # appendage free
    assert quat_angle(quat0, scene.mj_data.qpos[3:7]) < 0.02  # root held
    scene.reset()
    scene.set_rest_anchor(False)
    scene.mj_data.ctrl[0] = 1.0
    quat1 = scene.mj_data.qpos[3:7].copy()
    for _ in range(300):
        scene.step()
    assert quat_angle(quat1, scene.mj_data.qpos[3:7]) > 0.2  # root spun


def test_state_write_disables_rest_anchor(free_tiny_mj_model) -> None:
    """Explicit state writes take over the physics: spawn/set_state turn
    the rest anchor off; reset() returns to the anchored rest state."""
    scene = make_free_tiny_scene(free_tiny_mj_model)
    assert scene.rest_anchor is True
    scene.set_root_pose((0.1, 0.2, 0.5), (1.0, 0.0, 0.0, 0.0))
    assert scene.rest_anchor is False
    scene.reset()
    assert scene.rest_anchor is True
    scene.apply_state_arrays(qpos=scene.mj_data.qpos.copy())
    assert scene.rest_anchor is False


def test_neutral_restore_full_state(free_tiny_mj_model) -> None:
    """neutral_restore() restores the full neutral state (qpos/qvel/ctrl;
    keyframe state when a keyframe exists, plain default state when not)
    and re-enables the rest anchor, without touching the sim clock."""
    m, d = free_tiny_mj_model
    scene = make_free_tiny_scene(free_tiny_mj_model)
    fresh = mj.MjData(m)
    mj.mj_forward(m, fresh)
    qpos0, qvel0, ctrl0 = (fresh.qpos.copy(), fresh.qvel.copy(),
                           fresh.ctrl.copy())
    scene.step()  # advance the clock + move the state away
    t_before = float(d.time)
    d.qpos[0] += 0.3
    d.qvel[0] = 2.0
    d.ctrl[0] = 0.123
    scene.set_rest_anchor(False)
    scene.neutral_restore()
    assert np.allclose(d.qpos, qpos0)
    assert np.allclose(d.qvel, qvel0)
    assert np.allclose(d.ctrl, ctrl0)
    assert scene.rest_anchor is True
    assert d.time == pytest.approx(t_before)  # clock untouched


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


# --- Camera views: viewcube + 1st/3rd person (body-frame cameras) -----------

#: Tiny two-body model for camera-view tests: a free root on "Thorax"
#: (body names mirror the FlyMimic MSK fallback chain) carrying a "Head" at
#: +x (forward = thorax -> head, matching the real fly: rostrum/eyes lead,
#: abdomen/wings trail). No keyframe, no actuator; the target site sits at
#: the thorax.
CAM_TINY_XML = """<mujoco model="cam_tiny">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="Thorax" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05" rgba="0.9 0.2 0.1 1"/>
      <body name="Head" pos="0.2 0 0">
        <geom type="sphere" size="0.1" rgba="0.2 0.9 0.1 1"/>
      </body>
    </body>
    <site name="target" pos="0 0 0.3" size="0.01"/>
  </worldbody>
</mujoco>"""


def make_cam_tiny_scene() -> Scene:
    m = mj.MjModel.from_xml_string(CAM_TINY_XML)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return Scene(m, d, actuator_ids=[], target_site="target")


def cam_pos(cam) -> tuple[float, float, float]:
    """Reconstruct the free-camera position from its az/el/dist/lookat."""
    az = math.radians(cam.azimuth)
    el = math.radians(cam.elevation)
    look = np.asarray(cam.lookat, dtype=float)
    off = cam.distance * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
         math.sin(el)])
    return tuple(look + off)


def test_camera_defaults_to_orbit_mode() -> None:
    s = make_cam_tiny_scene()
    assert s.camera_mode == "orbit"


def test_fly_forward_and_heading_follow_the_root() -> None:
    s = make_cam_tiny_scene()
    # At the identity root the head leads the thorax along +x.
    assert np.allclose(s.fly_forward(), [1.0, 0.0, 0.0], atol=1e-6)
    assert s.fly_heading_deg() == pytest.approx(0.0)
    # Rotate the root 90 deg about z: forward becomes +y, heading 90.
    s.mj_data.qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0,
                           math.sin(math.pi / 4)]
    mj.mj_forward(s.mj_model, s.mj_data)
    assert np.allclose(s.fly_forward(), [0.0, 1.0, 0.0], atol=1e-6)
    assert s.fly_heading_deg() == pytest.approx(90.0)


def test_viewcube_snaps_orbit_relative_to_the_fly_heading() -> None:
    s = make_cam_tiny_scene()
    s.set_orbit_camera(azimuth=30.0, elevation=-20.0, distance_mm=200.0)
    # At rest the fly faces +x (heading 0): viewcube directions are world.
    s.set_orbit_view("front")
    assert s.camera_mode == "orbit"
    assert s.orbit[0] == pytest.approx(0.0)
    assert s.orbit[1] == pytest.approx(-20.0)  # view keeps the elevation
    assert s.orbit[2] == pytest.approx(200.0)  # ... and the distance
    s.set_orbit_view("back")
    assert s.orbit[0] == pytest.approx(180.0)
    s.set_orbit_view("left")
    assert s.orbit[0] == pytest.approx(90.0)   # camera on the fly's left
    s.set_orbit_view("right")
    assert s.orbit[0] == pytest.approx(-90.0)
    s.set_orbit_view("top")
    assert s.orbit[0] == pytest.approx(-90.0)  # azimuth kept
    assert s.orbit[1] == pytest.approx(scene_mod.MAX_ORBIT_ELEVATION_DEG)
    s.set_orbit_view("bottom")
    assert s.orbit[1] == pytest.approx(-scene_mod.MAX_ORBIT_ELEVATION_DEG)
    # Fly-relative: rotate the root 90 deg (heading 90), front now faces +y.
    s.mj_data.qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0,
                           math.sin(math.pi / 4)]
    mj.mj_forward(s.mj_model, s.mj_data)
    s.set_orbit_view("front")
    assert s.orbit[0] == pytest.approx(90.0)


def test_first_person_camera_sits_on_top_of_head_looking_forward() -> None:
    s = make_cam_tiny_scene()
    s.set_camera_mode("first")
    assert s.camera_mode == "first"
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    head = s.mj_data.xpos[mj.mj_name2id(
        s.mj_model, mj.mjtObj.mjOBJ_BODY, "Head")]
    pos = cam_pos(cam)
    look = np.asarray(cam.lookat, dtype=float)
    # Looking well ahead of the head, along +x (the fly's forward).
    assert look[0] > head[0] + 0.5 and abs(look[1] - head[1]) < 1e-6
    # The camera itself is at the top of the head: laterally at the head,
    # above it by the head geom radius (0.1 in this model).
    assert abs(pos[0] - head[0]) < 1e-6 and abs(pos[1] - head[1]) < 1e-6
    assert pos[2] == pytest.approx(head[2] + 0.1, abs=1e-6)


def test_first_person_camera_follows_a_moving_fly() -> None:
    s = make_cam_tiny_scene()
    s.set_camera_mode("first")
    cam0 = mj.MjvCamera()
    s.update_mjv_camera(cam0)
    s.mj_data.qpos[0:3] += [1.0, 2.0, 0.0]  # teleport the free root
    mj.mj_forward(s.mj_model, s.mj_data)
    cam1 = mj.MjvCamera()
    s.update_mjv_camera(cam1)
    delta = np.asarray(cam1.lookat) - np.asarray(cam0.lookat)
    assert np.allclose(delta, [1.0, 2.0, 0.0], atol=1e-6)


def test_third_person_camera_trails_behind_and_above() -> None:
    s = make_cam_tiny_scene()
    s.set_camera_mode("third")
    cam = mj.MjvCamera()
    s.update_mjv_camera(cam)
    head = s.mj_data.xpos[mj.mj_name2id(
        s.mj_model, mj.mjtObj.mjOBJ_BODY, "Head")]
    pos = cam_pos(cam)
    look = np.asarray(cam.lookat, dtype=float)
    assert pos[0] < head[0]          # behind the head (fly faces +x)
    assert pos[2] > head[2]          # ... and above it
    assert look[0] > head[0]         # looking forward, past the head


def test_orbit_mode_captures_the_body_view_on_exit() -> None:
    """Switching a body view back to orbit adopts the current effective
    camera as the stored orbit: the view continues without a jump."""
    s = make_cam_tiny_scene()
    s.set_camera_mode("first")
    az, el, dist = s.orbit  # effective body-view camera
    s.set_camera_mode("orbit")
    assert s.camera_mode == "orbit"
    assert s.orbit == (pytest.approx(az), pytest.approx(el),
                       pytest.approx(dist))


def test_set_orbit_view_switches_back_to_orbit_mode() -> None:
    s = make_cam_tiny_scene()
    s.set_camera_mode("third")
    s.set_orbit_view("front")
    assert s.camera_mode == "orbit"


def test_invalid_camera_mode_and_view_rejected() -> None:
    s = make_cam_tiny_scene()
    with pytest.raises(ValueError):
        s.set_camera_mode("side")
    with pytest.raises(ValueError):
        s.set_orbit_view("up")


def test_orbit_property_tracks_effective_body_view() -> None:
    """In body-view modes the orbit readout (status source) is the live
    effective camera, not the stale stored orbit."""
    s = make_cam_tiny_scene()
    s.set_orbit_camera(azimuth=30.0, elevation=-20.0, distance_mm=200.0)
    s.set_camera_mode("third")
    az, el, dist = s.orbit
    eff_az, eff_el, eff_dist, _ = s.effective_camera()
    assert (az, el, dist) == (pytest.approx(eff_az), pytest.approx(eff_el),
                              pytest.approx(eff_dist * 1000.0))


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


@pytest.mark.network
def test_default_orbit_distance_is_scale_aware(fly_scene: Scene) -> None:
    """The full-size NeuroMechFly is ~3 m across (wing span), not the 5 mm
    a hardcoded 40 mm default assumed; the default orbit distance must scale
    with the model so the initial view frames the whole fly instead of
    sitting inside its body (drag then appears to do nothing)."""
    az, el, dist_mm = fly_scene.orbit
    extent_m = fly_scene.target_extent_m("nmf")
    assert extent_m > 1.0, f"full-size fly should be metres across, got {extent_m} m"
    assert dist_mm / 1000.0 >= 1.8 * extent_m, (
        f"default distance {dist_mm} mm too close for a {extent_m:.1f} m fly")


@pytest.mark.network
def test_target_extent_filters_world_geoms(fly_scene: Scene) -> None:
    """target_extent_m counts only geoms the prefix selects (the fly); an
    unfiltered extent would be dominated by the 1 km ground plane."""
    assert fly_scene.target_extent_m("nmf") < 10.0
    assert fly_scene.target_extent_m("") > 1000.0


@pytest.mark.network
def test_scene_exposes_current_orbit_state(fly_scene: Scene) -> None:
    """The orbit state (az/el deg, distance mm) is readable back so the
    server can report it in status and the client never drifts."""
    fly_scene.set_orbit_camera(azimuth=123.0, elevation=5.0, distance_mm=6400.0)
    assert fly_scene.orbit == (123.0, 5.0, pytest.approx(6400.0))


@pytest.mark.network
def test_body_view_cameras_frame_the_real_fly(fly_scene: Scene) -> None:
    """On the real fly the body-view cameras frame the animal: 1st person
    sits on top of the head looking forward (+x at rest, rostrum leads);
    3rd person trails behind and above, looking ahead."""
    s = fly_scene
    head = s.mj_data.xpos[mj.mj_name2id(
        s.mj_model, mj.mjtObj.mjOBJ_BODY, "nmf/c_head")].copy()
    s.set_camera_mode("first")
    az, el, dist, look = s.effective_camera()
    look = np.asarray(look)
    off = dist * np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                           math.cos(math.radians(el)) * math.sin(math.radians(az)),
                           math.sin(math.radians(el))])
    assert look[0] > head[0] + 0.5            # well ahead of the head
    pos = look + off
    assert pos[2] > head[2] + 0.2             # above the head center
    assert abs(pos[0] - head[0]) < 0.5        # laterally at the head
    s.set_camera_mode("third")
    az, el, dist, look = s.effective_camera()
    look = np.asarray(look)
    off = dist * np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)),
                           math.cos(math.radians(el)) * math.sin(math.radians(az)),
                           math.sin(math.radians(el))])
    pos = look + off
    assert pos[0] < head[0] - 0.5             # behind the head (fly faces +x)
    assert pos[2] > head[2]                   # ... and above
    assert look[0] > head[0]                  # looking ahead of the head
    s.set_camera_mode("orbit")  # leave the fixture in a sane view


def test_model_is_built_only_in_scene_module() -> None:
    """DRY guard: the fly model is constructed in the scene layer
    (scene.py / scenes.py, PLAN.md Phase 7 M7.1) and nowhere else."""
    allowed = {"scene.py", "scenes.py"}
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "NeuroMechFly(" in text or "FlatGroundWorld(" in text:
            offenders.append(path.name)
    assert not offenders, f"model construction leaked into: {offenders}"
