"""Scene construction v2 (PLAN.md Phase 7, M7.1): SceneConfig + build_scene.

The single builder for every NeuroMechFly scene variant the service runs
(PLAN.md §7.1): DOF/actuator presets, mesh type, world (terrain +
tethered), vision/adhesion options, spawn pose. ``build_default_scene()``
(scene.py) is just ``build_scene(SceneConfig())``; the DRY guard test
allows model construction only in scene.py + this file. Units: model
length is mm (PLAN.md §3: gravity -9810 mm/s², SCALE=1000).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco as mj

from flygym import Simulation as FlygymSimulation
from flygym.anatomy import (ActuatedDOFPreset, AxisOrder, ContactBodiesPreset,
                            JointPreset, Skeleton)
from flygym.compose import (ActuatorType, BlocksTerrainWorld, FlatGroundWorld,
                            GappedTerrainWorld, KinematicPosePreset,
                            MeshType, MixedTerrainWorld, NeuroMechFly,
                            TetheredWorld)
from flygym.utils.math import Rotation3D

from flygym_stream.scene import (DEFAULT_AZIMUTH_DEG,
                                 DEFAULT_ELEVATION_DEG,
                                 DEFAULT_ORBIT_DISTANCE_FACTOR, Scene)

#: Joint-actuator defaults (historical default scene, PLAN.md Phase 3).
ACTUATOR_KP = 50.0
ACTUATOR_CTRL_RANGE = (-3.14, 3.14)
DEFAULT_FORCE_RANGE = (-30.0, 30.0)  # forcelimited=True (MuJoCo default)

_WORLD_CLASSES = {
    "flat": FlatGroundWorld,
    "blocks": BlocksTerrainWorld,
    "gapped": GappedTerrainWorld,
    "mixed": MixedTerrainWorld,
    "tethered": TetheredWorld,
}


@dataclass(frozen=True)
class SceneConfig:
    """Every knob of a buildable NeuroMechFly scene (PLAN.md §7.1).

    Defaults reproduce the historical default scene (full-size mesh, all
    biological joints in yaw-pitch-roll order, 42 leg actuators
    (POSITION), flat ground, tracking camera). ``spawn=None`` (default):
    the root keeps the neutral keyframe pose — the fly starts dorsal-up
    at ~2.1 m, dips to the ground and stands on its feet (rest state);
    an explicit ``spawn`` overrides the root position instead. The
    tethered world (pinned mocap root) falls back to (0, 0, 0.8) when no
    spawn is given. ``world_kwargs`` pass through to the world
    constructor (e.g. flat ``half_size``). The musculoskeletal fly
    (``musculoskeletal=True``) builds the FlyMimic model (M7.6).

    Raises (``__post_init__``):
        ValueError: If ``world`` is not a known world name.
    """

    mesh_type: MeshType = MeshType.FULLSIZE
    joint_preset: JointPreset = JointPreset.ALL_BIOLOGICAL
    axis_order: AxisOrder = AxisOrder.YAW_PITCH_ROLL
    actuated_preset: ActuatedDOFPreset = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    actuator_type: ActuatorType = ActuatorType.POSITION
    actuator_kp: float = ACTUATOR_KP
    ctrl_range: tuple[float, float] = ACTUATOR_CTRL_RANGE
    force_range: tuple[float, float] = DEFAULT_FORCE_RANGE
    contact_bodies: ContactBodiesPreset = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
    world: str = "flat"
    world_kwargs: dict[str, Any] = field(default_factory=dict)
    vision: bool = False
    adhesion: bool = False
    tracking_camera: bool = True
    spawn: tuple[float, float, float] | None = None
    spawn_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    musculoskeletal: bool = False

    def __post_init__(self) -> None:
        if self.world not in _WORLD_CLASSES:
            raise ValueError(
                f"unknown world {self.world!r}; expected one of "
                f"{sorted(_WORLD_CLASSES)}")


def build_scene(config: SceneConfig) -> Scene:
    """Build, compile and reset a Scene per ``config`` (PLAN.md §7.1).

    The scene is reset to the neutral keyframe with a scale-aware default
    orbit (2.8x the fly's extent from the target site, see scene.py).

    Raises:
        ValueError: If no model camera can back the render pass
            (tracking camera off and no eye cameras from vision), or if
            musculoskeletal and vision are combined (experimental in
            flygym — M7.6 rejects it by design).
    """
    if config.musculoskeletal and config.vision:
        raise ValueError(
            "musculoskeletal + vision is experimental in flygym (M7.6); "
            "rebuild without --musculoskeletal or --vision")
    if config.musculoskeletal:
        return _build_musculoskeletal_scene()  # M7.6
    if not config.tracking_camera and not config.vision:
        raise ValueError(
            "a model camera is required for the render pass: keep "
            "tracking_camera=True or enable vision (eye cameras)")

    fly = NeuroMechFly(mesh_type=config.mesh_type)
    skeleton = Skeleton(joint_preset=config.joint_preset,
                        axis_order=config.axis_order)
    fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(config.actuated_preset)
    # Pose-based neutral input is only valid for POSITION actuators;
    # flygym rejects joint angles as inputs for other actuator types.
    neutral = (KinematicPosePreset.NEUTRAL
               if config.actuator_type is ActuatorType.POSITION else None)
    fly.add_actuators(actuated_dofs, config.actuator_type, neutral_input=neutral,
                      kp=config.actuator_kp, ctrlrange=config.ctrl_range,
                      forcelimited=True, forcerange=config.force_range)
    if config.adhesion:
        fly.add_leg_adhesion()
    if config.vision:
        fly.add_vision()
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    if config.tracking_camera:
        fly.add_tracking_camera(name="trackingcam")

    world = _WORLD_CLASSES[config.world](**config.world_kwargs)
    # The tethered world places its pinned (mocap) root at the spawn:
    # it needs a concrete position even when the caller used the
    # default (keyframe-root) rest state.
    world_spawn = (config.spawn if config.spawn is not None
                   else (0.0, 0.0, 0.8))
    rotation = Rotation3D("quat", config.spawn_quat)
    if config.world == "tethered":
        # TetheredWorld pins the root (mocap body); the contact-body kwarg
        # does not apply to its add_fly.
        world.add_fly(fly, world_spawn, rotation)
    else:
        world.add_fly(fly, world_spawn, rotation,
                      bodysegs_with_ground_contact=config.contact_bodies)

    if config.vision:
        # M7.6: flygym.Simulation compiles the world itself and exposes
        # mj_model/mj_data; its get_raw_vision renders the eye cameras
        # (fisheye-corrected) from the SAME MjData the service steps.
        sim_wrapper = FlygymSimulation(world)
        mj_model, mj_data = sim_wrapper.mj_model, sim_wrapper.mj_data
    else:
        sim_wrapper = None
        mj_model, mj_data = world.compile()
    # Home root pose for reset(): an explicit spawn only — the default
    # (spawn=None) rest state keeps the neutral keyframe's own root
    # (dorsal-up; the fly dips to the ground and stands on its feet).
    # Free root only — the tethered world pins the root as a mocap
    # body placed at spawn.
    free_root = mj_model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    home_qpos = ([*config.spawn, *config.spawn_quat]
                 if free_root and config.spawn is not None else None)
    scene = Scene(mj_model, mj_data, actuator_ids=list(range(mj_model.nu)),
                  target_site=fly.name,
                  # Pacing camera for the render pass: the tracking camera,
                  # or the left eye when tracking is off (vision is on).
                  camera_name=(f"{fly.name}/trackingcam"
                               if config.tracking_camera
                               else f"{fly.name}/l_eye_cam_camera"),
                  home_qpos=home_qpos, vision=sim_wrapper)
    scene.reset()  # neutral keyframe + home root + forward pass
    scene.set_orbit_camera(azimuth=DEFAULT_AZIMUTH_DEG,
                           elevation=DEFAULT_ELEVATION_DEG,
                           distance_mm=DEFAULT_ORBIT_DISTANCE_FACTOR
                           * scene.target_extent_m(fly.name) * 1000.0)
    return scene


# --- Phase 7 (M7.6): musculoskeletal (FlyMimic) ------------------------------

def _build_musculoskeletal_scene() -> Scene:
    """Build the FlyMimic musculoskeletal scene (M7.6).

    The self-contained MJCF already provides floor, lighting and an
    anchored (pinned) thorax, so there is no world to compose and no
    spawn pose; the only addition is an orbit-target site at the thorax
    (the MJCF ships no ``nmf`` site). Muscle actuators take activation
    ctrl in (0, 1]; only the left-front leg is muscle-driven in the
    current model (flygym docs).
    """
    import numpy as np
    from flygym.compose import MusculoskeletalFly, MusculoskeletalWorld

    fly = MusculoskeletalFly()
    fly.mjcf_root.body("Thorax").add_site(name=fly.name, size=0.02)
    world = MusculoskeletalWorld(fly)
    mj_model, mj_data = world.compile()
    scene = Scene(mj_model, mj_data,
                  actuator_ids=list(range(mj_model.nu)),
                  target_site=fly.name, camera_name="scene",
                  home_qpos=None)
    scene.reset()
    # Orbit extent: all fly geoms (the MJCF has no common name prefix);
    # exclude the floor so the camera isn't parked across the room.
    site = mj_data.site_xpos[scene._target_site_id]
    extent = 0.0
    for i in range(mj_model.ngeom):
        if mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_GEOM, i) == "floor":
            continue
        size = mj_model.geom_size[i]
        radius = float(size.max()) if size.size else 0.0
        extent = max(extent, float(np.linalg.norm(
            mj_data.geom_xpos[i] - site)) + radius)
    scene.set_orbit_camera(azimuth=DEFAULT_AZIMUTH_DEG,
                           elevation=DEFAULT_ELEVATION_DEG,
                           distance_mm=DEFAULT_ORBIT_DISTANCE_FACTOR
                           * extent * 1000.0)
    return scene
