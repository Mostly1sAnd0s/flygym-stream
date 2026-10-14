"""Scene construction + wrapper for the streaming service (PLAN.md Phase 3).

``build_default_scene`` is the ONLY place the fly model is built (guarded
by tests/test_scene.py::test_model_is_built_only_in_scene_module). ``Scene``
wraps the compiled model + data and exposes the control surface the sim
loop and server drive: reset, set_actuators, orbit camera, step.

Orbit camera note (mujoco 3.x Python): ``MjData``'s free camera is NOT
exposed by the bindings and ``update_scene(data, -1)`` builds a fresh
default free camera per call, so orbit control passes our own persistent
``MjvCamera`` (type FREE) to ``update_scene``; ``update_mjv_camera`` fills
it from the stored orbit + the target site's live world position. MuJoCo
is SI (meters); distances cross the API boundary in mm.
"""

from collections.abc import Sequence

import mujoco as mj
import numpy as np

from flygym.anatomy import (ActuatedDOFPreset, AxisOrder, ContactBodiesPreset,
                            JointPreset, Skeleton)
from flygym.compose import (ActuatorType, FlatGroundWorld, KinematicPosePreset,
                            MeshType, NeuroMechFly)
from flygym.utils.math import Rotation3D

# Initial orbit view (PLAN.md §3): fly is ~5 mm long, spawns at z = 0.8 mm.
DEFAULT_AZIMUTH_DEG = 30.0
DEFAULT_ELEVATION_DEG = -20.0
DEFAULT_DISTANCE_MM = 40.0
MIN_ORBIT_DISTANCE_M = 0.005      # MuJoCo is SI: 5 mm hard floor
MAX_ORBIT_ELEVATION_DEG = 89.0    # avoid the gimbal singularity at +/-90

# Scene composition constants (pattern: flygym's launch_interactive_viewer).
SPAWN_POSITION = (0.0, 0.0, 0.8)  # xyz, mm
SPAWN_ROTATION = Rotation3D("quat", (1.0, 0.0, 0.0, 0.0))  # identity, wxyz
ACTUATOR_KP = 50.0
ACTUATOR_CTRL_RANGE = (-3.14, 3.14)


class Scene:
    """Compiled MuJoCo scene plus the service's control surface.

    Args:
        mj_model/mj_data: Compiled model + data container (owned here).
        actuator_ids: ``mj_data.ctrl`` indices exposed to clients.
        target_site: Site the orbit camera looks at (must exist).
        neutral_keyframe: Keyframe restored by ``reset()`` (falls back to a
            plain data reset when absent).
        azimuth/elevation/distance_mm: initial orbit camera.
    """

    def __init__(
        self, mj_model: mj.MjModel, mj_data: mj.MjData, actuator_ids: Sequence[int],
        *, target_site: str = "nmf", neutral_keyframe: str = "neutral",
        azimuth: float = DEFAULT_AZIMUTH_DEG, elevation: float = DEFAULT_ELEVATION_DEG,
        distance_mm: float = DEFAULT_DISTANCE_MM,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.actuator_ids = list(actuator_ids)
        site_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SITE, target_site)
        if site_id < 0:
            raise ValueError(f"target site {target_site!r} not found in model")
        self._target_site_id = site_id
        self._neutral_key_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY,
                                             neutral_keyframe)
        self.set_orbit_camera(azimuth=azimuth, elevation=elevation,
                              distance_mm=distance_mm)

    @property
    def n_actuators(self) -> int:
        """Number of client-controllable actuators."""
        return len(self.actuator_ids)

    def step(self) -> None:
        """Advance the simulation by one substep (the model's dt)."""
        mj.mj_step(self.mj_model, self.mj_data)

    def reset(self) -> None:
        """Restore the neutral keyframe (or a plain data reset), then forward."""
        if self._neutral_key_id >= 0:
            mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, self._neutral_key_id)
        else:
            mj.mj_resetData(self.mj_model, self.mj_data)
        mj.mj_forward(self.mj_model, self.mj_data)

    def set_actuators(self, values: Sequence[float]) -> None:
        """Set ctrl of the exposed actuators (order = actuator_ids)."""
        if len(values) != len(self.actuator_ids):
            raise ValueError(f"expected {len(self.actuator_ids)} values, "
                             f"got {len(values)}")
        for ctrl_id, value in zip(self.actuator_ids, values):
            self.mj_data.ctrl[ctrl_id] = float(value)

    def set_orbit_camera(self, *, azimuth: float, elevation: float,
                         distance_mm: float) -> None:
        """Store the orbit state around the target site (clamped).

        Args:
            azimuth: Degrees, any range.
            elevation: Degrees, clamped to +/-MAX_ORBIT_ELEVATION_DEG.
            distance_mm: Camera-to-target distance in mm (stored in meters).
        """
        self._azimuth = float(azimuth)
        self._elevation = float(np.clip(elevation, -MAX_ORBIT_ELEVATION_DEG,
                                        MAX_ORBIT_ELEVATION_DEG))
        self._distance_m = max(float(distance_mm) / 1000.0, MIN_ORBIT_DISTANCE_M)

    def update_mjv_camera(self, cam: mj.MjvCamera) -> None:
        """Fill ``cam`` from the stored orbit + live target site position.

        The renderer calls this before every ``update_scene`` so the camera
        tracks the fly.
        """
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self._azimuth
        cam.elevation = self._elevation
        cam.distance = self._distance_m
        cam.lookat[:] = self.mj_data.site_xpos[self._target_site_id]


def build_default_scene(*, mesh_type: MeshType = MeshType.FULLSIZE) -> Scene:
    """Build the default fly scene (PLAN.md Phase 3), ready to run.

    Pattern from flygym's launch_interactive_viewer.py: NeuroMechFly +
    all biological joints + position actuators on the active leg DOFs,
    colorized, in a FlatGroundWorld; reset to the neutral keyframe with
    the default orbit view.
    """
    fly = NeuroMechFly(mesh_type=mesh_type)
    skeleton = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL,
                        axis_order=AxisOrder.YAW_PITCH_ROLL)
    fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated_dofs, ActuatorType.POSITION,
                      neutral_input=KinematicPosePreset.NEUTRAL,
                      kp=ACTUATOR_KP, ctrlrange=ACTUATOR_CTRL_RANGE)
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")
    world = FlatGroundWorld()
    world.add_fly(fly, SPAWN_POSITION, SPAWN_ROTATION,
                  bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
    mj_model, mj_data = world.compile()
    scene = Scene(mj_model, mj_data, actuator_ids=list(range(mj_model.nu)),
                  target_site=fly.name)
    scene.reset()  # neutral keyframe
    return scene
