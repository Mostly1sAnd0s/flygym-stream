"""Scene construction + wrapper for the streaming service (PLAN.md Phase 3).

``build_default_scene`` is the ONLY place the fly model is built (guarded
by tests/test_scene.py::test_model_is_built_only_in_scene_module). ``Scene``
wraps the compiled model + data with the control surface the sim loop and
server drive: reset, set_actuators, orbit camera, step. Orbit camera note
(mujoco 3.x Python): ``MjData``'s free camera is not exposed by the
bindings, so orbit control passes our own persistent ``MjvCamera`` (FREE)
to ``update_scene``; ``update_mjv_camera`` fills it from the stored orbit
+ target site's live position. MuJoCo is SI (m); boundary units: mm.
"""

from collections.abc import Sequence

import mujoco as mj
import numpy as np

# Initial orbit view (PLAN.md §3). The full-size NeuroMechFly is ~3000 mm
# across (wing span); the spawn height is in MuJoCo model units (mm).
DEFAULT_AZIMUTH_DEG, DEFAULT_ELEVATION_DEG = 30.0, -20.0
# Default orbit distance = FACTOR x the fly's extent from the target site:
# at the free camera's 45-degree FOV that frames the whole fly with generous
# margin (a fixed distance is wrong because the model scale is not 1:1
# biological; the full-size NMf is ~3 m across).
DEFAULT_ORBIT_DISTANCE_FACTOR = 3.5
DEFAULT_DISTANCE_MM = 40.0   # generic Scene fallback (tiny-model tests)
MIN_ORBIT_DISTANCE_M = 0.005      # MuJoCo is SI: 5 mm hard floor
MAX_ORBIT_ELEVATION_DEG = 89.0    # avoid the gimbal singularity at +/-90


def geom_extent_m(mj_model: mj.MjModel, mj_data: mj.MjData, site_id: int,
                  geom_prefix: str) -> float:
    """Farthest edge (metres) of any geom named ``geom_prefix*`` from a site.

    The prefix selects the model's own geoms (e.g. the fly's ``nmf/`` geoms
    incl. its ~1.5 m wings) so world geoms like the 1 km ground plane don't
    dominate. Requires a forward pass (fresh ``MjData`` has zero positions).
    """
    site_x = mj_data.site_xpos[site_id]
    extent = 0.0
    for i in range(mj_model.ngeom):
        if not mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_GEOM, i).startswith(
                geom_prefix):
            continue
        center = mj_data.geom_xpos[i]
        radius = float(np.max(mj_model.geom_size[i]))
        extent = max(extent, float(np.linalg.norm(center - site_x)) + radius)
    return extent


class Scene:
    """Compiled MuJoCo scene plus the service's control surface.

    Args:
        mj_model/mj_data: Compiled model + data (owned here).
        actuator_ids: ``mj_data.ctrl`` indices exposed to clients.
        target_site: Site the orbit camera looks at (must exist).
        neutral_keyframe: Keyframe restored by ``reset()`` (plain data reset
            when absent). camera_name: model camera for the render pass
            (may be None). azimuth/elevation/distance_mm: initial orbit.
    """

    def __init__(
        self, mj_model: mj.MjModel, mj_data: mj.MjData, actuator_ids: Sequence[int],
        *, target_site: str = "nmf", neutral_keyframe: str = "neutral",
        camera_name: str | None = None,
        azimuth: float = DEFAULT_AZIMUTH_DEG, elevation: float = DEFAULT_ELEVATION_DEG,
        distance_mm: float = DEFAULT_DISTANCE_MM,
        home_qpos: Sequence[float] | None = None,
    ) -> None:
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.actuator_ids = list(actuator_ids)
        self.camera_name = camera_name
        # Root (qpos[0:7]) to restore on reset(): the spawn pose the scene
        # was built with (the neutral keyframe stores its own root pose,
        # which the spawn overrides).
        self._home_qpos = (np.asarray(home_qpos, dtype=float)
                           if home_qpos is not None else None)
        site_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SITE, target_site)
        if site_id < 0:
            raise ValueError(f"target site {target_site!r} not found in model")
        self._target_site_id = site_id
        self._neutral_key_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_KEY, neutral_keyframe)
        self._neutral_ctrl = self._capture_neutral_ctrl()
        self.set_orbit_camera(azimuth=azimuth, elevation=elevation, distance_mm=distance_mm)

    def _capture_neutral_ctrl(self) -> np.ndarray:
        """Neutral-pose ctrl for the exposed actuators (zeros if no keyframe)."""
        if self._neutral_key_id < 0:
            return np.zeros(len(self.actuator_ids))
        scratch = mj.MjData(self.mj_model)
        mj.mj_resetDataKeyframe(self.mj_model, scratch, self._neutral_key_id)
        return scratch.ctrl[self.actuator_ids].copy()

    @property
    def n_actuators(self) -> int:
        """Number of client-controllable actuators."""
        return len(self.actuator_ids)

    def step(self) -> None:
        """Advance the simulation by one substep (the model's dt)."""
        mj.mj_step(self.mj_model, self.mj_data)

    def reset(self) -> None:
        """Restore the initial state: the neutral keyframe (or a plain
        data reset) plus the home root pose (spawn), then forward."""
        if self._neutral_key_id >= 0:
            mj.mj_resetDataKeyframe(self.mj_model, self.mj_data, self._neutral_key_id)
        else:
            mj.mj_resetData(self.mj_model, self.mj_data)
        if self._home_qpos is not None:
            self.mj_data.qpos[0:7] = self._home_qpos
        mj.mj_forward(self.mj_model, self.mj_data)

    def set_actuators(self, values: Sequence[float]) -> None:
        """Set ctrl of the exposed actuators (order = actuator_ids)."""
        if len(values) != len(self.actuator_ids):
            raise ValueError(f"expected {len(self.actuator_ids)} values, got {len(values)}")
        for ctrl_id, value in zip(self.actuator_ids, values):
            self.mj_data.ctrl[ctrl_id] = float(value)

    def set_actuator(self, index: int, value: float) -> None:  # bounds-checked
        """Set one exposed actuator's ctrl."""
        if index not in self.actuator_ids:
            raise ValueError(f"actuator index {index} not exposed")
        self.mj_data.ctrl[index] = float(value)

    def neutral_actuators(self) -> np.ndarray:
        """Neutral-pose ctrl values for the exposed actuators."""
        return self._neutral_ctrl.copy()

    def set_orbit_camera(self, *, azimuth: float, elevation: float,
                         distance_mm: float) -> None:
        """Store the orbit state around the target site (clamped).

        Args: azimuth/elevation in degrees (elevation clamped to
            +/-MAX_ORBIT_ELEVATION_DEG); distance_mm in mm (stored in m).
        """
        self._azimuth = float(azimuth)
        self._elevation = float(np.clip(elevation, -MAX_ORBIT_ELEVATION_DEG,
                                        MAX_ORBIT_ELEVATION_DEG))
        self._distance_m = max(float(distance_mm) / 1000.0, MIN_ORBIT_DISTANCE_M)

    @property
    def orbit(self) -> tuple[float, float, float]:
        """Current orbit state: (azimuth_deg, elevation_deg, distance_mm)."""
        return self._azimuth, self._elevation, self._distance_m * 1000.0

    def target_extent_m(self, geom_prefix: str) -> float:
        """Farthest edge (m) of geoms named ``geom_prefix*`` from the target."""
        return geom_extent_m(self.mj_model, self.mj_data,
                             self._target_site_id, geom_prefix)

    def update_mjv_camera(self, cam: mj.MjvCamera) -> None:
        """Fill ``cam`` from stored orbit + live target position (tracks fly)."""
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.azimuth = self._azimuth
        cam.elevation = self._elevation
        cam.distance = self._distance_m
        cam.lookat[:] = self.mj_data.site_xpos[self._target_site_id]


def build_default_scene() -> Scene:
    """Build the default fly scene (PLAN.md Phase 3), ready to run.

    Thin wrapper over ``scenes.build_scene`` with ``SceneConfig()``
    defaults (Phase 7 M7.1): NeuroMechFly + all biological joints +
    position actuators on the active leg DOFs, colorized, in a
    FlatGroundWorld; reset to the neutral keyframe with the scale-aware
    default orbit view (pattern: flygym's viewer script).
    """
    # Imported here (not at module top) to avoid a scene <-> scenes cycle:
    # scenes.py imports Scene + the orbit constants from this module.
    from flygym_stream.scenes import SceneConfig, build_scene

    return build_scene(SceneConfig())
