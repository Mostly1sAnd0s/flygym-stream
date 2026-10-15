"""Shared fixtures and fakes for the flygym-stream test suite.

- ``tiny_mj_model``: a real, tiny CPU MuJoCo model (no GL needed) with one
  actuated hinge joint and a named site — enough surface for scene/sim
  tests without the full fly.
- ``FakeRenderer``: stands in for ``rendering.OffscreenRenderer`` (no GPU);
  exposes the same duck-typed interface SimThread drives:
  ``render_as_needed(data) -> bool``, ``render_frame() -> np.ndarray``,
  ``encode_jpeg(frame) -> bytes``, ``reset()``, ``close()``.
- ``FakeBackend``: stands in for a ``simulation.Backend`` (no physics);
  advances a counter as "sim time" and records applied inputs.
"""

from __future__ import annotations

import os

import mujoco as mj
import numpy as np
import pytest
from types import SimpleNamespace

from flygym_stream.config import Config

#: Tiny model: plane + one box on an actuated hinge, with a named site.
TINY_XML = """<mujoco model="tiny">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="body" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.9 0.2 0.1 1"/>
      <site name="target" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" ctrlrange="-3.14 3.14"/>
  </actuator>
</mujoco>"""


#: Free-root tiny model: a heavy free root (box on a freejoint) with one
#: actuated child hinge — the actuator's reaction torque spins the root
#: (rest-anchor tests). No keyframe (named keyframes are created
#: programmatically by flygym, not via MJCF); neutral_restore() falls
#: back to the plain default state for this model.
FREE_TINY_XML = """<mujoco model="free_tiny">
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="root" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.5" rgba="0.9 0.2 0.1 1"/>
      <site name="target" size="0.01"/>
      <body name="pend" pos="0 0.2 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.03 0.03 0.03" rgba="0.2 0.9 0.1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" ctrlrange="-3.14 3.14" kp="5"
              forcelimited="true" forcerange="-1 1"/>
  </actuator>
</mujoco>"""


@pytest.fixture
def tiny_mj_model() -> tuple[mj.MjModel, mj.MjData]:
    """A compiled tiny MuJoCo model plus a fresh data container."""
    m = mj.MjModel.from_xml_string(TINY_XML)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return m, d


@pytest.fixture
def free_tiny_mj_model() -> tuple[mj.MjModel, mj.MjData]:
    """The free-root tiny model (freejoint + actuated child hinge)."""
    m = mj.MjModel.from_xml_string(FREE_TINY_XML)
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return m, d


@pytest.fixture
def tmp_config() -> Config:
    """A small, fast config for tests (320x240 @ 30 fps)."""
    return Config(width=320, height=240, output_fps=30)


class FakeRenderer:
    """Duck-typed stand-in for ``rendering.OffscreenRenderer``.

    Paces ``render_as_needed`` on sim time exactly like the real one
    (render when sim time advanced >= 1/output_fps seconds); the sim data
    container it paces on is injected at construction. Frames are
    synthetic: filled with a per-frame counter so every frame is unique.
    """

    def __init__(self, config: Config, data: object) -> None:
        self.config = config
        self._data = data
        self._interval = 1.0 / config.output_fps
        self._last_render_time = -np.inf
        self._counter = 0
        self.closed = False

    def render_as_needed(self) -> bool:
        """Return True (and update the timer) when a frame is due."""
        t = float(getattr(self._data, "time"))
        if t < self._last_render_time + self._interval - 1e-9:
            return False
        self._last_render_time = t
        return True

    def render_frame(self) -> np.ndarray:
        """Return a synthetic unique RGB frame of the configured size."""
        frame = np.full(
            (self.config.height, self.config.width, 3),
            (self._counter % 250) + 1,
            dtype=np.uint8,
        )
        self._counter += 1
        return frame

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        """Return fake JPEG bytes tagged with the frame counter."""
        return f"JPEG-{self._counter}".encode("ascii")

    def reset(self) -> None:
        self._last_render_time = -np.inf
        self._counter = 0

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_renderer(tmp_config: Config, fake_backend: FakeBackend) -> FakeRenderer:
    return FakeRenderer(tmp_config, fake_backend.data)


class FakeBackend:
    """Duck-typed stand-in for a ``simulation.Backend`` (no physics).

    ``step(dt)`` advances a counter as sim time; ``apply_input(msg)``
    records the message. ``data`` mimics the subset of ``mj.MjData``
    attributes the sim loop and renderer touch (``time``, ``ctrl``).
    """

    def __init__(self, dt: float = 0.001, n_actuators: int = 1) -> None:
        self.dt = dt
        self.n_actuators = n_actuators
        self.name = "fake"
        self.camera_mode = "orbit"  # M7.7 (duck-types Backend.camera_mode)
        self._time = 0.0
        self._steps = 0
        self.inputs: list = []
        self._data = SimpleNamespace(
            time=0.0, ctrl=np.zeros(n_actuators)
        )

    @property
    def data(self) -> SimpleNamespace:
        return self._data

    @property
    def time(self) -> float:
        return self._time

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def orbit(self):
        """Last orbit state applied (duck-types simulation.Backend.orbit;
        None until the first orbit message)."""
        for msg in reversed(self.inputs):
            if type(msg).__name__ == "Orbit":
                return msg
        return None

    def step(self, dt: float) -> None:
        """Advance sim time by one substep of ``dt`` seconds."""
        if abs(dt - self.dt) > 1e-12:
            raise ValueError(f"expected dt={self.dt}, got {dt}")
        self._time += dt
        self._steps += 1
        self._data.time = self._time

    def apply_input(self, msg: object) -> None:
        """Record an input message (no physical effect)."""
        self.inputs.append(msg)

    def apply_state(self, msg: object) -> None:
        """Record a state message (duck-types simulation.Backend).
        Fake data has no qpos/qvel, so state writes are record-only."""
        self.inputs.append(msg)

    def reset(self) -> None:
        self._time = 0.0
        self._steps = 0
        self.inputs.clear()
        self._data.time = 0.0
        self._data.ctrl = np.zeros(self.n_actuators)

    def set_rest_anchor(self, active: bool) -> None:
        """Record rest-anchor control (duck-types simulation.Backend)."""
        self.inputs.append(SimpleNamespace(type_name="SetRestAnchor",
                                           value=active))


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


class FakeClock:
    """Deterministic monotonic clock: ``sleep`` advances the time.

    Injected into SimThread so pacing tests are exact and race-free.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(autouse=True)
def _gl_env_for_gpu_tests(request: pytest.FixtureRequest) -> None:
    """Gpu tests need the headless GL env exported *before* pytest starts:

    mujoco reads MUJOCO_GL at import time, so setting it inside the process
    is too late. Run: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl pytest -m gpu
    """
    if any(m.name == "gpu" for m in request.node.iter_markers()):
        if os.environ.get("MUJOCO_GL") != "egl" or \
                os.environ.get("PYOPENGL_PLATFORM") != "egl":
            pytest.skip("export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl before pytest")
