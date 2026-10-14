"""Simulation loop + backends (PLAN.md Phase 4).

One stepping interface (Strategy, PLAN.md §2.2): ``Backend`` has a CPU
(MuJoCo) and a GPU (MJWarp, optional) implementation. ``SimThread`` owns
the loop: drain inputs -> dispatch -> pace & step -> render-if-due ->
FrameQueue. Pacing is wall-clock-anchored so sim time advances at
``playback_speed`` x real time; ``clock``/``sleep`` are injectable for
exact, race-free tests.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod

from flygym_stream.config import Config
from flygym_stream.protocol import (
    Actuator, ActuatorSet, InputMessage, Neutral, Orbit, Pause, Reset,
    Resume, Speed, Step,
)
from flygym_stream.scene import Scene
from flygym_stream.stream import FrameQueue, InputQueue

logger = logging.getLogger(__name__)
MAX_CATCHUP_STEPS = 2000  # per-tick cap: avoid pathological catch-up spirals


class Backend(ABC):
    """Stepping interface shared by the CPU and GPU paths.

    ``apply_input`` only receives simulation-affecting messages
    (Actuator, ActuatorSet, Neutral, Orbit); loop-control messages are
    dispatched by SimThread itself.
    """

    dt: float
    name: str
    n_actuators: int

    def __init__(self) -> None:
        """Base: record applied inputs (test hooks + debugging)."""
        self.inputs: list[InputMessage] = []

    @property
    @abstractmethod
    def time(self) -> float:
        """Current simulation time in seconds."""

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance one substep of ``dt`` seconds."""

    @abstractmethod
    def apply_input(self, msg: InputMessage) -> None:
        """Apply a validated, simulation-affecting control message."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to its initial state."""


class MuJoCoBackend(Backend):
    """CPU stepping via MuJoCo, driven through the Scene control surface."""

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        self._scene = scene
        self.dt = scene.mj_model.opt.timestep
        self.name = "cpu"
        self.n_actuators = scene.n_actuators

    @property
    def time(self) -> float:
        return self._scene.mj_data.time

    def step(self, dt: float) -> None:
        """Advance one substep; ``dt`` must equal the model timestep."""
        if abs(dt - self.dt) > 1e-12:
            raise ValueError(f"dt {dt} != model timestep {self.dt}")
        self._scene.step()

    def apply_input(self, msg: InputMessage) -> None:
        """Route a simulation-affecting message to the Scene."""
        if isinstance(msg, Actuator):
            self._scene.set_actuator(msg.index, msg.value)
        elif isinstance(msg, ActuatorSet):
            self._scene.set_actuators(msg.values)
        elif isinstance(msg, Neutral):
            self._scene.set_actuators(self._scene.neutral_actuators())
        elif isinstance(msg, Orbit):
            self._scene.set_orbit_camera(azimuth=msg.azimuth, elevation=msg.elevation,
                                         distance_mm=msg.distance)
        else:
            raise TypeError(f"not a simulation input: {type(msg).__name__}")
        self.inputs.append(msg)

    def reset(self) -> None:
        """Reset the model to its initial (neutral) state."""
        self._scene.reset()


class MJWarpBackend(Backend):
    """GPU stepping via MJWarp (optional 'gpu' extra + NVIDIA CUDA).

    The GPU stack is imported lazily so the service runs CPU-only by
    default; without the extra, construction fails with an actionable
    error (unit-tested). Real mjw stepping lands in Phase 6.
    """

    dt, name, n_actuators = 0.0, "gpu", 0
    _NOT_YET = "MJWarp stepping lands in Phase 6"

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        try:
            import mujoco_warp  # noqa: F401  (lazy import, PLAN.md §4)
        except ImportError as exc:
            raise RuntimeError(
                "backend 'gpu' needs the 'gpu' extra (warp-lang + mujoco_warp) "
                "with NVIDIA CUDA; install it (`uv sync --extra gpu`) or use "
                "--backend cpu"
            ) from exc
        raise RuntimeError(f"{self._NOT_YET}; use --backend cpu")

    @property
    def time(self) -> float:
        raise NotImplementedError(self._NOT_YET)

    def step(self, dt: float) -> None:
        raise NotImplementedError(self._NOT_YET)

    def apply_input(self, msg: InputMessage) -> None:
        raise NotImplementedError(self._NOT_YET)

    def reset(self) -> None:
        raise NotImplementedError(self._NOT_YET)


def make_backend(scene: Scene, config: Config) -> Backend:
    """Pick the backend for ``config.backend`` ('auto' = GPU when the
    extra is importable, else CPU with a logged note)."""
    if config.backend == "cpu":
        return MuJoCoBackend(scene)
    try:
        return MJWarpBackend(scene)
    except RuntimeError as exc:
        if config.backend == "gpu":
            raise
        logger.warning("auto: falling back to the CPU backend: %s", exc)
        return MuJoCoBackend(scene)


class SimThread:
    """Runs the sim loop on its own thread (or tick-by-tick in tests).

    tick(): drain inputs -> dispatch -> (if not paused) step as many
    substeps as wall clock demands at ``playback_speed`` -> if the
    renderer says a frame is due, render + encode + push to FrameQueue.
    """

    def __init__(self, backend: Backend, renderer: object, frames: FrameQueue,
                 inputs: InputQueue, config: Config, *, clock=time.monotonic,
                 sleep=time.sleep) -> None:
        self._backend = backend
        self._renderer = renderer
        self._frames = frames
        self._inputs = inputs
        self._speed = float(config.playback_speed)
        self._clock = clock
        self._sleep = sleep
        self._paused = False
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._rebase()

    def start(self) -> None:
        """Start the loop thread (re-anchors the pacing clock)."""
        if self._thread is not None:
            raise RuntimeError("SimThread is already running")
        self._rebase()
        self._thread = threading.Thread(target=self._run, name="flygym-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop thread and wait for it (idempotent)."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Freeze the sim loop (PLAN.md §3 ``pause``)."""
        self._paused = True

    def resume(self) -> None:
        """Unfreeze the sim loop, re-anchoring pacing (``resume``)."""
        self._paused = False
        self._rebase()

    def set_speed(self, speed: float) -> None:
        """Change playback speed live, re-anchoring the wall-clock base."""
        if not 0 < speed <= 100.0:
            raise ValueError(f"speed must be in (0, 100], got {speed}")
        self._speed = float(speed)
        self._rebase()

    def step(self, n: int) -> None:
        """While paused: drain inputs, advance exactly n substeps, render."""
        if not self._paused:
            raise RuntimeError("step(n) requires a paused sim (send pause first)")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._drain_inputs()
        for _ in range(n):
            self._backend.step(self._backend.dt)
        self._push_frame(force=True)

    def reset(self) -> None:
        """Reset model + renderer, then re-anchor pacing."""
        self._backend.reset()
        self._renderer.reset()
        self._rebase()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            self.tick()
            self._sleep(self._backend.dt)

    def tick(self) -> None:
        """One loop iteration: drain, pace, step, render-if-due."""
        self._drain_inputs()
        if self._paused or self._stop_evt.is_set():
            return
        target = self._t0_sim + (self._clock() - self._t0_wall) * self._speed
        due = int(round((target - self._backend.time) / self._backend.dt))
        for _ in range(min(max(due, 0), MAX_CATCHUP_STEPS)):
            self._backend.step(self._backend.dt)
        self._push_frame()

    def _rebase(self) -> None:
        self._t0_wall, self._t0_sim = self._clock(), self._backend.time

    def _drain_inputs(self) -> None:
        for msg in self._inputs.drain():
            self._dispatch(msg)

    def _dispatch(self, msg: InputMessage) -> None:
        if isinstance(msg, (Actuator, ActuatorSet, Neutral, Orbit)):
            self._backend.apply_input(msg)
        elif isinstance(msg, (Pause, Resume, Reset)):
            {Pause: self.pause, Resume: self.resume, Reset: self.reset}[type(msg)]()
        elif isinstance(msg, Speed):
            self.set_speed(msg.value)
        elif isinstance(msg, Step) and self._paused:
            self.step(msg.n)
        # Hello: handshake is the app layer's job (PLAN.md §3).

    def _push_frame(self, force: bool = False) -> None:
        if not force and not self._renderer.render_as_needed():
            return
        self._frames.put(self._renderer.encode_jpeg(self._renderer.render_frame()))
