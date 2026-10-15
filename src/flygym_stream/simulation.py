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

import mujoco as mj

from flygym_stream.config import Config
from flygym_stream.protocol import (
    Adhesion, Actuator, ActuatorSet, Force, Hello, InputMessage, Keyframe,
    Loco, Mode, Neutral, Orbit, Param, Pause, Reset, Resume, SetState,
    Spawn, Speed, Step,
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

    @property
    def orbit(self) -> Orbit:
        """Current orbit state (the Scene is the source of truth)."""
        azimuth, elevation, distance_mm = self._scene.orbit
        return Orbit(azimuth, elevation, distance_mm)

    @property
    def data(self):
        """The CPU MjData this backend steps (locomotion/telemetry read)."""
        raise NotImplementedError(f"{type(self).__name__} has no CPU MjData")

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

    @property
    def data(self):
        return self._scene.mj_data

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

    def apply_state(self, msg: InputMessage) -> None:
        """Apply a state message through the Scene surface (M7.2).

        Raises:
            ValueError: If the scene cannot honor it (e.g. spawn on a
                pinned root, unknown keyframe/body, non-adhesion leg).
        """
        if isinstance(msg, SetState):
            self._scene.apply_state_arrays(msg.qpos, msg.qvel, msg.ctrl)
        elif isinstance(msg, Spawn):
            self._scene.set_root_pose(msg.pos, msg.quat)
        elif isinstance(msg, Force):
            self._scene.apply_force(msg.body, msg.force, msg.torque)
        elif isinstance(msg, Param):
            self._scene.apply_param(msg.name, msg.value)
        elif isinstance(msg, Keyframe):
            self._scene.apply_keyframe(msg.name)
        elif isinstance(msg, Adhesion):
            self._scene.set_adhesion(msg.leg, msg.state)
        else:
            raise TypeError(f"not a state message: {type(msg).__name__}")
        self.inputs.append(msg)


class MJWarpBackend(Backend):
    """GPU stepping via MJWarp (optional 'gpu' extra + NVIDIA CUDA).

    The GPU stack is imported lazily so the service runs CPU-only by
    default. After every step the GPU state is written back to the CPU
    MjData the (CPU-side) renderer reads. Models compiled with MuJoCo's
    NoSlip solver are refused: no released mujoco-warp implements that
    solver, and the default fly scene uses it — use ``--backend cpu``
    there (PLAN.md Phase 6).
    """

    def __init__(self, scene: Scene) -> None:
        super().__init__()
        if scene.mj_model.opt.noslip_iterations > 0:
            raise RuntimeError(
                "model uses MuJoCo's NoSlip solver, which mujoco-warp does "
                "not implement; the default fly scene needs --backend cpu "
                "(PLAN.md Phase 6)"
            )
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "backend 'gpu' needs the 'gpu' extra (mujoco-warp + "
                "warp-lang) with NVIDIA CUDA; install it "
                "(`uv sync --extra gpu`) or use --backend cpu"
            ) from exc
        self._mjw = mjw
        self._wp = wp
        self._scene = scene
        m = scene.mj_model
        # mjwarp pre-allocates contact/constraint buffers on the GPU; its
        # built-in heuristic underestimates for the fly model (94 resting
        # contacts), so size them from a short CPU warm-up instead.
        self._m = mjw.put_model(m)
        nconmax = _nconmax(m, scene.mj_data)
        self._d = mjw.put_data(m, scene.mj_data, nconmax=nconmax, njmax=2 * nconmax)
        self.dt = m.opt.timestep
        self.name = "gpu"
        self.n_actuators = scene.n_actuators

    @property
    def time(self) -> float:
        """Current simulation time in seconds (GPU -> host)."""
        return float(self._d.time.numpy()[0])

    @property
    def data(self):
        """The CPU mirror MjData (locomotion writes its ctrl here; the
        ctrl push to the GPU uses this same array)."""
        return self._scene.mj_data

    def step(self, dt: float) -> None:
        """Advance one model timestep on the GPU, then mirror to CPU."""
        if abs(dt - self.dt) > 1e-12:
            raise ValueError(f"dt {dt} != model timestep {self.dt}")
        self._mjw.step(self._m, self._d)
        self._sync_cpu()

    def _sync_cpu(self) -> None:
        """Copy GPU qpos/qvel/time into the CPU MjData and forward-kinematics it."""
        d = self._scene.mj_data
        d.qpos[:] = self._d.qpos.numpy()[0]
        d.qvel[:] = self._d.qvel.numpy()[0]
        d.time = self.time
        mj.mj_forward(self._scene.mj_model, d)

    def apply_input(self, msg: InputMessage) -> None:
        """Route a control message; actuator values go to the GPU ctrl."""
        if isinstance(msg, Orbit):
            self._scene.set_orbit_camera(azimuth=msg.azimuth,
                                         elevation=msg.elevation,
                                         distance_mm=msg.distance)
        elif isinstance(msg, Actuator):
            self._scene.set_actuator(msg.index, msg.value)
            self._push_ctrl()
        elif isinstance(msg, ActuatorSet):
            self._scene.set_actuators(msg.values)
            self._push_ctrl()
        elif isinstance(msg, Neutral):
            self._scene.set_actuators(self._scene.neutral_actuators())
            self._push_ctrl()
        else:
            raise TypeError(f"not a simulation input: {type(msg).__name__}")
        self.inputs.append(msg)

    def _push_ctrl(self) -> None:
        """Push the CPU ctrl (just validated/updated by Scene) to the GPU."""
        ctrl = self._scene.mj_data.ctrl[self._scene.actuator_ids]
        self._d.ctrl = self._wp.array([list(ctrl)], dtype=self._wp.float32,
                                      device="cuda")

    def reset(self) -> None:
        """Reset GPU + CPU to the same (keyframe or initial) state."""
        self._mjw.reset_data(self._m, self._d)
        self._scene.reset()  # CPU side: keyframe/neutral + forward
        d = self._scene.mj_data
        wp = self._wp
        self._d.qpos = wp.array([list(d.qpos)], dtype=wp.float32, device="cuda")
        self._d.qvel = wp.array([list(d.qvel)], dtype=wp.float32, device="cuda")
        self._d.time = wp.array([0.0], dtype=wp.float32, device="cuda")
        self._push_ctrl()


def _nconmax(m: mj.MjModel, d: mj.MjData) -> int:
    """GPU contact capacity: 2x the peak seen in a CPU warm-up (floor 64)."""
    scratch = mj.MjData(m)
    mj.mj_copyData(scratch, m, d)
    peak = 0
    for _ in range(200):
        mj.mj_step(m, scratch)
        peak = max(peak, scratch.ncon)
    return max(64, 2 * peak)


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
                 sleep=time.sleep, loco: object | None = None,
                 telemetry: object | None = None, vision: object | None = None,
                 vision_queue: object | None = None) -> None:
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
        self._loco = loco  # M7.3 pattern engine (None = no built-in gait)
        self._telemetry = telemetry  # M7.4 TelemetrySink (None = off)
        self._vision = vision  # M7.6 VisionFeed (None = off)
        self._vision_queue = vision_queue  # M7.6 (V1-prefixed frames)
        self._next_telemetry = 0.0
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

    @property
    def loco_mode(self) -> str:
        """Active locomotion mode (M7.3); 'neutral' with no engine."""
        return self._loco.mode if self._loco is not None else "neutral"

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
            self._loco_apply()
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
            self._loco_apply()
            self._backend.step(self._backend.dt)
        self._maybe_telemetry()
        self._maybe_vision()
        self._push_frame()

    def _loco_apply(self) -> None:
        """Apply the active locomotion pattern before a substep (M7.3)."""
        if self._loco is not None:
            self._loco.apply(self._backend.data)

    def _maybe_telemetry(self) -> None:
        """Rate-limit + publish a telemetry snapshot (M7.4, running only:
        while paused the state is static and the 1 Hz status covers it)."""
        sink = self._telemetry
        if sink is None or sink.hz <= 0.0:
            return
        t = self._backend.time
        if t < self._next_telemetry:
            return
        interval = 1.0 / sink.hz
        # Whole-interval catch-up: no burst after a slow tick.
        self._next_telemetry = (int(t // interval) + 1) * interval
        sink.queue.put(sink.gatherer.gather(
            paused=False, mode=self.loco_mode))

    def _maybe_vision(self) -> None:
        """Render the fly-eye frame if due (M7.6): sim-time paced, on the
        sim thread (the eye renderer's GL context binds to this thread);
        the V1-prefixed bytes go to the vision queue, which the WS send
        loop checks every iteration (same starvation rule as telemetry).
        """
        feed = self._vision
        if feed is None or self._vision_queue is None:
            return
        if feed.render_as_needed(self._backend.data):
            self._vision_queue.put(feed.render_frame())

    def _rebase(self) -> None:
        self._t0_wall, self._t0_sim = self._clock(), self._backend.time

    def _drain_inputs(self) -> None:
        for msg in self._inputs.drain():
            self._dispatch(msg)

    def _dispatch(self, msg: InputMessage) -> None:
        if isinstance(msg, (Actuator, ActuatorSet, Neutral, Orbit)):
            self._backend.apply_input(msg)
        elif isinstance(msg, (SetState, Spawn, Force, Param, Keyframe,
                              Adhesion)):
            # Research-sandbox state writes (M7.2): a rejection must not
            # kill the sim thread — log and keep stepping (error
            # telemetry lands in M7.4).
            try:
                self._backend.apply_state(msg)
            except (ValueError, NotImplementedError) as err:
                logger.warning("state message rejected: %s", err)
        elif isinstance(msg, (Pause, Resume, Reset)):
            {Pause: self.pause, Resume: self.resume, Reset: self.reset}[type(msg)]()
        elif isinstance(msg, Speed):
            self.set_speed(msg.value)
        elif isinstance(msg, (Mode, Loco)):
            # Locomotion mode/tuning (M7.3): rejections must not kill the
            # sim thread — log and keep stepping.
            if self._loco is None:
                logger.warning("locomotion message ignored (no pattern "
                               "engine on this sim)")
            else:
                try:
                    if isinstance(msg, Mode):
                        self._loco.set_mode(msg.value)
                    else:
                        self._loco.set_params(msg.freq, msg.amplitude,
                                              msg.phase)
                except ValueError as err:
                    logger.warning("locomotion message rejected: %s", err)
        elif isinstance(msg, Step) and self._paused:
            self.step(msg.n)
        # Hello: handshake is the app layer's job (PLAN.md §3).

    def _push_frame(self, force: bool = False) -> None:
        if not force and not self._renderer.render_as_needed():
            return
        self._frames.put(self._renderer.encode_jpeg(self._renderer.render_frame()))
