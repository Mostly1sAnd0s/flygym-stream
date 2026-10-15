# PLAN — FlyGym Remote Streaming Service (Option B)

Server-side NeuroMechFly/FlyGym simulation with headless GPU rendering,
streamed as MJPEG over WebSocket to a browser front end on the client's
(underpowered) machine. All simulation and rendering happens on the server;
the client needs only a web browser.

## 1. Context

- **flygym** (NeLy-EPFL/flygym, v2.1.0, Apache-2.0) is a Python ≥ 3.12,
  < 3.15 library. Core deps: `mujoco>=3.9,<3.10`, numpy 2.x. Optional `warp`
  extra: `warp-lang>=1.14,<1.15`, `mujoco_warp>=3.9,<3.10` (GPU simulation,
  NVIDIA/CUDA only).
- The flygym repo ships a `Dockerfile` that runs it **headless**:
  `ENV MUJOCO_GL=egl`, `ENV PYOPENGL_PLATFORM=egl`, installs
  `libegl1-mesa-dev` and `ffmpeg`. We follow the same environment.
- The NeuromechFly browser viewer on the public site is a **client-side**
  WebAssembly app — it does NOT offload work. We are building a new, thin
  server-side streaming service instead. We do **not** modify flygym itself.
- flygym model meshes are **lazy-loaded from an S3 bucket** on first use
  (`flygym.utils.assets_lazy_loading`). First run needs network; afterwards
  the local cache is enough.

### Verified flygym APIs we build against

- Scene composition (pattern from `scripts/launch_interactive_viewer.py`):
  `Fly()`, `Skeleton(joint_preset, axis_order)`, `fly.add_joints(...)`,
  `fly.add_actuators(...)`, `fly.colorize()`, `FlatGroundWorld()`,
  `world.add_fly(...)`, then `mj_model, mj_data = world.compile()`.
- Headless rendering: `flygym.rendering.Renderer(mj_model, cameras,
  camera_res=(h, w), playback_speed=..., output_fps=..., buffer_frames=...,
  scene_option=..., render_rgb=..., **kwargs)` — `**kwargs` are passed to
  `mujoco.Renderer`, so EGL/CUDA device selection comes either from the
  `MUJOCO_GL` env vars or `device_type=mujoco.mjtDevice_EGL` kwarg.
  `renderer.render_as_needed(mj_data) -> bool` paces rendering by **sim
  time**; `renderer.reset()`, `renderer.close()`.
- GPU sim (optional path): `flygym.warp` (`mjw.put_model`, `mjw.Data`,
  `mjw.step`); `flygym.warp.rendering` subclasses `Renderer` for MJWarp data.
- Keyframes: `mj.mj_resetDataKeyframe(mj_model, mj_data,
  mj.mj_name2id(mj_model, mj.mjtKey, "neutral"))`.
- Research-sandbox surface (verified 2026-10-15, for Phase 7):
  `flygym.Simulation` (CPU single-world wrapper: `step`/`reset`/
  `set_actuator_inputs`/`set_tendon_actuator_inputs`/`warmup` + getters for
  joint angles/velocities, body pos/rot, site positions, actuator forces,
  body-segment contact forces, ground-contact info, raw vision,
  ommatidia readouts); worlds `FlatGroundWorld(half_size)`,
  `BlocksTerrainWorld`, `GappedTerrainWorld`, `MixedTerrainWorld`,
  `TetheredWorld` (root pinned as a mocap body — appendages stay free),
  `MusculoskeletalWorld`; `NeuroMechFly` options `add_vision()`,
  `add_leg_adhesion(gain)`, `add_tracking_camera(mode, pos_offset, fovy)`,
  `add_actuators(..., actuator_type, forcelimited, forcerange)`,
  `SCALE=1000`; `MusculoskeletalFly(muscle_names)` +
  `build_musculoskeletal_simulation(add_vision=...)`; `ContactParams`
  (10 friction/solver knobs); `ActuatorType` = {MOTOR, POSITION, VELOCITY,
  INTVELOCITY, DAMPER, CYLINDER, MUSCLE, ADHESION, TENDON};
  `ActuatedDOFPreset` = {ALL (133 DOF), LEGS_ONLY, LEGS_ACTIVE_ONLY (42)}.
  **Units: model length is mm** — `opt.gravity = -9810 mm/s²` (1 g),
  `SCALE = 1000` (biological mm → model units; full-size NMf ≈ 3 m).

## 2. Working agreements (MUST follow)

1. **TDD.** For every behavior: write the test first, run it and confirm it
   FAILS for the expected reason, then write the minimum code to make it
   pass, then refactor. Never add production code without a failing test
   (except pure glue: `__init__.py`, packaging, static assets — which are
   covered by integration tests). Keep the suite green between commits.
2. **DRY.**
   - One stepping interface for CPU (MuJoCo) and GPU (MJWarp) backends —
     the sim loop is written once against that interface (Strategy pattern).
   - One frame pipeline (render → encode → queue → broadcast); one producer.
   - `protocol.py` dataclasses are the single source of truth for the wire
     format; the JS client mirrors it from one place in `app.js`.
   - One `Config` dataclass — no magic numbers scattered across modules.
   - Reuse `flygym.rendering.Renderer` for scene setup/render calls; do not
     reimplement MuJoCo rendering internals.
   - One scene builder function used by the CLI, the server, and tests.
   - If a block of >10 lines appears twice with >90% similarity, extract it.
3. **File size limit: every file < 500 lines.** Enforced by a permanent
   meta-test (`tests/test_line_limits.py`, written in Phase 1) that fails
   CI if any project file (`.py`, `.js`, `.html`, `.css`, `.toml`) reaches
   500 lines. Line budgets per file are listed in §4.
4. Python 3.12, `uv` for environment management (matches flygym's tooling),
   `pytest` (flygym already uses it). Type hints + docstrings on all public
   functions.

## 3. Architecture

```
 client PC (browser)                    GPU SERVER
 +--------------------------------+     +--------------------------------------------------+
 | index.html / app.js            |     | flygym_stream/                                   |
 |  <img> MJPEG  < 30 fps JPEG    |<====|  app.py (FastAPI, asyncio)                       |
 |  orbit cam, sliders, buttons   | WSS |   /            serves web/                      |
 |  JSON control messages ----->  |====>|   /ws          JSON in (control),               |
 +--------------------------------+     |                  binary out (JPEG)               |
                                        |   /healthz       liveness                        |
                                        | stream.py  FrameQueue (size 1, drop-oldest)     |
                                        | rendering.py OffscreenRenderer                  |
                                        |    (flygym Renderer, EGL)                       |
                                        | simulation.py SimThread                         |
                                        |    step (CPU MuJoCo | MJWarp) + pacing + inputs |
                                        | scene.py   build_default_scene() / compile      |
                                        +--------------------------------------------------+
```

### Threading model

- **Sim thread** (plain `threading.Thread`): owns `mj_data`. Loops:
  drain input queue → apply inputs → step N sim substeps → if
  `renderer.render_as_needed(mj_data)` is True, encode JPEG → push to
  `FrameQueue`. Real-time pacing: sleep so the loop tracks wall clock
  (sim runs at `playback_speed`; default 1.0).
- **Asyncio server** (uvicorn/FastAPI): single client by design. WS
  handler pops frames from the queue (with timeout) and sends binary
  JPEGs; control JSON is parsed in `protocol.py` and pushed into the
  input queue. One active client; a second connection is rejected with a
  clear JSON error (close code 1013, message `single-client`).
- **FrameQueue**: `queue.Queue(maxsize=1)` — always shows the newest
  frame, no backpressure, no latency buildup when the client is slow.
- **Input queue**: `queue.Queue()` of validated `InputMessage` objects.

### Wire protocol (v1)

Client → server (WebSocket text, JSON):

| type | fields | meaning |
|---|---|---|
| `hello` | `proto: 1` | first message; triggers initial `status` |
| `orbit` | `azimuth`, `elevation`, `distance` (deg/deg/mm) | camera orbit; target = fly position. Default distance is scale-aware: `2.8 × target_extent_m` (the full-size NMf is ~3 m across, so a fixed mm default would park the camera inside it); `status` carries the live `orbit` so the client adopts the true view |
| `actuator` | `index: int`, `value: float` | set one actuator ctrl |
| `actuators` | `values: [float, ...]` | set all at once |
| `neutral` | — | reset actuators to neutral pose |
| `pause` / `resume` | — | freeze/unfreeze sim loop |
| `step` | `n: int` | while paused, advance n substeps + render |
| `speed` | `value: float` | playback speed multiplier |
| `reset` | — | reset model to initial keyframe |
| `camera` | `mode`: `orbit`\|`first`\|`third` and/or `view`: `front`\|`back`\|`left`\|`right`\|`top`\|`bottom` (M7.7) | camera mode and/or fly-relative viewcube snap; at least one field required; manual `orbit` messages imply orbit mode; `status` carries the mode in `cam_mode` |

Server → client:
- **binary frames**: JPEG at `output_fps` (default 30), quality 80.
- **text status** (on `hello`, then ~1 Hz): `{"type":"status","sim_time":
  float,"fps": float,"paused": bool,"n_actuators": int,"backend":"cpu"|"gpu",
  "model":"fly","orbit":{"azimuth","elevation","distance"}|null}`.

Client video strategy (simplest that works): on each binary message,
`img.src = URL.createObjectURL(blob)` (revoke previous). 30 fps of
720p MJPEG ≈ 0.5–1 MB/s — fine over LAN/Tailscale. (WebRTC is an
out-of-scope Phase 6 upgrade path; the protocol's JSON control channel
stays identical.)

### Configuration (`Config` dataclass, `config.py`)

`width=1280, height=720, output_fps=30, jpeg_quality=80, playback_speed=1.0,
bind="127.0.0.1", port=8765, backend="cpu"|"gpu"|"auto",
scene="fly_flat_ground"`. Overridable via CLI flags; env vars for
deployment (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` are REQUIRED at
runtime and asserted at startup with a clear error if unset/ignored).

## 4. Repository layout (with line budgets — all < 500)

```
flygym-stream/
├── PLAN.md                     (this file)
├── pyproject.toml              ≤ 80
├── README.md                   ≤ 150  (setup, run, tunneling, systemd)
├── src/flygym_stream/
│   ├── __init__.py             ≤ 20   (version)
│   ├── config.py               ≤ 100  (Config dataclass + defaults)
│   ├── protocol.py             ≤ 150  (message dataclasses, encode/decode, errors)
│   ├── scene.py                ≤ 150  (build_default_scene(), Scene wrapper:
│   │                                  compile/reset/apply_actuators/orbit camera)
│   ├── simulation.py           ≤ 200  (Backend ABC: MuJoCoBackend, MJWarpBackend;
│   │                                  SimThread: pacing, pause, input drain)
│   ├── rendering.py            ≤ 120  (OffscreenRenderer: wraps flygym Renderer,
│   │                                  JPEG encode via Pillow)
│   ├── stream.py               ≤ 120  (FrameQueue, InputQueue, producer glue)
│   ├── app.py                  ≤ 250  (FastAPI app, /, /ws, /healthz, WS handler)
│   └── cli.py                  ≤ 120  (serve, render-preview, fetch-assets)
├── web/
│   ├── index.html              ≤ 200  (video pane + control panel layout)
│   ├── style.css               ≤ 150
│   └── app.js                  ≤ 350  (WS client, orbit drag/zoom, sliders,
│                                       buttons; one MESSAGE map for all senders)
└── tests/
    ├── conftest.py             ≤ 250  (fixtures: tiny MuJoCo model, FakeRenderer,
    │                                  FakeBackend, FakeClock, tmp Config)
    ├── test_line_limits.py     ≤ 60   (meta-test: every project file < 500 lines)
    ├── test_protocol.py        ≤ 150
    ├── test_scene.py           ≤ 150
    ├── test_simulation.py      ≤ 220  (pacing, pause/step, input drain; FakeBackend;
    │                                  landed at 219 — pacing + step semantics)
    ├── test_rendering.py       ≤ 150  (JPEG encode pure + GPU-marked real renders,
    │                                  incl. cross-thread EGL regression test)
    ├── test_stream.py          ≤ 120  (drop-oldest, disconnect behavior)
    ├── test_cli.py             ≤ 150  (GL env guard, render-preview DoD, fetch-assets,
    │                                  serve subprocess smoke: real WS + real JPEG)
    ├── test_app.py             ≤ 200  (end-to-end with FakeBackend/FakeRenderer:
    │                                  connect WS, get frame, send control,
    │                                  single-client 1013, bad input 1008)
    └── test_app_gpu.py         ≤ 150  (GL-only: full real scene through the app;
                                        control -> pixels proof over the WS)
```

PyPI deps (add to `pyproject.toml`): `flygym>=2.1.0` (or local path
`flygym @ file://...` if you check out the repo on the server and want to
patch it), `fastapi`, `uvicorn[standard]`, `pillow`. Optional extra
`gpu`: `warp-lang`, `mujoco_warp` (mirror flygym's version pins).
Dev extra: `pytest`, `pytest-cov`.

## 5. Phases (each phase = green test suite + commit)

### Phase 0 — Environment validation (manual, on the server, BEFORE coding)

Purpose: de-risk headless GPU rendering. If this fails, everything else is
moot — solve it before writing any code.

1. `python3.12 -m venv .venv` / `uv venv`; install flygym + deps
   (`uv sync` if using a pyproject, or `pip install flygym`).
2. Export `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.
3. EGL smoke test (script it as `scripts/egl_smoke.py` so it's re-runnable):
   load a small model (tiny flygym model, or `mujoco`'s bundled
   `humanoid.xml`), create `mujoco.Renderer(model, 240, 320)`, render one
   frame, assert shape `(240, 320, 3)` and non-trivial content.
4. Verify which GL backend is actually used (NVIDIA EGL vs mesa) — check
   `eglinfo -B` / `nvidia-smi` during render. If NVIDIA EGL is unavailable,
   try `device_type=mujoco.mjtDevice_CUDA`; if neither, fall back to
   software (llvmpipe) and note the perf ceiling in README.
5. First run will trigger flygym's S3 lazy-asset fetch — confirm it succeeds
   and identify the cache dir for the offline note in README.
6. Record results (backend used, resolution/fps achieved) in `README.md`.

**DoD:** `scripts/egl_smoke.py` exits 0 with a rendered frame; notes in
README. (This phase is the one exception to "tests first" — it's
exploratory validation, not product code.)

### Phase 1 — Skeleton + constraints

- `pyproject.toml`, package skeleton, `config.py` + tests,
  `tests/test_line_limits.py` (meta-test enforcing §2.3 — written first,
  trivially green, guards everything after).
- `tests/conftest.py`: fixtures for a tiny CPU MuJoCo model, `FakeRenderer`
  (returns a synthetic frame, no GPU), `FakeBackend` (advances a counter as
  "sim time", no physics).

**DoD:** `uv sync` works; `pytest` green; line-limit test passes.

### Phase 2 — Protocol

`protocol.py`: dataclasses (`Orbit`, `ActuatorSet`, `Pause`, `Resume`,
`Step`, `Speed`, `Reset`, `Status`) + `decode_control(bytes|str) ->
InputMessage` + `encode_status(...) -> str` + `ProtocolError`.
Tests: round-trips, bad type/rejected payload → `ProtocolError`, unknown
type tolerated (log + ignore, don't crash the server), `proto` mismatch on
`hello` → close with version message.

**DoD:** `pytest tests/test_protocol.py` green; zero non-test code beyond
`protocol.py`.

### Phase 3 — Scene

`scene.py`: `build_default_scene()` (the `scripts/launch_interactive_viewer.py`
pattern: Fly + biological joints + leg actuators, `FlatGroundWorld`,
`colorize()`, neutral keyframe) → returns `(mj_model, mj_data, actuator_ids)`.
`Scene` wrapper: `reset()`, `set_actuators(values)`,
`set_orbit_camera(azimuth, elevation, distance)` (composes
`mj_data.cam` free-camera fields, target = fly body site), `step()` delegate.
Tests (CPU, real tiny/flygym model where assets allow): scene compiles,
neutral keyframe resets data, `set_actuators` writes `mj_data.ctrl`,
orbit camera changes `cam` fields, reset restores initial state. Mark any
test needing S3-fetched assets `@pytest.mark.network`.

**DoD:** suite green; `build_default_scene()` is the ONLY place the model
is built (DRY check).

### Phase 4 — Simulation loop + rendering (no network yet)

- `rendering.py`: `OffscreenRenderer` wraps `flygym.rendering.Renderer`
  (`buffer_frames=False`, `render_rgb=True`), `render_frame() -> np.ndarray`,
  `encode_jpeg(frame) -> bytes` (Pillow, configurable quality), device
  selection per `Config.backend` (`auto` → EGL, fallback CUDA, fallback
  software with warning). GPU/real-model tests marked `@pytest.mark.gpu`.
- `simulation.py`: `Backend` ABC (`step(dt)`, `apply_input(msg)`,
  `reset()`, `time`) with `MuJoCoBackend` and `MJWarpBackend` (MJWarp
  imports lazily; `backend="gpu"` errors clearly if CUDA/warp missing —
  that error path IS unit-tested with a fake import failure or by asserting
  the message when the extra is absent).
  `SimThread`: input queue drain → step → render-if-due → FrameQueue push →
  wall-clock pacing; `pause()`, `step(n)`, `set_speed(f)`; clean `stop()`.
  Tests use `FakeBackend`: exact step counts for given dt and pacing,
  pause blocks stepping, `step(n)` advances exactly n while paused,
  inputs from queue are applied in order, stop is clean (no leaked thread).

**DoD:** suite green; `uv run python -m flygym_stream.cli render-preview
--duration 3 --out /tmp/preview.mp4` produces a playable video of the fly
(headless, GPU, real model) — this is the end-to-end proof that
simulate→render works on the server.

### Phase 5 — Server + web front end

- `stream.py`: `FrameQueue` (maxsize 1, drop-oldest, `get(timeout)`),
  `InputQueue`; producer/consumer glue. Tests: drop-oldest under overflow,
  timeout returns None, no loss of the latest frame.
- `app.py`: FastAPI app factory `create_app(config, producer) -> FastAPI`
  (DI for tests): `GET /` serves `web/index.html`, `GET /healthz`,
  `WS /ws`: hello handshake → status; binary frames out; control JSON in
  (via `protocol.decode_control`, bad input → close 1008 without killing
  the server); second concurrent client rejected (1013, `single-client`);
  disconnect leaves the sim thread running. Tests with
  `httpx`/`websockets` test client + `FakeBackend`/`FakeRenderer`
  end-to-end: connect → receive ≥1 JPEG frame → send `orbit` → status
  reflects state → disconnect → server still healthy.
- `web/`: `index.html` (video pane + control panel), `style.css`, `app.js`
  (WS with auto-reconnect, MJPEG `<img>` swapping, pointer drag = orbit
  azimuth/elevation + wheel = distance, one slider per actuated DOF
  (count discovered from `status.n_actuators`), neutral/pause/resume/step/
  reset buttons, speed slider, HUD line with sim_time + fps). Keep ALL
  outgoing JSON construction in a single `const MSG` map (DRY across JS).
- `cli.py`: `serve` (build scene, start SimThread, run uvicorn),
  `render-preview` (Phase 4 artifact), `fetch-assets` (force flygym asset
  prefetch). Asserts `MUJOCO_GL`/`PYOPENGL_PLATFORM` at startup.

**DoD:** on the server: `uv run python -m flygym_stream.cli serve
--bind 127.0.0.1 --port 8765`; from the client PC via
`ssh -L 8765:127.0.0.1:8765 user@server` (or Tailscale): open
`http://127.0.0.1:8765` → live 720p@30 fly scene renders; orbit drag/zoom
is smooth; sliders move the legs; pause/step/speed/reset work; closing the
tab and reconnecting works; full `pytest` green.

### Phase 6 (optional stretch — only after Phase 5 DoD)

Delivered 2026-10-14 (see `git log`):

- MJWarp GPU backend: **implemented + parity-tested** (NoSlip-free models
  match CPU to float32 precision; `make_backend(auto)` falls back to CPU
  with a logged note). Blocker, documented: no released `mujoco-warp`
  implements MuJoCo's NoSlip solver, which the default fly scene uses
  (`noslip_iterations=5`); the backend refuses such models with an
  actionable error (CPU stays the fast path for the default scene).
- Depth overlay: `Config.overlay` + `serve --overlay depth` (pure
  `apply_depth_overlay` compositor + GPU test).
- WebRTC transport: `Config.transport` + `serve --transport webrtc`
  + `?webrtc` browser mode; `webrtc.py` (FrameTrack + WebRtcSession,
  server-as-offerer); JSON control channel unchanged (app-layer
  `webrtc-start`/`webrtc-offer`/`webrtc-answer` signaling); headless
  e2e test (real scene -> RTP -> decoded frames). `web` extra: aiortc.
- systemd unit + logrotate snippet in `deploy/` (documented in README).
- Not done (deliberate): replacing MJPEG with WebRTC as the *default*
  transport (MJPEG is simpler/lower-latency for a single LAN client;
  WebRTC is an opt-in alternative, as the plan's "replacing" phrasing
  would break the DoD baseline).

### Phase 7 — Research sandbox: full NeuroMechFly access (planned)

Goal: turn the service into a research sandbox — every NeuroMechFly/
flygym capability reachable from the browser over the JSON protocol:
all 133 DOFs, all worlds, joint-actuator and muscle/tendon models,
vision, leg adhesion, contact telemetry, physics tuning, built-in
locomotion, and full state read/write. v1 core invariants hold:
single client, one stepping interface, `protocol.py` the wire source of
truth, every file < 500 lines, strict TDD.

Model facts (default scene, verified): 133 q / 132 v DOFs; 42 actuators
(LEGS_ACTIVE_ONLY) — 6 wing DOFs, head/eye DOFs and the free root joint
are unactuated; 6 built-in leg contact sensors
(`ground_contact_{l,r}{f,m,h}_leg` in `sensordata`); 49 joint sites;
1 keyframe (`neutral`); 1 camera (tracking); NoSlip solver (Phase 6).

**7.1 Scene v2 — `scenes.py` (new)**: frozen `SceneConfig` dataclass:
`mesh_type` (FULLSIZE|SIMPLIFIED_MAX2000FACES), `joint_preset`, `axis_order`,
`actuated_preset` (42 vs 133 actuators), `actuator_type`,
`force_range`, `contact_bodies`, `world` (flat|blocks|gapped|mixed|
tethered) + `world_kwargs`, `vision`, `adhesion`, `tracking_camera`,
`spawn`/`spawn_rotation`, `musculoskeletal`. `build_scene(config)` is
the single builder; `build_default_scene() = build_scene(SceneConfig())`
so the DRY guard test (model built only in the scene layer) survives.
CLI flags: `--dof all|legs|legs-active`, `--mesh`, `--world`,
`--actuator`, `--spawn x,y,z`, `--tethered`, `--vision`, `--adhesion`,
`--musculoskeletal`. Decision: **no runtime scene rebuild in v1** —
variants are process-startup choices (warm-cache rebuild ≈ 30 s).

**7.2 Protocol v2** (protocol version stays 1; additive; unknown types
already tolerated both ways):
- Control (client → server):
  - `set_state {qpos?[], qvel?[], ctrl?[]}` — partial state write +
    `mj_forward` (teleport / throw the fly).
  - `spawn {pos[3], quat[4]}` — set the free root (qpos[0:7]); rejected
    (1008) in tethered worlds.
  - `force {body: name|int, force[3], torque?[3], duration? s}` —
    `qfrc_applied` injection: perturbation pulses, tether/wind forces.
  - `param {name, value|values}` — whitelist only: `gravity` (3),
    `timestep`, `ContactParams` fields (10); unknown name → 1008.
  - `keyframe {name}`, `adhesion {leg, state 0|1}` (adhesion builds).
  - `mode {value: neutral|walk|flap|manual}` + `loco {freq?, amplitude?,
    phase?}` — locomotion pattern + live parameter sweeps.
  - `log {on: 0|1}` — start/stop server-side JSONL logging.
- Telemetry (server → client):
  - `telemetry` @ `--telemetry-hz` (default 10, 0 = off):
    `{t, qpos[133], qvel[132], ctrl[u], root: {pos, quat},
    contact: {per-leg Fz from the 6 sensors}, paused, mode}` ≈ 10 KB/s
    JSON at 10 Hz — fine for LAN; fixed field set in v1.
  - `model-info` (sent after hello; re-sent on `info` request): units
    ("mm"), scene config, joints table (id, name, type, qadr, actuator
    {id, type, range} | null), sites, cameras, keyframes, sensors,
    muscle names (MSK builds) — enables arbitrary external clients.
  - `status` gains `mode` and `n_dof`.
- Telemetry reads **MjData directly** (single source of truth; the 6
  contact sensors are in `sensordata`). `flygym.Simulation` is used
  only for the vision stretch (7.6), where its `get_raw_vision` helper
  is needed.

**7.3 `locomotion.py` (new) — pattern engine**: pure functions of sim
time (deterministic, frame-rate independent, applied on the sim thread
before each substep while a pattern mode is active; `manual`/`neutral`
restore slider authority). `walk`: six-leg gait, tripod phasing
(lf/rm/rh vs rf/lm/lh), per-leg sinusoids on the proximal DOFs
(hip yaw/roll, femur/tibia flexion) with live-tunable {freq, amplitude,
phase}. `flap` (requires `--dof all` for wing actuators): antiphase
sinusoid on the wing-pitch DOFs, 1–10 Hz.

**7.4 `telemetry.py` (new)**: rate-limited gather from MjData on the sim
thread → latest-wins `TelemetryQueue` (same semantics as FrameQueue);
app send loop multiplexes binary frames + text telemetry/status.
`log`: JSONL of telemetry + control events to `logs/`; `GET /log`
returns the latest file (fetch-and-analyze workflow).

**7.5 UI (web/)**: DOF table (replaces the 42-slider list when
`n_actuators > 42`): virtualized rows (name + slider + value + live
qpos readout), coalesced `actuators` sends per animation frame. Mode
row: neutral | walk | flap | manual + freq/amplitude sliders.
Perturbation panel: spawn x/y/z + quat, gravity-z slider, force-pulse
button, timestep. Telemetry strip: root z, |v_root|, six per-leg
contact LEDs, sim time, fps. Vision pane (when `--vision`): second
`<img>` fed by fly-eye frames. The default-scene UI is unchanged.

**7.6 Stretch (each flag-gated, each its own milestone + DoD)**:
`--world blocks|gapped|mixed` (locomotion over terrain), `--tethered`
(flight without locomotion: flap with pinned body), `--musculoskeletal`
(muscle/tendon model: `tendon {id, value}` message; muscle list in
model-info), `--vision` (fly-eye feed via `flygym.Simulation`).

**7.7 TDD (tests first, per milestone)**:
- scenes: variant build matrix (variant × asserts: nact, root free?,
  geoms, keyframes, units); default-scene regression stays green
  (42 actuators, neutral keyframe, `nmf` site, scale-aware orbit).
- protocol: round-trip + malformed per new message (whitelist rejects,
  units, finiteness, partial `set_state`, `spawn` on tethered → 1008).
- locomotion: pure math, no GL (zero-crossings = freq, L/R symmetry,
  phase offsets, bounds, determinism).
- telemetry: values == MjData on the tiny model (fake) + GPU.
- app e2e: **`mode walk` → root translates > 50 mm within 2 sim s
  (GPU, the headline test)**; `force` impulse → root velocity spike;
  `set_state`/`spawn` → root at requested pose; `param gravity 0` →
  root drifts; `log` on/off → JSONL written and retrievable via /log.
- DoD per milestone = full suite green + commit (house rule).

**7.8 Budget supplement (§4)** (hard 500-line ceiling unchanged):
`scenes.py` (new) ≤ 300; `locomotion.py` (new) ≤ 250;
`telemetry.py` (new) ≤ 200; `protocol.py` 210 → ≤ 380 (split decode
table if tighter); `app.py` 211 → ≤ 280; `simulation.py` 358 → ≤ 450;
`cli.py` 149 → ≤ 230; `config.py` 104 → ≤ 160; `web/app.js` 257 → ≤ 480
(split `app_sandbox.js` ≤ 200 if tighter); tests: `test_scenes.py` ≤ 220,
`test_locomotion.py` ≤ 150, `test_telemetry.py` ≤ 120 (all new), existing
protocol/app/gpu suites +≤ 100 each.

**7.9 Milestones**:
- ~~M7.1 `scenes.py` + `SceneConfig` + CLI flags.~~ ✅ `a185829`
- ~~M7.2 Protocol v2 control (set_state/spawn/force/param/keyframe/
  adhesion) + backend dispatch.~~ ✅ `af27734`
  - Impl notes: `force` injects via `xfrc_applied` (spatial, 6/body: f,τ)
    — `qfrc_applied` is nv-length joint-space, not per-body; spawn on a
    pinned root raises at apply time (logged; protocol is model-
    agnostic so it cannot 1008 at decode); `param` whitelist v1 =
    {gravity 3, timestep, margin, solref 2, solimp 5, friction 3} →
    MjOpt `o_*` fields (bindings prefix `o_`).
- ~~M7.3 Locomotion (neutral/walk/flap/mode + `loco` tuning).~~ ✅
  - Impl notes: pattern = offsets from a **baseline captured on mode entry**
    (not the neutral keyframe) — walking tracks whatever pose the user
    set; `neutral` mode restores the neutral keyframe ctrl; `manual` =
    no-op (slider authority). flap = **in-phase** wing pitch (Drosophila
    wings couple; the "antiphase" plan wording was a plan bug).
  - Verified gait facts (default scene, 2026-10-15):
    - The fly faces **−x** (head x < thorax x by ~20 mm).
    - The `neutral` keyframe stores an **upright** pose (root quat
      identity, z=2.1 m, feet 0.84 m above the ground, no contacts).
      Default rest state (2026-07-21, user bug fix): the scene NO LONGER
      overwrites the root with a 0.8 m spawn (`SceneConfig.spawn` default
      is None) — the fly starts dorsal-up at the keyframe, dips ~1 m in
      ~0.5 sim s, and **stands on its six feet** (root z 1.081 m, all 6
      tarsus5 geoms in ground contact, ncon≥5).
    - Standing pose (2026-07-21, CORRECTED conclusion): the earlier
      "no passive standing pose holds" conclusion was based on the WRONG
      pose — manually extended joint offsets whose feet all sit head-side
      (COM behind the support polygon). The model's OWN neutral-ctrl pose
      is a self-stable stance: COM x=−0.20 is INSIDE the support polygon
      (feet x −1.53…+1.38), z/x/y constant to the millimetre over 10 s,
      dorsal up (wing z − thorax z = +0.215), position oscillation
      0.01–0.2 mm p-p. Its one flaw: a ~1 Hz **yaw limit cycle** (w_y
      ±0.19 rad/s) — the leg actuators' reaction torque pumps the 1 g
      root (zero passive yaw stiffness) — which is the "slowly
      rotating" creep users saw. Fixed by the **rest anchor**: the
      Scene zeroes the root's angular velocity (`qvel[3:6]`) every
      substep while active (yaw drift ≤0.01° over 10 s; position held by
      the foot contacts; appendages stay fully dynamic). Anchor is on by
      default and in `neutral` mode; released by `spawn`/`set_state`
      (explicit state writes take over the physics) and by any
      non-neutral locomotion mode (walk/flap/manual = free physics).
      `mode neutral` / the `neutral` message do a full state restore
      (`Scene.neutral_restore`: qpos/qvel/ctrl + anchor on, sim clock
      untouched) — the UI's neutral button returns the fly to its
      standing reference pose. A force-based "hover" (xfrc_applied
      gravity compensation) was tried first and rejected: the 42
      position actuators (±30 force vs 0.01 N weight) fling the 1 g
      thorax regardless of the gravity term (MuJoCo 3.x note: applied
      spatial forces act in the WORLD frame, empirically verified).
    - The v1 open-loop tripod gait does **not** walk quasi-statically:
      from the standing rest state it translates ~0.5 m in 2 sim s
      (right-side-up; the headline test's >50 mm in 2 sim s passes with
      huge margin); gait QUALITY (even footfall, no stumbling) is the
      open research task — exactly what the sandbox (set_state +
      per-DOF ctrl + telemetry + force) is for.
- ~~M7.4 Telemetry + model-info + JSONL logging + `GET /log`.~~ ✅ `eab746b`
  - Impl notes: telemetry runs only while unpaused (1 Hz status covers
    paused); root = qpos[0:7] when free-root else the `nmf/c_thorax`
    body frame (tethered); MuJoCo 3.9 has no `actuator_acttype` field —
    model-info reports the (gaintype, biastype, dyntype) triple + target
    instead; binding enum names come from `mjtJoint(v).name` (HINGE=3
    there, not 1); FastAPI cannot take union return annotations on
    routes (`/log` returns FileResponse|JSONResponse un-annotated).
- ~~M7.5 UI sandbox (DOF table, mode row, perturbation panel, telemetry
  strip; default UI unchanged).~~ ✅ `4fa78cc`
  - Impl notes: sandbox UI in `web/app_sandbox.js` (fake-DOM headless test
    `scripts/ui_smoke.cjs`, 19 checks); ctrl shadow resyncs from telemetry
    (skips dirty/dragged indices) so pattern-mode gait is visible in the
    sliders; `actuators` sends are full-array + coalesced per rAF.
- M7.6 Stretch: terrain worlds → tethered → musculoskeletal → vision.
  - ✅ Terrain (`--world blocks|gapped|mixed`) + tethered: buildable since
    M7.1 (`a185829`, network-tested); tethered root pinned, spawn
    rejected at apply time (logged).
  - ✅ Musculoskeletal (`--musculoskeletal`): `e8dbe48`.
    FlyMimic self-contained MJCF via `MusculoskeletalWorld` (floor/
    lights/anchored thorax — no world compose, no spawn); 14 q/nv DOF,
    15 Hill-type muscle actuators (dyntype 4, ctrl = activation 0..1,
    only L-front leg muscle-driven per flygym); orbit-target site
    `nmf` added at the `Thorax` body; renderer pacing camera = model
    `scene` camera; telemetry root falls back to body `Thorax`; CLI
    skips the leg-locomotion engine. Also fixed telemetry starvation
    (send loop checks the telemetry queue every iteration).
  - ✅ Vision (`--vision`): `226f674`.
    Vision scenes build through `flygym.Simulation(world)` (it compiles
    the world itself and exposes `mj_model`/`mj_data`; the Scene steps
    the SAME MjData the eye renderer reads); model cameras are
    `nmf/trackingcam` + `nmf/l_eye_cam_camera` + `nmf/r_eye_cam_camera`
    (ncam==3); the render-pass pacing camera falls back to the left eye
    when the tracking camera is off; musculoskeletal+vision is rejected
    with ValueError (experimental in flygym). `SimThread` gains
    `vision`/`vision_queue` kwargs; `_maybe_vision()` runs after
    `_maybe_telemetry()` (sim-time paced at half the output fps, on the
    sim thread — the eye renderer's GL context binds to it). The app
    send loop drains the vision queue every iteration exactly like
    telemetry (V1-prefixed binary on the same WS channel; main frames
    stay bare JPEG — no protocol.py change). UI: `#vision` overlay
    (top-right, 180 px, pointer-events none) revealed on the first V1
    frame; `onBinary` routes 0x56 0x31 frames to the vision img, bare
    JPEGs to the main img. Verified: GPU test (V1 + real fisheye-eye
    JPEG, PIL decodes 450x1024 stacked eyes, non-black), network test
    (ncam==3, scene steps), WS V1 pass-through test, ui_smoke V1
    routing (4 checks), live `serve --vision` (status/model-info +
    V1 vision frames interleaved with main frames).
  - Wire note: vision framing is binary (no protocol.py change); main
    MJPEG frames = bare JPEG, vision frames = `V1` + JPEG.
- ~~M7.7 Camera control: viewcube + 1st/3rd-person views.~~ ✅ (this commit)
  - Wire: client `camera` message `{mode: orbit|first|third}` and/or
    `{view: front|back|left|right|top|bottom}` (viewcube face; at least
    one field required, else 1008); `status` gains `cam_mode`. Manual
    `orbit` messages always imply orbit mode.
  - Scene: `fly_forward()` = thorax→head world direction (fly faces +x
    at rest; verified 3-DOF network test); `set_orbit_view` snaps the
    orbit to the fly-relative side (az = heading ± 90 / 0, el = 90/−89
    for top/bottom — verified by real-fly pixel test); `effective_camera`
    is the rendered camera in all modes (body views recomputed from the
    live head/thorax pose every frame; 1st = above the head looking
    forward, 3rd = trailing behind+above looking ahead); the stored
    orbit is the source of truth, body views never mutate it, and
    switching back to orbit captures the current effective camera so
    mode changes never jump the view.
  - UI: CSS-3D viewcube (top-left of the video pane; the cube rotates by
    `rotateX(−el) rotateY(az − heading)` so the face the camera is on
    fronts; heading from the telemetry root quat, +x forward); orbit/1st/
    3rd button row (active state from `status.cam_mode`); dragging the
    video in a body view sends `camera {mode: orbit}` first; the orbit
    HUD now leads with the camera mode. ui_smoke +8 checks.
  - Verified: protocol round-trip/validation tests; tiny-scene camera
    geometry tests; GPU pixel test (`orbit vs first 0.787`, `first vs
    third 0.997`, `third vs front 0.999` of 320×180 px changed); live
    `serve` e2e (cam_mode round-trip, view snap, body-view effective
    orbit in status).

**7.10 Risks & decisions**:
- Runtime scene rebuild: out of scope v1 (process restart; decision).
- 133 sliders: virtualization + coalesced sends; DOM never holds > 40
  rows; slider input batches into one `actuators` message per frame.
- Telemetry: JSON v1, fixed fields, `--telemetry-hz 0` disables;
  binary packing only if bandwidth ever matters.
- Full state write = full physical authority: this is a trusted
  research tool; binding policy (single client, LAN/SSH-tunnel) and the
  README security note are unchanged.
- MJWarp/NoSlip (Phase 6) applies to every DOF preset — the GPU backend
  still refuses the default scene; CPU is the default path. No new work.
- Protocol version stays 1 (additive changes only); a version bump
  happens only if a v1 field is ever renamed.

## 6. Server environment (record actual values in README)

- OS: Linux x86-64; NVIDIA driver (for `backend=auto`/gpu path) — required
  only for the MJWarp sim path and NVIDIA EGL; mesa EGL covers CPU-sim +
  GPU/other-vendor rendering.
- `apt`/system: `libegl1` (or `libegl1-mesa-dev` per flygym's Dockerfile),
  `ffmpeg`.
- Python 3.12 via `uv`; env vars **always** `MUJOCO_GL=egl`,
  `PYOPENGL_PLATFORM=egl` (put them in the systemd unit / README run
  commands; `cli.serve` refuses to start without them, with a helpful
  message).
- Network: service binds `127.0.0.1` only. Clients connect via SSH tunnel
  or Tailscale. Never bind 0.0.0.0 without an explicit `--insecure` flag.

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Headless EGL not available on the GPU/driver | Phase 0 smoke test first; fallback chain EGL → `mjtDevice_CUDA` → llvmpipe; result recorded in README |
| MJWarp requires NVIDIA CUDA | `backend=cpu` is default and fully supported (MuJoCo is ~2× real-time, enough for one fly); gpu is opt-in with clean error |
| Latency over WAN | orbit inputs are independent of the frame loop (applied at next substep, not piggybacked on frames); Tailscale/SSH-tunnel for LAN-like performance; MJPEG at 30 fps is plenty given slow-mo biology (playback_speed default 1.0 of a sub-real-time sim) |
| Asset fetch from S3 fails on air-gapped server | `cli fetch-assets` run once on a connected box, then copy the cache dir; document it |
| Frame latency buildup with slow client | `FrameQueue(maxsize=1)` drop-oldest — server never blocks on the client |
| Scope creep (multiple clients, WebRTC) | Out of scope for v1; single-client by design, protocol kept transport-agnostic |

## 8. Global acceptance criteria

1. `pytest` (default marks: no `gpu`, no `network`) passes on the server.
2. `tests/test_line_limits.py` passes — no file ≥ 500 lines.
3. `render-preview` CLI produces a video of the fly, headless, on the GPU.
4. End-to-end (Phase 5 DoD) works through an SSH tunnel from a second
   machine: smooth 720p@30, orbit + actuator + playback controls all
   functional, reconnect clean, `/healthz` up.
5. README covers: prerequisites, setup (uv + env vars), run commands,
   tunneling, asset cache location, systemd snippet, backend selection.
6. Every behavior in §3 (protocol table, threading model, queue semantics,
   rejection behaviors) has at least one test.
7. Code reviews against §2: no duplicated scene construction, no
   duplicated protocol literals, one Config, flygym `Renderer` reused.

## 9. Out of scope (v1)

Multiple simultaneous clients, WebRTC/HEVC, the WASM game
(`wasm/game`), RL training integration, multi-fly scenes, authentication
(beyond bind-address + tunneling), flygym modifications.
