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
| `orbit` | `azimuth`, `elevation`, `distance` (deg/deg/mm) | camera orbit; target = fly position |
| `actuator` | `index: int`, `value: float` | set one actuator ctrl |
| `actuators` | `values: [float, ...]` | set all at once |
| `neutral` | — | reset actuators to neutral pose |
| `pause` / `resume` | — | freeze/unfreeze sim loop |
| `step` | `n: int` | while paused, advance n substeps + render |
| `speed` | `value: float` | playback speed multiplier |
| `reset` | — | reset model to initial keyframe |

Server → client:
- **binary frames**: JPEG at `output_fps` (default 30), quality 80.
- **text status** (on `hello`, then ~1 Hz): `{"type":"status","sim_time":
  float,"fps": float,"paused": bool,"n_actuators": int,"backend":"cpu"|"gpu",
  "model":"fly"}`.

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
    │                                  FakeBackend, WS test client, tmp Config)
    ├── test_line_limits.py     ≤ 60   (meta-test: every project file < 500 lines)
    ├── test_protocol.py        ≤ 150
    ├── test_scene.py           ≤ 150
    ├── test_simulation.py      ≤ 200  (pacing, pause/step, input drain; FakeBackend)
    ├── test_rendering.py       ≤ 100  (JPEG shape/size with FakeRenderer; GPU test
    │                                  marked)
    ├── test_stream.py          ≤ 120  (drop-oldest, disconnect behavior)
    └── test_app.py             ≤ 200  (end-to-end: tiny model + FakeRenderer,
                                        connect WS, get frame, send control)
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

- MJWarp GPU backend enabled by default when CUDA present (`backend=auto`),
  with `@pytest.mark.gpu` test.
- Depth/segmentation overlay toggle (flygym `Renderer` already supports
  `render_depth=True`).
- WebRTC transport replacing MJPEG (protocol JSON channel unchanged).
- systemd unit + logrotate snippet in README.

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
