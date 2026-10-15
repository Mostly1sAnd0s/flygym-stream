# flygym-stream

Server-side NeuroMechFly/FlyGym simulation with headless GPU rendering,
streamed as MJPEG over WebSocket to a thin browser front end. All simulation
and rendering happens on the GPU server; the client needs only a web browser.
See [PLAN.md](PLAN.md) for the full design.

## Server environment (validated Phase 0, 2026-10-14)

- Ubuntu 26.04.1 LTS (x86-64), NVIDIA GeForce RTX 5090, driver 615.71.09
  (CUDA UMD 13.4).
- GL backend: **NVIDIA EGL** (`libEGL_nvidia` loaded by `mujoco.Renderer`
  under `MUJOCO_GL=egl`; mesa EGL also present as fallback).
- Python **3.12.14** via `uv` virtualenv (`.venv`); `uv sync --extra dev`
  to set up; flygym installed editable from the local `../flygym` checkout.
- System packages installed via apt: `ffmpeg`, `libegl1-mesa-dev`.
- flygym assets: simplified meshes are bundled in the package (no network).
  Full-size meshes fetch once from S3 (40 files, 14 MB) and cache under
  `~/.cache/flygym_assets/neuromechfly_fullsize_meshes_20260623a/`
  (override dir: `FLYGYM_ASSET_CACHE_DIR` / `XDG_CACHE_HOME`).
- Sim-loop headroom (Phase 4): 30,000 steps + 90 double-pass 720p renders
  in ~2.2 s wall -> ~1.4x real-time, comfortable for 30 fps streaming.
  `MUJOCO_GL`/`PYOPENGL_PLATFORM` must be exported *before* the process
  starts (mujoco reads them at import time).

### Phase 0 results (`scripts/egl_smoke.py`)

- Headless `mujoco.Renderer` over EGL renders non-trivial frames: mini model
  320x240 and the full flygym fly scene both pass shape/content asserts.
- Default scene (Fly `NeuroMechFly` + biological joints + leg actuators +
  `FlatGroundWorld`, 720p tracking camera): 30,000 sim steps (3 s sim,
  dt=1e-4) in **1.5 s wall** with rendering — ~2× real-time headroom at
  30 fps output. Full-size meshes: 1.9 s wall (~1.6× real-time).
- MuJoCo sim config: Euler integrator, dt=0.1 ms, nq=133 for the full
  biological skeleton.
- flygym API notes: `Fly` is deprecated — use `NeuroMechFly`. Camera names
  are namespaced by the fly name (`nmf/trackingcam`). The streamed view uses
  a persistent `mujoco.MjvCamera` (orbit state) passed to
  `update_scene` — MuJoCo's free camera is not exposed for mutation.

## Setup

```bash
cd flygym-stream
uv sync --extra dev        # creates .venv (Python 3.12), installs deps
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # REQUIRED at runtime
```

## Run

```bash
# headless rendering smoke test (Phase 0; --flygym adds the full fly scene)
uv run python scripts/egl_smoke.py --flygym

# warm the one-time S3 mesh cache (offline environments; idempotent)
uv run python -m flygym_stream.cli fetch-assets

# headless MP4 preview of the fly (Phase 4 DoD; ~3 s wall for 3 s sim @720p)
uv run python -m flygym_stream.cli render-preview --duration 3 --out /tmp/preview.mp4

# streaming service (Phase 5): scene + sim thread + uvicorn
uv run python -m flygym_stream.cli serve --bind 127.0.0.1 --port 8765
# options: --width/--height/--fps (default 1280x720@30), --speed (playback),
#          --backend cpu|gpu|auto, --overlay none|depth, --transport
#          mjpeg|webrtc (Phase 6 stretch), --insecure (0.0.0.0 only)
```

Then open <http://127.0.0.1:8765> in a browser
(`http://127.0.0.1:8765/?webrtc` for the WebRTC transport). For LAN
access on a headless server: `serve --bind 0.0.0.0 --insecure`, then
open `http://<server-LAN-IP>:8765` (see "Accessing across the LAN").

### Phase 6 stretch status

- **Depth overlay**: `serve --overlay depth` blends a blue-cyan depth tint
  over the streamed frames (second GL pass, composited server-side).
- **WebRTC transport**: `serve --transport webrtc` (needs the `webrtc`
  extra: `uv sync --extra webrtc`) + open the URL with `?webrtc`.
  Video goes over WebRTC (H.264/VP8, server is the offerer); the JSON
  control channel on the WebSocket is byte-identical to MJPEG mode.
- **MJWarp GPU backend** (`--backend gpu`/`auto`, needs the `gpu` extra:
  `uv sync --extra gpu`): implemented and parity-tested against CPU MuJoCo
  (matching to float32 precision on NoSlip-free models). **Caveat:** the
  default fly scene is compiled with MuJoCo's NoSlip solver, which no
  released `mujoco-warp` implements — for that scene the GPU backend
  raises an actionable error and `auto` falls back to CPU (which here is
  the faster path anyway: ~1.5-2x real-time headroom). NoSlip-free models
  step on the GPU and render through the same frame pipeline.

### Web UI

- **Video pane**: drag to orbit (azimuth/elevation), scroll to zoom. Frames
  are MJPEG over WebSocket at the configured fps.
- **Controls**: one slider per actuated DOF (42 in the default scene),
  `neutral` / `pause` / `resume` / `step` / `reset` buttons, and a playback
  speed slider (0.1-10x).
- **HUD**: sim time, measured fps, backend, model, paused state.
- **Protocol** (v1, mirrored from `src/flygym_stream/protocol.py`): the
  client sends `hello` first; the server replies with `status` (~1 Hz) plus
  binary JPEG frames; control messages (`orbit`, `actuator(s)`, `neutral`,
  `pause`, `resume`, `step`, `speed`, `reset`) are forwarded to the sim
  thread. One client at a time — a second connection is rejected with
  close code 1013 / `single-client`. Malformed input closes that
  connection with 1008; the server keeps running.

### Accessing across the LAN (headless server)

Two options:

1. **Direct LAN access** — bind all interfaces (the flag is required
   because this is an unauthenticated service):

   ```bash
   uv run python -m flygym_stream.cli serve --bind 0.0.0.0 --insecure --port 8765
   ```

   then open `http://<server-LAN-IP>:8765` (e.g. `http://172.16.109.63:8765`;
   find the IP with `ip -4 addr`). Note: the service has **no auth or TLS**
   and accepts one client at a time, so anyone on the LAN can connect and
   drive the simulation. For anything beyond a trusted LAN, use option 2.
   (No firewall is active by default on this host; if you enable `ufw`,
   allow the port: `sudo ufw allow 8765/tcp`.)

2. **SSH tunnel** (safer — stays bound to `127.0.0.1`):

   ```bash
   ssh -L 8765:127.0.0.1:8765 user@server
   # then open http://127.0.0.1:8765 in the browser
   ```

Tailscale works as an alternative to the SSH tunnel.

### systemd (optional)

`deploy/flygym-stream.service` runs the service under systemd (logs go to
the journal: `journalctl -u flygym-stream -f`). It binds `0.0.0.0` with
`--insecure` for LAN access — edit `ExecStart` back to
`--bind 127.0.0.1` (drop `--insecure`) if you'd rather tunnel:

```bash
sudo cp deploy/flygym-stream.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now flygym-stream
```

`deploy/logrotate.flygym-stream` is a logrotate snippet only if you
redirect output to files instead of the journal.

## Tests

```bash
uv run pytest                          # default: no gpu / network marks
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run pytest -m gpu   # GL tests
```
