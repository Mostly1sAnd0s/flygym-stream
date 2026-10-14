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
  are namespaced by the fly name (`nmf/trackingcam`). The orbit camera is
  the MuJoCo free camera (`mj_data.cam`), no named camera needed.

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

# serve (Phase 5)
uv run python -m flygym_stream.cli serve --bind 127.0.0.1 --port 8765
```

The service binds `127.0.0.1` only. From a client machine tunnel first:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
# then open http://127.0.0.1:8765 in the browser
```

Tailscale works as an alternative to the SSH tunnel.

## Tests

```bash
pytest                          # default: no gpu / network marks
pytest -m gpu                   # GL-context / GPU tests
```
