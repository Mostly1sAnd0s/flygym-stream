# Flywire integration

This adds an opt-in Eon connectome experiment to the existing `serve` command.
The server continues to use its existing port, bind checks, WebSocket/WebRTC
transport, browser interface, EGL renderer, single-client policy, idle/pause
handling, telemetry and deployment scripts. No new server process is started.
Without `--flywire-config` (or `FLYWIRE_CONFIG`), the original scene and gait
remain the default. Nothing is automatically deployed by this integration.

## Install on the GPU server

### First test on a Windows laptop

With `flygym-stream` and `Flywire-Brain-Model` beside each other, use the
existing Flywire environment and dataset. From PowerShell in `flygym-stream`:

```powershell
& ../Flywire-Brain-Model/.venv-body312/Scripts/python.exe -m pip install 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.34,<1' pillow
./scripts/serve-flywire-windows.ps1
```

After loading finishes, open `http://127.0.0.1:8765` in your browser; the page
connects automatically. Keep the terminal open; Ctrl+C stops it.
The script uses GPU neural compute, CPU body physics, Windows GLFW rendering,
640×480 video at 15 fps and 0.1× playback to make laptop testing manageable.
Pass `-NeuralBackend cpu` to diagnose CuPy problems, or `-Port 8766` if the port
is occupied. `-FlywireRoot C:/path/to/Flywire-Brain-Model` selects another checkout.
The listener is local to this computer. No systemd service, firewall rule or
remote deployment changes are needed. Local success does not test Linux EGL
or the remote machine's environment.

### Linux GPU host

Use the server's existing Python 3.12 environment and editable `../flygym`
checkout. The integration requires its FlyGym 2.1-compatible
`flygym_demo.complex_terrain` API. Do not replace that checkout with a different
FlyGym release just to install this feature.

```bash
uv sync --extra dev
uv pip install --python .venv/bin/python -r requirements-flywire-cuda12.txt
.venv/bin/python -c 'import cupy as cp; print(cp.arange(10).sum().get())'
```

The CUDA requirements file targets CUDA 12 CuPy. The server must have its
compatible runtime/toolkit as well as an NVIDIA driver. A newer driver alone
does not establish which toolkit is installed. Use a CuPy package matching the
server runtime if it differs; install only one CuPy distribution. CPU checking
only needs `requirements-flywire.txt` and `--neural-backend cpu`.

Dependencies are installed separately so the original dependency graph and
`uv.lock` remain unchanged. An exact `uv sync` may remove these optional
packages: reinstall the requirements afterwards. Launch with `.venv/bin/python`
as the existing systemd unit does, rather than a synchronizing `uv run`.

## Data and configuration

Create writable experiment configuration and output directories:

```bash
mkdir -p config/flywire logs/flywire
cp src/flygym_stream/flywire/{eon_config,hybrid_config}.json config/flywire/
```

Transfer the original `Data/neurons.csv` and `Data/connections_princeton.csv`
to the GPU server separately. They are not bundled or committed. Set
`data.neurons_csv` and `data.connections_csv` in `config/flywire/hybrid_config.json`
to their absolute server paths. Relative paths are resolved against that
configuration's directory, not the current working directory. Keep the
readout/input IDs and transmitter policy unless intentionally changing the
experiment. The loader audits the selected populations and Eon annotations.

## Run in the existing browser interface

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
.venv/bin/python -m flygym_stream.cli serve \
  --flywire-config config/flywire/eon_config.json \
  --neural-backend gpu --backend cpu \
  --bind 127.0.0.1 --port 8765
```

Keep your existing bind/port/transport flags and access method. If the normal
deployment uses `--bind 0.0.0.0 --insecure`, retain those options. The same
browser URL receives the Flywire body frames after the connectome loads.
Pause/resume, single-step, reset, playback speed and camera controls still
work. Manual gait/actuator/state controls do not override the connectome:
gait messages are ignored, state writes are rejected, and Neutral resets the
experiment. Neural readouts are recorded in run traces, not added to the
existing browser telemetry schema.

`--neural-backend gpu` accelerates the neural equations and sparse connectivity.
`--backend cpu` retains the original MuJoCo body physics and hybrid gait;
headless rendering still uses EGL/GPU. Do not pass `--backend gpu` to this
experiment: that selects the server's different MJWarp physics path.
Scene/terrain overrides are rejected rather than silently changing the original
Flywire experiment. The Y-maze remains available as a standalone experiment.

To opt in through the existing systemd unit without replacing its launch
command, add `FLYWIRE_CONFIG=/absolute/path/to/eon_config.json` to the unit's
environment using your normal deployment process. Optional settings are
`FLYWIRE_NEURAL_BACKEND=gpu` and `FLYWIRE_OUTPUT=/writable/path`.
No service files, ports, proxy settings or restart policies were changed.
Remove the environment variables to return to the original default simulation.

## Scientific behavior and output

The streaming experiment uses the copied Shiu equations at 0.1 ms, the Eon
readout populations, seeded Poisson input, and the configured 15 ms coupling:
sample odor -> advance the brain -> hold its decoded drive while advancing
the body for the same interval. It reuses the original `build_body` and
FlyGym `HybridTurningController`, adding only a tracking camera for streaming.
Reset restores the warmed body state, controller seed, neural state, delay
queue, rates and Poisson RNG. Reset episodes are marked in the trace.

Each run writes `manifest.json` (resolved settings, network audit, source/data
hashes) and `trace.jsonl` (inputs, readout rates, drive, spikes, position) under
`logs/flywire/<timestamp>`. Use `--flywire-condition` for `sensory`, `no-input`,
`direct-output`, or `disconnect-output`, and `--flywire-seed` for repeatability.
These remain local model approximations, not a full reproduction of Eon's demo.

## Preserved analysis and experiments

`src/flygym_stream/flywire/` contains the original conductance and Shiu models,
connectome loader/audits, motor decoders, Eon runner/verification/plotting,
hybrid runner, Y-maze runner/arena, tests, configurations and research notes.
Imports are package-relative; invoke tools with `python -m`, for example:

```bash
.venv/bin/python -m flygym_stream.flywire.load_network --config config/flywire/hybrid_config.json
.venv/bin/python -m flygym_stream.flywire.run_eon --config config/flywire/eon_config.json --backend gpu --neural-only --seconds .15
.venv/bin/python -m flygym_stream.flywire.run_hybrid --config config/flywire/hybrid_config.json --mode banc --backend gpu --neural-only --seconds .2
.venv/bin/python -m flygym_stream.flywire.run_ymaze --help
```

Use explicit configuration paths outside site-packages; standalone runners
write their `Outputs` beside the selected configuration. Original experiment
notes retain their historical commands/results and are not new server claims.
`tools/flywire/` preserves the original graph/coordinate visualization and body
preview scripts. Those are separate desktop/offline tools, not browser-stream
endpoints; their paths need configuring for your dataset. Graph tools additionally
need `pyarrow networkx plotly tqdm`.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p test_flywire.py -v
.venv/bin/python -m pytest
```

The integration tests check block timing, intervention behavior, deterministic
reset and controller stepping. Original equation tests remain in the package;
the readout annotation test requires the configured dataset, and Brian2 parity
tests require Brian2. Run the server's existing EGL/GPU tests on the GPU host
before deployment. Local Windows checks cannot establish Linux EGL performance
or the remote machine's installed dependencies.
