# Programmable Y-maze for your hybrid fly

No CAD model is required. `ymaze_arena.py` creates walls from primitive MuJoCo boxes. The existing FlyGym fly meshes are reused. This add-on requires your existing FlyGym 2.1 environment; no new dependencies beyond its requirements are added.

## Installation

Copy these three files into the existing folder:

`C:\Users\sysco\Documents\GitHub\Flywire-Brain-Model\hybrid_fly`

- run_ymaze.py
- ymaze_arena.py
- ymaze_config.json

Keep your existing neural_model.py, load_network.py, hybrid_config.json and run_hybrid.py. This add-on does not replace them. Extract the files directly into hybrid_fly, not into another nested folder.

Open PowerShell in your project root:

```powershell
cd C:\Users\sysco\Documents\GitHub\Flywire-Brain-Model
```

## Start here: a visible scripted test

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_ymaze.py --mode scripted --arm left --odor left
```

The fly follows position-guided waypoints into the left arm. This mode does NOT load the neural network, and odor does NOT determine the scripted choice. It tests maze geometry, gait, steering, logging and choice detection. The window closes after a committed choice, escape, or the time limit (15 simulated seconds by default). The trajectory image remains available afterward. Startup can take longer the first time as libraries compile code/load body meshes.

Test the other arm:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_ymaze.py --mode scripted --arm right --odor right
```

Scripted mode is much lighter than full-network mode, but physics still has a cost. No real-time performance is guaranteed on your computer.

## Design without running physics

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_ymaze.py --preview
```

This writes a top-down trajectory.png showing the empty maze and odor markers. It does not open MuJoCo or load the brain. Open the path printed by the command.

Edit ymaze_config.json to change stem length, arm length, corridor width, branch angle, wall dimensions, source location and choice dwell time. Dimensions are millimetres. Defaults are a 12 mm-wide corridor, 18 mm stem, 22 mm arms, branches at plus/minus 45 degrees, and 5 mm-high walls. This is a convenient engineering test arena, not a standardized published assay. Keep clearance for the fly's legs. Very narrow or extreme geometry may prevent walking; invalid combinations are rejected.

The stem points along +x. The left arm is +y and right arm -y when the fly starts facing forward. Walls have explicitly enabled fly-body collisions, unlike purely visual geometry. All body segments can contact the walls; only the usual selected feet/leg segments contact the ground. The open top permits climbing/escape, which is recorded if the thorax leaves the polygon or rises more than 1 mm above the wall height. This is not a full-body containment test.

## Odor conditions

Use `--odor left`, `--odor right`, `--odor both`, or `--odor none`.

The field is an abstract Gaussian proxy sampled at the antenna body origins. A green marker indicates an active source; a grey marker an inactive source. Markers do not collide with the fly. Both-source fields add and are clipped to 1; left/right samples need not be equal if the fly is off-center. There is no fluid simulation or wall occlusion: concentration can mathematically extend across walls. This is suitable for testing signal delivery, not claiming a realistic laboratory plume. No simulated vision is connected to marker colors.

## Connect the brain later

Use the same real-data files/configuration that already work with run_hybrid.py:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_ymaze.py --mode banc --odor left --seconds 5
```

BANC mode ignores the scripted --arm target entirely. Antenna odor measurements drive the configured sensory neurons, and configured output rates drive the hybrid controller. It does NOT force a choice, follow waypoints or fall back to scripted motion. It inherits the transmitter omissions, parameter assumptions and readout mapping in your brain configuration. No contact or proprioceptive signal is injected into the brain; the existing hybrid controller still uses its own body observations. VNC-to-muscle control remains bypassed.

The default neural step is 0.1 ms. An optional faster approximation updates the brain every 0.5 ms while leaving physics at its normal timestep:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_ymaze.py --mode banc --odor left --seconds 5 --neural-dt-ms 0.5
```

The runner averages sensory samples over each completed neural interval, advances the neural state and holds its commands for the next interval. The first interval uses zero commands. A larger timestep changes temporal accuracy and can change spikes/behavior; it must be compared against the 0.1 ms baseline before scientific interpretation. This option is local to this new runner; it does not change run_hybrid.py. It reduces neural updates fivefold, not necessarily total runtime fivefold.

## Output and interpretation

Each run creates a new `hybrid_fly/Outputs/ymaze_TIMESTAMP/` folder:

- trial_config.json: full maze and brain configurations, arguments, seed and timesteps.
- trace.csv: approximately 50 Hz thorax trajectory, heading, antenna odor samples, controller commands, decoder-channel firing rates, zones, committed choice and wall-contact counts. Sensory samples/heading are from the start of each physics step; positions and updated commands are recorded at its end.
- result.json: status, first committed choice, decision time, simulated/wall time and compute timing.
- trajectory.png: top-down maze and recorded path.

A choice requires the thorax center to remain in the final 35% of an arm for 0.25 seconds. Leaving the zone resets the timer. Decision time is measured from the end of initial body warmup to completion of that dwell. A trial stops at the first committed choice. Timeout has no choice. Viewer closure is recorded separately and should not be counted as a no-choice behavioral trial. Escape is recorded separately. `wall_contact_samples` counts logged samples with contact, not distinct collisions. `output_channel_L_hz`/`R_hz` refer to decoder channels; your existing crossed DNp09 mapping means they are not anatomical hemisphere labels.

Run reproducible controls with mirrored source positions, both-source and no-source conditions. `--seed` controls the hybrid gait controller's random seed; it does not randomize neural parameters or starting position. Identical deterministic runs are not independent biological replicates. The runner supports one trial per invocation; use distinct invocations for conditions, and record all outcomes.

## Other options

- `--seconds 30`: extend trial time limit, in simulated seconds.
- `--no-viewer --fast`: headless execution with no real-time pacing.
- `--fast`: remove real-time pacing while keeping the viewer.
- `--maze-config PATH`: another arena configuration.
- `--brain-config PATH`: another existing neural configuration.
- `--output PATH`: custom output folder (use a fresh path; existing files of the same names are overwritten).

Use Ctrl+C to interrupt. Once the main simulation loop has started, it attempts to save partial results. Full-network loading and body construction occur before this loop.

## Validation

See VALIDATION.md for measured checks and remaining limitations. A scripted success validates the arena/controller interface; it is not evidence of successful neural odor navigation.
