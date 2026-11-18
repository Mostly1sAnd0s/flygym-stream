# Eon comparison baseline

This separate profile applies the reproducible portions of Eon's methodology to
the existing local network. It is **not an exact reproduction** of Eon's embodied
demonstration. Keep this baseline separate from future biological extensions.
The original conductance runner/configuration remains available.

## Sources reviewed 2026-11-12

- [Eon embodied methodology, March 10, 2026](https://eon.systems/updates/embodied-brain-emulation).
- [Eon public Brian2 implementation](https://github.com/eonsystemspbc/fly-brain/blob/main/code/run_brian2_cuda.py).
- [Shiu original implementation](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py).
- [Sapkal et al. primary study](https://www.nature.com/articles/s41586-024-07854-7).

These are mutable upstream URLs, not a pinned Eon software release. Local run
manifests record SHA-256 hashes of the actual input data, configuration and core
local implementation files. No Eon body-controller code or weights were imported.

## Matched components and local choices

| Component | Implementation here | Fidelity boundary |
|---|---|---|
| Neural equations | Exact linear update of voltage and voltage-valued synaptic state; 0.1 ms steps | Compared analytically; independent Brian2 comparison must also pass |
| Neural constants | Rest/reset -52 mV, threshold > -45 mV, membrane 20 ms, synaptic 5 ms, refractory 2.2 ms, delay 1.8 ms, 0.275 mV per anatomical synapse | Shiu/Eon reference constants; no conductance-unit substitution |
| Sensory activation | Independent Bernoulli events equivalent to N=1 Brian2 PoissonInput per step; 68.75 mV event; zero refractory on declared input neurons | Default 100 Hz follows current Eon neural benchmark; actual embodied sensory calibration unavailable |
| Connectivity | All retained local metadata and positive matched connection rows; duplicate pairs summed | Local export is not established as the exact Eon dataset |
| Synapse signs | Existing verified-over-predicted transmitter policy | Mixed labels, unknowns, and modulator approximations retained and audited; not asserted to match Eon |
| Synchronization | Sample input, evolve brain 15 ms, decode block spike counts, advance body 15 ms | Actual block coupling, not merely a 15 ms motor readout |
| Steering | DNa01/DNa02 on each side | Equal averaging, gains and differential leg-drive mapping are local choices |
| Forward readout | oDN1, locally annotated as DNg97 | Positive speed gain is an engineering interpretation of Eon's summary, not physiology; Sapkal also implicates these neurons in halting |
| Body | Existing FlyGym 2.1/MuJoCo hybrid walking controller | Eon's modifications and complete controller mapping not supplied by the cited public code |
| Other readouts | aDN1/DNg62 grooming and MN9 feeding rates logged | No grooming/feeding actuation or behavioral selection implemented |

`eon_config.json` identifies all output neurons explicitly. Both primary and
alternative cell-type annotations are verified before loading the graph; steering
sides are checked. The loader also checks edge presence and directed structural
reachability for every output. These checks do not establish functional responses.

The environment is the existing abstract Gaussian odor field sampled at the two
antennae, mapped to ORN_DM1. Neural-only tests prescribe bilateral levels 1 and
0.35. This differs from Eon's taste/dust demonstration. Taste, dust, visual-model
coupling, behavioral arbitration, feeding and grooming body controllers remain
reproduction gaps, not completed features or proposed biological improvements.

## Run from repository root

```powershell
# Full local connectome, 15 ms coupling, headless body and odor feedback
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --seconds 1.5

# Show the body viewer
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --seconds 1.5 --viewer

# Prescribed sensory input, no body
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --neural-only --seconds 0.15

# Controls, each resets the neural state and random seed
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --seconds 0.15 --condition no-input
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --seconds 0.15 --condition direct-output
.\.venv-body312\Scripts\python.exe hybrid_fly/run_eon.py --seconds 0.15 --condition disconnect-output
```

Use `--backend cpu` for CPU; default inherits the project's GPU preference.
Durations must be exact multiples of 15 ms. Direct-output is artificial Poisson
stimulation of the locomotor readouts, bypassing the sensory pathway.
Disconnect-output preserves neural activity but forces motor commands to zero.
Zero commands need not imply zero body displacement: settling and physics remain.

Each run writes `manifest.json`, `trace.csv` (one row per completed block),
`spike_counts.npz` (all IDs and spike counts), and `summary.json` under a unique
`hybrid_fly/Outputs/eon_*` directory. A missing summary means the run did not
complete successfully. Feed/groom columns are neural observations only.

## Verification

```powershell
.\.venv-body312\Scripts\python.exe -m unittest discover -s hybrid_fly -p test_eon.py -v
.\.venv-body312\Scripts\python.exe -m unittest discover -s hybrid_fly -p test_shiu_reference.py -v
.\.venv-body312\Scripts\python.exe -m unittest discover -s hybrid_fly -p test_transmitter_policy.py -v
```

Acceptance must distinguish: equations, implementation parity, graph coverage,
closed-loop execution, and biological behavior. A walking fly alone does not
validate the sensory pathway. No gain tuning to force a positive result is part
of this baseline. See EON_VERIFICATION.md for measured results and blockers.
