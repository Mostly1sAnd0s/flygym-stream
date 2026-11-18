# Connect your CSV network to the existing hybrid fly

Current transmitter policy: the default config uses an expanded, explicitly
approximate sign assignment for previously unsupported transmitters and unknown
labels. See [TRANSMITTER_POLICY.md](TRANSMITTER_POLICY.md) for rules, evidence,
uncertainty and overrides. Historical disabled-transmitter counts below describe
the earlier configuration; use a fresh audit for the current configuration.

This patch uses real IDs from your uploaded neurons.csv. It is compatible with the previously supplied run_hybrid.py and neural_model.py and does not replace either. It requires the existing NumPy, pandas and SciPy dependencies. The audit itself does not import FlyGym.

## Install

Neural simulation now defaults to the CuPy GPU backend, including BANC mode in
`run_ymaze.py`. Set `neural.backend` in `hybrid_config.json` or pass
`--backend cpu` to either runner to select CPU explicitly. GPU mode requires
CuPy compatible with your NVIDIA CUDA driver/runtime; it reports an error if GPU
initialization fails. FlyGym/MuJoCo body physics and CSV loading still run on CPU.
The base `requirements.txt` contains the CPU dependencies; install a CuPy build
matching your CUDA environment separately.

Both runners now read neural motor outputs every 1 ms and hold the last command
between readouts. Neural integration remains at its existing timestep (0.1 ms by
default). Use `--motor-dt-ms 0.1` to restore the original readout frequency, or
`--motor-dt-ms 1` for the new default. Actual intervals round to whole simulation
steps (whole neural updates in the Y-maze). This sample-and-hold can change body
trajectories; full-network behavior still needs validation.

`run_hybrid.py` writes `timing.json` with body compute time, neural host submission
and wait time, and motor readout count. Neural time is not isolated GPU kernel
time: asynchronous GPU work can overlap CPU physics. Viewer and logging overhead
are included in total wall time. Use `--no-viewer` for timing comparisons.

1. Close a running simulation. Keep a backup of your current hybrid_config.json and load_network.py.
2. Extract this archive. Copy its files directly into:
   C:\Users\sysco\Documents\GitHub\Flywire-Brain-Model\hybrid_fly
3. Replace hybrid_config.json and load_network.py when prompted. Do not put another hybrid_fly directory inside that directory.
4. Keep your data files in the existing Data folder. No CSV modifications are needed.
5. Open PowerShell in the project root:

```powershell
cd C:\Users\sysco\Documents\GitHub\Flywire-Brain-Model
```

## First: audit the actual connections

### Walking diagnostics

From the project root, run the isolated no-input, fixed-position odor, and direct
DNp09 current tests (writes population voltages, conductances, rates and spike
counts plus the connectivity audit under `hybrid_fly/Outputs/diagnostic_*`):

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\diagnose_pathway.py --seconds 0.5
```

Test the neural-to-body interface with a diagnostic 700 pA current to both output
populations:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_hybrid.py --mode banc --output-current-pa 700 --seconds 3
```

Neural motor commands require zero baseline and zero minimum drive. Nonzero values in custom configs are rejected. See PROPAGATION_RESEARCH.md for the research review.
Direct current is also diagnostic, not a calibrated physiological intervention.
Conductance snapshots in the diagnostic include newly released gates; sampled
voltage maxima are not exact pre-reset spike peaks. All trials reset neural state.

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\load_network.py
```

This inspects the full connections_princeton.csv, saves hybrid_fly/network_audit.json when the graph can be constructed, and prints PRECHECK PASSED or a specific error. It checks exact IDs, rejects unmatched connection endpoints, excludes explicitly non-neuronal nodes and incident edges, checks selected populations have edges, and checks directed structural reachability from the combined sensory input population to both selected outputs. An inhibitory path counts as a structural path; this does NOT establish that the output will fire. It reports which VNC classes have edges. Overlapping IDs and class coverage do not prove release identity or complete anatomical coverage: retain original export provenance.

Send network_audit.json back for interpretation. If the script fails before saving it, send the terminal error. A missing-ID error is a reason to obtain matched exports, not to disable the check or invent connections.

## Then: run a short neural test

Only after the audit passes:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_hybrid.py --mode banc --neural-only --seconds 0.2
```

This loads the real network and injects prescribed currents into ORN_DM1. It does not open the fly. The terminal rates are OUTPUT population rates; zero output does not imply the sensory neurons were silent. Inspect the printed report and Outputs/hybrid_*/trace.csv. This is a computational smoke test, not a biological validation. Whole-network runs may be much slower than real time, and loading can use substantial RAM.

## Connect the body and odor field

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_hybrid.py --mode banc --input odor --seconds 1
```

The body samples an abstract Gaussian odor field at the two antenna origins. Those values inject current into the selected sensory neurons. DNp09 activity is decoded into the hybrid controller's two leg-drive commands. The green marker indicates the source. This is not a calibrated vinegar plume or a validated odor-navigation model. Network activity may not activate DNp09 under the current parameters. There is no fixed-drive fallback: if output activity is zero, commanded drive stays zero.

## Optional: isolate the real-neuron-to-body interface

If the sensory run has zero output, this alternative explicitly injects current directly into the two real DNp09 neurons while still loading the same CSV network:

```powershell
.\.venv-body312\Scripts\python.exe .\hybrid_fly\run_hybrid.py --mode banc --config .\hybrid_fly\hybrid_motor_check.json --seconds 1
```

This is an artificial stimulation diagnostic, not an odor experiment and not evidence of an intact sensory pathway. Unlike the old demo, it still uses real neuron IDs and the loaded connectivity. It can separate a decoder/body issue from failure of sensory-to-output activation. Do not use --input odor with this diagnostic.

## What is configured

- 45 left and 45 right ORN_DM1 neurons, all explicitly verified acetylcholine in your file (one has a conflicting GABA prediction; verified annotation wins).
- Left DNp09: 720575941566493282.
- Right DNp09: 720575941433155799.
- Engineered output convention: right DNp09 activity feeds left leg drive, and left DNp09 activity feeds right leg drive. This is motivated by ipsilateral turns from DNp09 stimulation; the exact gains and turning sign must be verified in the body controller. The existing trace's rate_L_hz/rate_R_hz are decoder-channel rates, so they correspond to right/left DNp09 respectively, not anatomical left/right.
- All 157,789 retained metadata nodes receive neural state. This is not a 92-neuron network. Only 90 nodes receive external sensory input, and two are read out.
- 458 glia, 8 trachea and 7 not_a_neuron rows excluded. Unlabelled rows retained; retention does not certify that each is a complete neuron.
- All positive matched neuronal edge rows are read; duplicate neuron-pair rows are summed by the existing neural model. No additional synapse threshold is imposed.

## Explicit limitations of this first configuration

The existing NeuralModel supports one excitatory or inhibitory conductance class per presynaptic neuron. It cannot implement receptor-specific effects, cotransmission, neuromodulatory chemistry or glial physiology.

This patch therefore explicitly uses data.unmapped_transmitters = "disable". ACh is mapped to excitation and GABA to inhibition. Exact listed ACh/GABA-plus-modulator labels preserve only that fast component; the accompanying dopamine, serotonin, octopamine or nitric-oxide effects are omitted. Ambiguous fast combinations such as ACh+GABA and ACh+histamine are disabled. Glutamate, histamine, pure modulators and unknown labels are also disabled rather than given invented universal signs.

The metadata-only allocation is 89,219 excitatory, 21,178 inhibitory and 47,392 with disabled outgoing effects. Disabled nodes retain LIF state and can receive signals, but do not influence downstream nodes. The connection audit reports how many of these actually have outgoing edges and how many synapses are omitted. These omissions can materially change dynamics and behavior; this is NOT a complete full-brain physiological model. To reject any such omissions, set unmapped_transmitters to "error"; the supplied data will then fail until the model is extended.

VNC neurons and matched recurrent connections are included, but movement is decoded from descending neurons into the existing hybrid gait controller. VNC motor-neuron outputs are not driving muscles. Parameters and rate-to-drive gains remain hypotheses rather than fitted measurements. No coordinate file is needed by this point-neuron model.

## Evidence for candidate selection

- Bidaye et al. (2020), Two Brain Pathways Initiate Distinct Forward Walking Programs in Drosophila: https://pubmed.ncbi.nlm.nih.gov/32822613/ . DNp09/P9 is linked to forward walking and ipsilateral turning.
- Semmelhack and Wang (2009), Select Drosophila glomeruli mediate innate olfactory attraction and aversion: https://pmc.ncbi.nlm.nih.gov/articles/PMC2702439/ . DM1/Or42b contributes to food-odor attraction.

These studies motivate candidate input and output classes separately. They do not establish that this dataset, transmitter approximation and generic LIF parameters produce a functional ORN_DM1-to-DNp09 pathway.

## Validation performed here

The uploaded metadata and all 92 configured population IDs were inspected. Small synthetic edge fixtures tested exact large-ID preservation, duplicate summation, non-neuronal filtering, verified-over-predicted transmitter precedence, active-path reachability, strict-mode rejection, missing-metadata rejection, and finite neural updates against the original NeuralModel. These fixtures test software mechanics only. Your actual connections CSV was not uploaded, so full graph coverage, memory cost, full-network dynamics, and embodied behavior remain untested. The local audit is the next necessary check.
