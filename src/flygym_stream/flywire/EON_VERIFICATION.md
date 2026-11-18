# Eon baseline verification — 2026-11-12

Status: **CPU neural checks and a short closed-loop body run passed; independent
Brian2 parity and GPU execution remain blocked by Windows Application Control.** This is not a
verified reproduction of Eon's complete embodied demo.

## Subsequent retry on the same day

Windows still reports Smart App Control enabled. No security settings were changed.
MuJoCo now imports and the CPU sensory/body trial completed 0.15 seconds in ten
15 ms blocks with finite body state, 43,618 neural spikes across 5,186 neurons,
maximum drive 0.8333, and 0.1320 mm net horizontal displacement. Output:
`Outputs/eon_20261112T153520_460658Z`. Position-dependent odor was sampled from the
body each block. This establishes short closed-loop execution, not navigation or
biological validity; displacement includes settling and passive dynamics.

Two additional CPU body controls completed ten blocks each with finite body state:

| Body condition | Total spikes | Maximum drive | Net horizontal displacement |
|---|---:|---:|---:|
| Sensory feedback | 43,618 | 0.8333 | 0.1320 mm |
| No input | 0 | 0 | 0.0645 mm |
| Direct output stimulation | 1,419 | 1.2 | 1.0872 mm |

The no-input body shifts under passive physics despite zero neural activity and
commands. The direct-output diagnostic demonstrates the neural-to-controller
interface, not a sensory pathway. Its spike totals match the neural-only direct
trial. The control directories are `Outputs/eon_20261112T153648_693922Z` (no input)
and `Outputs/eon_20261112T153650_043943Z` (direct output).
These short runs do not establish stable gait, correct turning sign, or navigation.

Brian2 now imports, but both reference comparisons fail when loading
`brian2.synapses.cythonspikequeue` (Application Control). CuPy now reaches a later
import before failing at `cupy.random._generator_api` with the same restriction.
The failures described below are the original attempts; these are the latest
dependency results.

The saved neural-trial plot is [eon_neural_results.png](Outputs/eon_neural_results.png).
Regenerate it with `plot_eon_results.py`. It depicts the four original neural-only
trials, not the body trajectory.

## Completed

- Eight equation/interface tests passed (`test_eon.py`): impulse timing and
  duplicate summation, inhibitory response, external-event ordering, invalid
  inputs, seeded reproducibility, refractory boundary, decoder symmetry/timing,
  and exact local readout annotations.
- Three existing transmitter-policy regression tests passed.
- The existing zero-input/timestep test in `test_shiu_reference.py` passed.
- Full-network loading and structural preflight passed for 157,789 retained
  metadata neurons, the 90 configured ORN_DM1 inputs, and all ten selected
  readout neurons. This does not establish that the export is Eon's dataset.
- Four CPU trials completed ten 15 ms blocks each (0.15 simulated seconds),
  seed 0, default 100 Hz maximum sensory/direct stimulation rate.

| Condition | Total spikes | Active neurons | Maximum commanded drive |
|---|---:|---:|---:|
| Prescribed sensory input | 43,598 | 5,240 | 1.2 |
| No input | 0 | 0 | 0 |
| Direct locomotor-output stimulation | 1,419 | 508 | 1.2 |
| Sensory input, motor output disconnected | 43,598 | 5,240 | 0 |

The sensory and disconnected trials match **every neuron's total spike count**
and every block's readout rates and population spike total, not merely aggregate
spike counts. Configurations, seeds, neural source hashes and data hashes match.
The sensory run used the runner before additional package/effective-parameter
metadata fields were added; the neural implementation and protocol were unchanged.

Sensory stimulation first produced steering readout activity in the block ending
at 60 ms. oDN1, aDN1 and MN9 readouts remained silent throughout this short trial.
Nonzero commands came from asymmetric steering; this is not evidence of a
functional forward-speed pathway, navigation, feeding, or grooming. Commands
reached the configured cap. No gains were adjusted to obtain these results.

The runs took approximately 44–45 seconds each inside the measured section;
loading and auditing occurred before that timer. This is not real-time execution
or a controlled performance benchmark; some trials ran concurrently.

Machine-readable comparison: [eon_verification.json](Outputs/eon_verification.json).
Raw runs contain manifests, traces and complete per-neuron spike counts.
`verify_eon_runs.py` can repeat the comparison on four new matched run directories.

## Blocked checks

These failures also occurred when Python ran outside the execution sandbox:

1. Brian2: importing `brian2.memory.cythondynamicarray` failed with an Application
   Control DLL restriction. The independent recurrent solver test could not run.
   An additional deterministic external-event replay comparison is supplied but
   remains unexecuted because of that dependency failure.
2. GPU: CuPy's `_routines_logic` extension was blocked by Application Control.
   GPU/CPU parity and GPU throughput are unverified.
3. Body: MuJoCo's bundled plugin loader failed with WinError 4551, Application
   Control blocked this file. The attempted embodied direct-output run is
   `Outputs/eon_20261112T151027_535544Z`; its manifest exists, but it has no
   successful summary or body trajectory.

A working policy-approved installation of those dependencies is required to
finish verification. No security settings were changed. Once available, rerun
the Brian2 comparisons, matched GPU trials, and embodied sensory/no-input/
direct-output/disconnected-output trials using the documented commands.

## Reproduction gaps

The current deliverable is an Eon-inspired locomotion comparison baseline.
Eon's taste, dust, visual-system coupling, feeding/grooming actuation and exact
behavior-controller mappings are not reproduced. Positive oDN1 speed gain,
15 ms block rate estimation and steering gains are documented local engineering
choices, not recovered Eon parameters. See [methodology](EON_METHODOLOGY.md).

Future biological improvements should be evaluated against this recorded
baseline, while keeping these verification and reproduction gaps explicit.
