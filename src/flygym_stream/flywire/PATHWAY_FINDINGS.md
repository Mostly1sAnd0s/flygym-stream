# Pathway diagnostic findings — 2026-11-02

Completed CPU diagnostic: `Outputs/diagnostic_20261102T153823_546733Z/summary.json`.
Each condition started from rest and ran for 0.2 simulated seconds at 0.1 ms.
Odor used fixed antenna proxies; this is not a moving-body odor trial.

| Condition | Left/right sensory spikes | Left/right output spikes | Final motor drive |
|---|---|---|---|
| No input | 0 / 0 | 0 / 0 | 0 / 0 |
| Odor | 751 / 672 | 0 / 0 | 0 / 0 |
| Direct 700 pA output current | 0 / 0 | 18 / 19 | 1.2 / 1.2 |

Odor stimulates the sensory populations successfully. The sampled DNp09 voltages
never exceeded -65 mV, versus a -50 mV firing threshold. Peak sampled excitatory
conductances were 2.79/1.68 nS and inhibitory conductances were 36.34/27.73 nS.
These maxima need not coincide. The time series supports investigating excessive
inhibitory influence under the assumed physiology; it does not establish that
the real circuit has this balance. Simply increasing decoder gain cannot fix zero
output spikes.

The loader audit passed: 157,789 neurons, no missing endpoint metadata, and
directed paths to both outputs. A synthetic matrix check confirmed that a pre→post
edge contributes to the postsynaptic row. Metadata confirms 45 ORN_DM1 neurons on
each correct sensory side and two cholinergic DNp09 neurons. The output mapping
intentionally swaps anatomical sides into controller channels, as documented in
the config; its behavioral steering sign remains unvalidated. Verified transmitter
labels take precedence over predictions. The configured policy omits 5,265,111 of
23,553,107 synapse counts (~22.4%); omitted effects cannot safely be assigned an
arbitrary common sign.

GPU embodied smoke tests (0.3 simulated seconds, no viewer):

- Direct-output current plus default pulse input: 1.58 mm XY displacement,
  nonzero output rates/commands. Run `hybrid_20261102T153615_769602Z`.
- Odor plus explicit baseline 0.8: 2.52 mm XY displacement; output rates remained
  zero. Movement was imposed by the baseline, not neural odor guidance.
  Run `hybrid_20261102T153705_706552Z`.

These short tests demonstrate interface response, not sustained stable walking or
odor attraction. No biological parameters were retuned: there is no calibration
evidence here justifying a new threshold, global inhibition scale, background
current, or transmitter assignment. Follow-up requires physiological constraints
on relevant cells/receptors and testing steering with unilateral output activation.
The diagnostic GPU attempt stalled during instrumentation initialization and was
stopped; completed diagnostics used CPU. The simulation smoke tests used GPU.

The baseline-walking override was subsequently removed at the user's request;
the baseline trial above is retained only as a historical diagnostic result.
`--output-current-pa` remains an explicit diagnostic intervention. Neural control
now rejects nonzero decoder baselines/minimum drive. Syntax, diff whitespace, matrix
direction, and decoder baseline/clipping checks passed.
