# Research priorities for neural propagation

Subsequent implementation: `TRANSMITTER_POLICY.md` documents the user-requested
expanded sign approximation now enabled by default. References below to disabled
transmitters describe the earlier policy. Receptor-specific physiology remains a
research priority; the expanded sign policy does not implement those mechanisms.

Reviewed 2026-11-02. No physiological constants were changed by this review.
The motor baseline override has been removed. The decoder rejects nonzero
baseline/minimum drive, so zero output activity produces zero motor command.
Fixed-controller and direct-stimulation modes remain explicit diagnostic tests.
The existing FlyGym leg controller still translates neural output into movement;
this is not yet a biological VNC-to-muscle model.

## What the current experiment establishes

In the 0.2 s fixed-position odor test, sensory neurons fired but DNp09 did not.
Large inhibitory conductances accompanied low output voltages. This identifies a
failure in this approximation, not proof that real DNp09 inhibition is excessive.
Counts of anatomical synapses alone do not specify conductance or receptor action.

## Recommended changes, in implementation order

1. **Separate receptor mechanisms from presynaptic transmitter labels.**
   Liu & Wilson (2013) experimentally demonstrate GluClα-mediated inhibitory
   glutamate effects in the antennal lobe. Our model disables glutamate entirely.
   Add receptor-class sparse matrices with their own reversal potentials and
   kinetics; apply assignments only where cell/region evidence supports them.
   Do not globally make glutamate excitatory to compensate for silent outputs.
   Unresolved edges need explicit uncertainty scenarios, not guessed certainty.
   [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3690841/).

2. **Distinguish presynaptic gain control from postsynaptic inhibition.**
   Root et al. (2008) show that ORN GABA-B receptors regulate transmission and
   dynamic range. Our model treats every retained inhibitory edge as the same
   fast postsynaptic conductance. Introduce a separate slow presynaptic modulation
   state for supported connections, with independent kinetics. Receptor locations
   and time constants must be established before assigning this mechanism to
   individual edges; the paper does not justify reclassifying every GABA edge.
   [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2539065/).

3. **Calibrate ORN-to-PN transmission before downstream walking.**
   Kazama & Wilson (2008) measured strong ORN-to-PN synapses, short-term depression,
   and matching of synaptic currents to postsynaptic properties. They also report
   spontaneous ORN activity. Our single 0.3 nS-per-contact weight, common membrane
   properties, and saturating release gate do not reproduce those mechanisms by
   construction. Add a distinct release-resource/recovery state and fit single
   spike and train responses at these identified synapses. The existing gate's
   saturation is not a calibrated vesicle-depletion model. Add spontaneous sensory
   activity only with measured cell-class rates, not global current chosen to
   provoke walking. Neural spontaneous activity is different from imposed motor
   drive: all movement would still require neural propagation.
   [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2429849/).

4. **Use a published propagation model as a comparison, not a parameter patch.**
   Shiu et al. (2024) provide a whole-brain LIF reference with signed synaptic
   effects and an experimentally anchored synaptic scale. Their equations differ
   from our conductance model. Reproduce the reference equations, delays, input
   protocol and validation benchmark in a separate backend before comparing
   pathways. Do not copy a voltage-valued synaptic weight into our nS parameter.
   Their transmitter simplifications are model assumptions, not receptor truth;
   their feeding/grooming validations do not validate our locomotor mapping.
   [Primary paper](https://www.nature.com/articles/s41586-024-07763-9).

5. **Validate the sensory-to-descending pathway being queried.**
   Bidaye et al. (2020) associate P9/DNp09 with object-directed walking and
   ipsilateral turning, including visual and courtship inputs; BPN supports a
   different walking program. This does not establish that stimulating DM1 alone
   must activate DNp09. Keep the output mapping explicit and inspect intermediate
   neurons rather than lowering DNp09 thresholds until it responds.
   [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9435592/).
   Wind-guided odor navigation involves additional central circuitry and sensory
   inputs. A static Gaussian odor field alone cannot test that complete behavior.
   Any added wind signal should enter sensory neurons, not set motor commands.
   [Primary experimental paper](https://www.nature.com/articles/s41467-022-32247-7.pdf).

## Acceptance criteria

Record the source and units of each new parameter; retain unknown assignments as
unknown. Validate single-neuron responses, synaptic impulse/train responses,
delay convergence, and ORN/PN population responses before asking for walking.
Compare no-input, odor, and direct-output diagnostic trials from identical reset
states. Track excitation/inhibition and firing in intervening cell classes.
Hold out stimulation strengths/conditions from calibration, report uncertainty,
and require zero motor drive for zero decoded output. Walking by itself is not a
physiological validation criterion.

Most useful next implementation: receptor-class infrastructure plus an identified
ORN-to-PN calibration harness. It can improve propagation fidelity without
injecting movement or making unsupported global changes to inhibition.
