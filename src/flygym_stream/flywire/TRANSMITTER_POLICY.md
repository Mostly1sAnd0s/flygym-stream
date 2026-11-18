# Expanded transmitter approximation

The default config now uses `data.unmapped_transmitters = "guess"`. Previously
unsupported outgoing connections participate in propagation. This is an expanded
single-sign conductance approximation, not a complete neurotransmitter model.

Resolution order: per-neuron overrides take precedence; otherwise exact normalized
whole-label rules in `nt_class_map` win. For remaining labels, comma, semicolon,
plus, slash and pipe separators split mixtures. Acetylcholine aliases map to
excitation; GABA, glutamate and histamine aliases map to inhibition. A recognized
fast transmitter takes precedence over modulators or unknown mixture components.
Mixed fast signs use `transmitter_guesses.mixed_fast_class` (default `inh`).
Pure recognized modulators use `pure_modulator_class` (default `exc`). Everything
else, including blank labels, uses `unknown_class` (default `exc`). These options
accept `exc` or `inh`; exact mappings may also explicitly disable connections.

Pure modulator aliases include dopamine/DA, serotonin/SER/5HT, octopamine/OCT,
tyramine/TYR and nitric oxide/NO. All currently use the same 5 ms synaptic gate,
conductance scale and reversal potentials as the existing exc/inh classes.
Mixtures collapse to one effect; separate cotransmitter release is not modeled.

## Evidence and guesses

- Glutamate inhibition is supported in the antennal lobe by
  [Liu & Wilson (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3690841/).
  Extending that assignment across the network is an approximation: receptor
  context can change the sign.
- Histamine-gated chloride channels mediate inhibitory responses in Drosophila:
  [primary receptor study](https://pubmed.ncbi.nlm.nih.gov/15531107/).
  A shared GABA-like kinetic model remains an approximation.
- [Shiu et al. (2024)](https://www.nature.com/articles/s41586-024-07763-9)
  used inhibitory GABA/glutamate and excitatory dopamine/octopamine/serotonin in a
  simplified whole-brain model. This supports a comparison assumption, not a
  claim that modulators always excite or act through fast conductances. Our
  conductance equations differ from their model.
- Excitatory tyramine, nitric oxide, blank/unknown labels and inhibitory dominance
  for conflicting mixtures are user-authorized exploratory guesses. They are not
  established by those papers. Nitric oxide volume transmission, receptor-specific
  modulation, peptides and slow biochemical pathways remain unimplemented.

Every real-network report lists label assignments, which were inferred rather
than explicitly mapped, per-neuron overrides, and the synapse count assigned by
fallback rules. Explicit mappings are not automatically experimentally verified.
Verified metadata labels still take priority over predicted labels. Unknown
labels are deterministic guesses, so repeated runs are reproducible.

To compare uncertainty, change the three guess classes or add precise whole-label
or neuron overrides. Legacy `disable` and `error` policies remain available.
All changes act through neural propagation; the motor baseline remains zero.
More included connections do not imply more accurate behavior or guarantee walking.

Full local CSV audit after this change: 23,553,107 retained neuronal synapse counts,
zero disabled by transmitter classification, and 5,265,111 assigned using the new
fallback rules. Structural preflight passed. See
`Outputs/expanded_transmitter_audit.json` for per-label assignments. Unit tests
cover label resolution, legacy policies, overrides, and the direction of voltage
changes from the newly included classes.
