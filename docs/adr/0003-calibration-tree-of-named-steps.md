---
status: proposed
---

# The calibration tree is a flat list of named steps, each naming its experiment, analysis and operating point

The calibration tree must be written and read by people, and the same experiment appears several times in it: with different fits (Rabi for readout levels, Rabi for a pi amplitude), producing different operating points (two-tone for the sweet spot, two-tone for the readout point), and in places that do not follow one another (sweet-spot calibration is the root of every branch). The previous format (`node_types → pipelines`, run in YAML order, with variants encoded as `two_tone-sweet_spot` suffixes) could only express this through string conventions and file order. We replaced it with a flat set of uniquely named **steps**, each naming the experiment it measures, the **analysis** that turns the measurement into results, and the operating point it runs at and writes into.

```yaml
sample: ConscienceWF02-3.7
cooldown: CD79-Mimas
targets: [q1, q14]

defaults:
  two_tone: {freq_sweep: [-5e6, 5e6, 101], averages: 10000}
  rabi:     {amplitude_sweep: [0.02, 0.8, 40], averages: 10000}

steps:
  sweet_spot_flux:
    experiment: two_tone_vs_flux
    analysis: parabola_vertex
    operating_point: sweet_spot
  readout_flux:
    experiment: two_tone
    analysis: readout_point_search
    operating_point: readout
    after: sweet_spot_flux
  rabi_readout:
    experiment: rabi
    analysis: readout_levels
    operating_point: readout
    targets: [q14]
    overrides: {q14: {amplitude_sweep: [0.02, 0.5, 25]}}
```

- A step name says what the step is for; `experiment` says what it measures. The same experiment can appear in any number of steps.
- Each experiment registers the analyses it supports and a default one.
- A step writes into the operating point it runs at; its analysis decides which fields. Writing to an experiment section is the explicit exception.
- `after` names the step (or list of steps) that must have succeeded for that target first. When omitted, it defaults to the previous step in the file, so a linear tree needs no `after` at all.
- Parameters resolve in three levels, each overriding the one before: `defaults.<experiment>`, the step's own values, and `overrides.<target>`.
- `targets` at the top is the default for every step; a step can narrow it and set its own `simultaneous`.
- A `check` command validates the file without touching hardware (unknown experiments or analyses, unknown `after` names, cycles, operating points that nothing provides) and prints the resulting graph per target. The same graph opens the PDF report.

## Considered options

- **Grouped branches** (`branches: {t1: {after: …, steps: […]}}`), mirroring the proposal page: rejected, because a step shared by two branches has to live outside both and be referenced anyway, and the file gains a level of nesting. Branch structure is shown by `check` instead.

## Consequences

- Existing trees in the `node_types`/`pipelines` format, and the example in `examples/config_files/` (which still uses an even older `nodes`/`dependencies` format), have to be rewritten.
