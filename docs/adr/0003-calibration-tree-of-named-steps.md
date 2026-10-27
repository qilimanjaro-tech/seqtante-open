---
status: accepted
---

# The calibration tree is a set of pipelines of named steps, each step naming its experiment, analysis and operating point

The calibration tree must be written and read by people, including operators at the sites we deploy to, and the same experiment appears several times in it: with different fits (Rabi for readout levels, Rabi for a pi amplitude), producing different operating points (two-tone for the sweet spot, two-tone for the reset point), and in places that do not follow one another (sweet-spot calibration is the root of every branch). Some parts also run on their own: the reset point and its readout levels have to be recalibrated right before Landau-Zener, much more often than a full recalibration. The previous format (`node_types → pipelines`, run in YAML order, with variants encoded as `two_tone-sweet_spot` suffixes) could only express this through string conventions and file order. We kept **pipelines** as the unit operators turn on and off and that a failure is scoped to, and made each pipeline an ordered set of uniquely named **steps**, each naming the experiment it measures, the **analysis** that turns the measurement into results, and the operating point it runs at and writes into.

```yaml
sample: ConscienceWF02-3.7
cooldown: CD79-Mimas
targets: [q1, q14]

defaults:
  two_tone: {freq_sweep: [-5e6, 5e6, 101], averages: 10000}
  rabi:     {amplitude_sweep: [0.02, 0.8, 40], averages: 10000}

pipelines:
  sweet_spot:
    steps:
      sweet_spot_flux:
        experiment: two_tone_vs_flux
        analysis: parabola_vertex
        operating_point: sweet_spot
        loop: z
        flux_sweep: [-0.02, 0.02, 11]
      two_tone_sweet_spot:
        experiment: two_tone
        analysis: lorentzian_peak
        operating_point: sweet_spot

  t1:
    after: two_tone_sweet_spot
    steps:
      t1_sweet_spot:
        experiment: t1
        operating_point: sweet_spot
        thresholds: {passed: 50e3, warning: 20e3}

  t2:
    after: two_tone_sweet_spot
    steps:
      rabi_sweet_spot:
        experiment: rabi
        analysis: pi_amplitude
        operating_point: sweet_spot
        pulses: {drive: drag}
      t2_sweet_spot:
        experiment: t2
        operating_point: sweet_spot
        thresholds: {passed: ..., warning: ...}

  lz:
    after: sweet_spot_flux
    targets: [q14]
    steps:
      reset_flux:
        experiment: two_tone
        analysis: reset_point_search
        operating_point: reset
        target_frequency: [2.79e9, 2.81e9]
      rabi_reset:
        experiment: rabi
        analysis: readout_levels
        operating_point: reset
        overrides:
          - targets: [q14]
            amplitude_sweep: [0.02, 0.5, 25]
      lz:
        experiment: landau_zener
        operating_point: reset
```

- A step name says what the step is for and is unique across the file; `experiment` says what it measures. The same experiment can appear in any number of steps.
- Each experiment registers the analyses it supports and a default one.
- A step writes into the operating point it runs at; its analysis decides which fields. Writing to an experiment section is the explicit exception.
- Steps in a pipeline run in order, each after the one before. A pipeline's `after` names the step, or list of steps, that must have succeeded for that target before its first step runs. Branches are separate pipelines; joins are an `after` list.
- A failure for a target skips the rest of its pipeline and every pipeline after it, for that target only. The run continues for other targets and for unrelated pipelines, so a T1 failure does not cancel the same qubit's LZ pipeline.
- A pipeline is switched off in the file with `enabled: false`, and a run can be launched with only some pipelines (`--pipelines lz`). A dependency on a step that is not run this time is satisfied by the values already in the calibration file.
- `targets` at the top is the default for every pipeline; a pipeline or a step can narrow it. `all_q` and `all_c` keep working.
- A flux sweep names the flux `loop` it sweeps (`z` or `x`).
- A step names the pulses it plays (`pulses`) when it does not use its experiment's default (ADR-0001).
- Parameters resolve in three levels, each overriding the one before: `defaults.<experiment>`, the step's own values, then every `overrides` entry whose `targets` include the target, in file order. `overrides` is a list so that one entry can cover many targets: on 50 qubits where 20 share an override, a per-target mapping would repeat it 20 times. The key replaces today's `overwrites`, which already has this shape.
- `simultaneous` is not in the format while no experiment supports parallel execution (ADR-0004).
- A `check` command that validates the file without touching hardware and prints the graph per target comes after the December delivery, on top of ADR-0005.

## Considered options

- **A flat set of named steps, each with its own `after` defaulting to the previous step** (the first proposal): readable for one linear tree, but turning a branch off, scoping failures and running only the LZ recalibration all need a grouping the format did not have.
- **Flat steps with a `pipeline:` tag on each**: less nesting, but the grouping is no longer visible when reading the file.
- **YAML anchors for shared values** instead of a `defaults` block: no parser change, but anchors are harder for operators to read and write, which goes against the first requirement on this file.
- **Overrides as a mapping by target** (`overrides: {q14: {...}}`): rejected for the scaling reason above.

## Consequences

- Existing trees in the `node_types`/`pipelines` format, and the example in `examples/config_files/`, have to be rewritten.
- `CalibrationGraph` gets real edges from `after`: `add_dependency` exists but nothing calls it today, so the graph runs in insertion order and `dead_targets` stops every later step for a failed target.
- Running one pipeline on its own reads the calibration file for what an unselected pipeline would have produced; if that value has never been calibrated, the step fails and the report says so.
