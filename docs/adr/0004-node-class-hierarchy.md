---
status: proposed
---

# Experiments are node classes, and the node owns what every experiment repeats

Each step of the calibration tree (ADR-0003) runs an experiment. Today an experiment is a function `(platform, platform_path, parameters)` looked up in `experiment_functions_dict`, and all four repeat the same frame: deserialise the `Calibration` from `calibration_path`, check it has a `CrosstalkMatrix`, `set_crosstalk`, `set_flux_to_zero`, then a `try/finally` that zeroes the bias and saves the runcard or the calibration file. Inside that frame sit two shapes: a per-target sweep at fixed flux (`single_tone`, `two_tone`) and a per-target, per-loop flux sweep with the coupler-to-readout-qubit mapping (`offset_calibration`, `two_tone_vs_flux`), each pair duplicating its `_run_experiment`. ADR-0001 moves applying operating points and saving results into `CalibrationNode`; with functions, that becomes one more copy per experiment, and Rabi, T1, Landau-Zener and the readout-point search would each add another. We decided that `CalibrationNode` becomes the abstract base class every experiment subclasses, exposing `execute_node(targets)`, so the parser builds one object per step.

- Two mid-level spines hold the shared code: `FixedFluxNode` for sweeps at the step's operating point and `FluxSweepNode` for per-loop flux sweeps. Names are provisional.
- `experiment_functions_dict` becomes `NODE_REGISTRY`, filled by `__init_subclass__` from each class's `experiment` name. A class also declares the analyses it supports and its default one (ADR-0003, ADR-0006), and its settings (ADR-0005).
- What varies between experiments is declared as class variables and handled by `CalibrationNode.run`:
  - `applies_operating_point` (default true): `run` places the targets at the step's operating point before `execute_node`. An experiment with edge cases, such as an LO that has to move outside a frequency sweep or an order of operations that avoids heating the fridge, sets it to false and calls the same base helper from `execute_node`.
  - `parallel`: `forbidden`, `allowed` or `required`. `run` honours a step's `simultaneous` only when `allowed`, and runs all targets together when `required`. Every current experiment is `forbidden`: the base for parallel execution is there, but it is not supported yet.
- A platform setting an experiment changes temporarily (a detuned IF, a parked flux) goes through a base-class context manager that restores it in `finally`. QProgram resets what it sets itself, not what is set through `platform.set_parameter`.
- A variation point becomes a hook only if at least three subclasses use it non-trivially; otherwise the subclass overrides `execute_node` and calls base helpers.
- Scheduling and failure propagation live on `CalibrationGraph`: a node reports which targets failed, and the graph skips, for those targets only, the steps that come `after` the failed one (ADR-0003).

## Considered options

- **Composition**, `CalibrationNode` holding an `Experiment`: leaves the graph untouched, but the parser builds two objects per step and the second earns nothing, since no experiment changes how targets are scheduled, only how they are executed.
- **Template method with a hook per variation point**: what the experiments share is the shape, not the content, so one hook per variation turns each subclass into many small overrides with control flow that is hard to follow.
- **Decorator on the execute function** for operating-point handling and parallel support: fits a function registry, but with node classes the same information sits more naturally as class variables next to the experiment's other declarations.

## Consequences

- The four experiments are ported to classes, and Rabi, T1, Landau-Zener and the readout-point search are written as classes directly, not as functions first.
- Adding an experiment is one class; auto-registration removes the "added the function, forgot the registry entry" failure.
- Today `dead_targets` stops every later step for a failed target, so a T1 failure cancels the same qubit's Landau-Zener branch. `CalibrationGraph.add_dependency` exists but nothing calls it, so the graph has no edges and runs in insertion order; `after` needs it wired up.
- Nested settings classes do not inherit automatically, so `__init_subclass__` needs a guard (ADR-0005).
- Nothing covers `CalibrationNode`, `CalibrationParser` or `CalibrationGraph` today; ADR-0007 sets the test net the port lands on.
