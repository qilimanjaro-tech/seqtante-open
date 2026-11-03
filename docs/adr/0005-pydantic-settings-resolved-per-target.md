---
status: accepted
---

# Step parameters are resolved per target, and become Pydantic settings validated before the run after the December delivery

A step's parameters come from several places: the runcard (`platform.get_parameter` for IFs and LOs), module-level `_DEFAULTS` dicts read with `.get(key, _DEFAULTS[key])`, the experiment's block in the tree, and the per-target entries of today's `overwrites` key, which ADR-0003 renames `overrides` without changing its list shape. Nothing declares the schema. A mistyped optional key silently falls back to its default; a missing required key (`target_params["averages"]`) raises a `KeyError` only when its node runs, after earlier steps have already used fridge time; and `parameters` mixes parameter names with target keys in one dict, read back as `{**parameters, **parameters[qubit]}`. Names drift between experiments too: `offset_calibration` reads `readout_amp` and `duration`, `single_tone` reads `readout_amplitude` and `readout_duration`. ADR-0003 adds a `defaults` layer. The layers cannot be validated one at a time, because a step's own values may be incomplete until an override fills them, and an override sets only some fields. We decided that the parser resolves the layers into one set of parameters **per target**, and that each node class later declares a strict Pydantic `Settings` model that this per-target view is validated against at parse time, before the platform connects.

## For the December delivery

- The parser resolves `defaults.<experiment>`, then the step's own values, then every `overrides` entry whose `targets` include the target, in file order, into a plain `dict[Target, dict]` per step. The parser handles `defaults` itself, so YAML anchors are not needed.
- The tree holds **user-authored parameters only**: sweeps, averages, timings, analysis quality thresholds and benchmark thresholds (ADR-0002). Operating points and pulses come from the calibration file (ADR-0001).
- A step naming an unknown experiment, analysis, pipeline or step is an error, since the graph cannot be built without them.
- `data_folder` is computed by the node and recorded in provenance, but cannot be set from the tree.

## After the delivery

- Two models: a lenient `StepConfig` checks the raw YAML structure of a step and resolves into strict per-target `Settings` (every field required or defaulted), held as `self.settings: dict[Target, Settings]`.
- `Settings` is nested in its node class and uses `extra="forbid"` with did-you-mean suggestions.
- Errors are accumulated across every step and target and raised once, with file paths such as `pipelines.t1.steps.t1_sweet_spot.wait_sweep`.
- A `defaults` block for an experiment no step uses is a warning, since trees are shared across branches where an experiment may not exist yet.
- The `check` command (ADR-0003) and the refusal to start when a step needs an operating point that nothing provides are built on this validation.

## Considered options

- **Pydantic settings in the December delivery**: safer runs, but it needs the parameters of every experiment declared and exposed first, which does not fit the implementation window next to five new experiments. Unit tests on the parser are the minimum in the meantime (ADR-0007).
- **One model mirroring the raw YAML layers**: the override sub-model needs every field optional, so completeness becomes unverifiable and a target that never receives a required sweep passes validation. The resolved per-target view is the only place a required-field check means anything.
- **Dataclasses**: no coercion (`2e6` to float, `[0, 2000, 31]` to a sweep), no aggregated multi-error report and no `extra="forbid"`. qililab is moving towards Pydantic, and it is already a transitive dependency.
- **YAML anchors for defaults**: no parser change, but keys that only hold an anchor are rejected under `extra="forbid"` unless they get a reserved prefix, and anchors are harder for operators to read than a `defaults` block.

## Consequences

- Until the settings land, a mistyped optional key still falls back to its default; the per-node call-assertion tests of ADR-0007 are what catch it.
- With the settings, `_DEFAULTS` dicts become field defaults, so a default is declared in exactly one place, and a `Sweep` type accepting `{start, stop, points}` and `{center, span, points}` replaces the `np.linspace(*…)` written in every node.
- With the settings, a bad parameter aborts the whole run instead of skipping a target. Accepted: it fails in seconds, before the platform connects, instead of hours into a run. Two policies then coexist: abort at parse time, skip the target at run time.
