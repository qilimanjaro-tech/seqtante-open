---
status: proposed
---

# Step parameters are Pydantic settings, resolved per target and validated before the run starts

A step's parameters come from several places: the runcard (`platform.get_parameter` for IFs and LOs), module-level `_DEFAULTS` dicts read with `.get(key, _DEFAULTS[key])`, the experiment's block in the tree, and the per-target `overrides` entries (`overwrites` in today's parser). Nothing declares the schema. A mistyped optional key silently falls back to its default; a missing required key (`target_params["averages"]`) raises a `KeyError` only when its node runs, after earlier steps have already used fridge time; and `parameters` mixes parameter names with target keys in one dict, read back as `{**parameters, **parameters[qubit]}`. Names drift between experiments too: `offset_calibration` reads `readout_amp` and `duration`, `single_tone` reads `readout_amplitude` and `readout_duration`. ADR-0003 adds a `defaults` layer and keeps overrides as a list. The layers cannot be validated one at a time, because a step's own values may be incomplete until an override fills them, and an override sets only some fields. We decided that each node class declares a strict Pydantic `Settings` model, that the parser resolves the layers into one `Settings` per target, and that validation runs at parse time, before the platform connects.

- Two models: a lenient `StepConfig` checks the raw YAML structure of a step and resolves into strict per-target `Settings` (every field required or defaulted), held as `self.settings: dict[Target, Settings]`.
- Resolution follows ADR-0003: `defaults.<experiment>`, then the step's own values, then every `overrides` entry whose `targets` include the target, in file order. The parser handles `defaults` itself, so YAML anchors are not needed.
- `Settings` is nested in its node class, uses `extra="forbid"` with did-you-mean suggestions, and holds **user-authored tree parameters only**: sweeps, averages and thresholds (ADR-0002). Pulse settings and operating points come from the calibration file (ADR-0001).
- Machine-supplied values sit in a small run context: the platform, the runcard path and the `Calibration`, deserialised once by the run instead of once per experiment. Its path becomes a CLI argument next to `--platform_path` and `--config_path`; the tree may still set it.
- Errors are accumulated across every step and target and raised once, with file paths such as `steps.t1_sweet_spot.wait_sweep`.
- A step naming an unknown experiment or analysis is an error. A `defaults` block for an experiment no step uses is a warning, since trees are shared across branches where an experiment may not exist yet.
- `data_folder` is computed by the node and recorded in provenance, but cannot be set from the tree.

## Considered options

- **One model mirroring the raw YAML layers**: the override sub-model needs every field optional, so completeness becomes unverifiable and a target that never receives a required sweep passes validation. The resolved per-target view is the only place a required-field check means anything.
- **Dataclasses**: no coercion (`2e6` to float, `[0, 2000, 31]` to a sweep), no aggregated multi-error report and no `extra="forbid"`. qililab is moving towards Pydantic, and it is already a transitive dependency.
- **YAML anchors for defaults**: no parser change, but keys that only hold an anchor are rejected under `extra="forbid"` unless they get a reserved prefix, and anchors are harder for operators to read than a `defaults` block.

## Consequences

- `_DEFAULTS` dicts become field defaults, so a default is declared in exactly one place.
- A `Sweep` type accepting `{start, stop, points}` and `{center, span, points}` replaces the `np.linspace(*…)` written in every node.
- The `check` command of ADR-0003 is a follow-up built on this validation, adding the graph-level checks (`after` names, cycles, operating points nothing provides).
- A bad parameter aborts the whole run instead of skipping a target. Accepted: it fails in seconds, before the platform connects, instead of hours into a run.
- Until failure handling gets its own ADR, two policies coexist: abort at parse time, skip the target at run time.
- Nested `Settings` classes do not inherit automatically, so subclasses need an `__init_subclass__` guard (ADR-0004).
