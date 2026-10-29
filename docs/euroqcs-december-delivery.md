# EuroQCS December delivery: final decisions and task breakdown

This closes the review of the design page "SeqOpen calibration tree and operating points" (decisions D1–D29, open questions O1–O3) and of the engine ADRs 0004–0007. Every decision now lives in an accepted ADR in `docs/adr/`; this page maps the review to them, fixes the scope for December and splits the work into tasks.

Goal: Seqtante-Open runs the EuroQCS 1q calibration tree unattended. That means recalibrating the sweet spot and the reset point, benchmarking T1, T2 and Landau-Zener, writing operating points to the calibration file so that Landau-Zener can be run through QiliSDK, and producing a PDF report. Implementation window: 5–18 November.

## Decision log

| ID | Final decision | ADR |
| --- | --- | --- |
| D1 | Operating points stored once per target. | 0001 |
| D2 | An experiment section names its operating point and its pulses, and holds only what is specific to it. | 0001 |
| D3 | qili-experiments reads operating points from the target level and pulses by name. **One-way door**: qili-experiments and qili-runtime read the format; Natàlia's approval covers the contract. Task 15 owner: to name. | 0001 |
| D4 | Nested `operating_points → target → name`, with a version key. | 0001 |
| D5 | LO and IF stored separately. When a frequency sweep needs the LO elsewhere, the helper that applies the point moves it; the stored values do not change. | 0001, 0004 |
| D6 | **Changed.** Operating points carry no pulses. Pulses are named waveforms in `Calibration.waveforms`, and steps and sections name the ones they play. | 0001 |
| D7 | Results go to the operating point by default, and to a section only when the tree says so. | 0001 |
| D8 | Rabi at the reset point writes the readout levels. Known caveat: they fold in thermal population and SNR. Accepted until a single-shot readout calibration exists. | 0001 |
| D9 | **Changed.** The EuroQCS tree has two points, `sweet_spot` and `reset`; the qubit is read out at the reset point, so a separate `readout` point would duplicate it. | 0001 |
| D10 | Sweet-spot recalibration corrects the crosstalk flux offset. **Still to be confirmed by the experimentalists** (see Open questions). | 0001 |
| D11 | AC crosstalk and predistortion stored at device level. | 0001 |
| D12 | **Clarified.** The in-memory platform is changed, but the runcard is never saved from a run; temporary settings are restored by the node. | 0001, 0004 |
| D13 | The node applies the operating point by default; an experiment can opt out and call the same helper itself. | 0004 |
| D14 | **Dropped.** QProgram resets what it sets; what `set_parameter` sets goes through a restore context manager. | 0004 |
| D15 | Initial operating points come from the tree. The refusal to start when nothing provides a point moves to after the delivery. | 0001, 0005 |
| D16 | A failure skips, for that target only, the rest of its pipeline and the pipelines after it. | 0003 |
| D17 | An analysis that does not converge or falls below its quality threshold raises an analysis error, and the target fails that step. A Failed verdict is not a step failure. | 0006, 0002 |
| D18 | The reset-point search ports the experimentalists' calibration script (see below). | this page |
| D19 | **Changed.** The tree keeps sweeps, averages, timings and thresholds; pulses come from the calibration file by name. | 0001, 0005 |
| D20, D21 | Thresholds in the tree; benchmarks write nothing back. | 0002 |
| D22 | PDF report contents fixed by the acceptance criteria. It can be turned off at launch, and PDF reporting is an optional install extra. How it is built gets its own ADR. | 0002 |
| D23 | **Changed.** The tree is a set of pipelines, each an ordered set of uniquely named steps. A pipeline can be switched off in the file or left out at launch (`--pipelines`), so the Landau-Zener recalibration can run on its own. | 0003 |
| D24, D25 | Each step names its analysis and writes into the operating point it runs at. | 0003 |
| D26 | **Changed.** Steps in a pipeline run in order; `after` sits on the pipeline and names a step or a list of steps. | 0003 |
| D27 | Three levels of parameters. `overrides` is a list of entries with `targets`, and the key replaces today's `overwrites`. Resolved by the parser, not by YAML anchors. | 0003, 0005 |
| D28 | `targets` narrows per pipeline and per step. **`simultaneous` removed**: parallel execution is not supported on fluxonium yet. | 0003, 0004 |
| D29 | `check` command after the delivery. | 0003, 0005 |
| O1 | **Open** (see Open questions). May an operating point park neighbours' fluxes? The simultaneous-conflict half is out of scope. | 0001 |
| O2 | **Open** (see Open questions). Where the Landau-Zener theory curve comes from. Until it is configured, the verdict is "Not evaluated". | 0002 |
| O3 | **Open** (see Open questions). Which metric measures Landau-Zener agreement with theory. | 0002 |

Other points settled in review:

- Steps name the flux loop they sweep (`loop: z | x`).
- T2 joins the delivery as its own pipeline after the sweet spot: Rabi for a pi amplitude, then T2 (`t2__drag`). T1 stays the saturation version.
- The `Calibration` is deserialised once per run, and its path becomes a CLI argument (ADR-0004).
- **Node** is the code name for the class that runs a step; the glossary scopes it.
- The QHC-1608 operating-point format and the QHC-1596 T1 node restart from this design, reusing parts.
- The redesign lands on `delivery/seqtante-open-euroqcs-dec`, with one approval per PR, and merges into `main` on 16 November, before the full run (ADR-0007). The ADRs and the glossary go to `main` now.

## Reset-point search

Delivering Landau-Zener through QiliSDK at the quality of the experimentalists' calibration script is an acceptance criterion, so `reset_flux` and `rabi_reset` port that script's algorithm instead of a simpler fixed wide sweep:

1. **Find the line.** Run a two-tone sweep at the current reset flux. If the Lorentzian fit is below its minimum R² (0.8 in the script), widen the span and retry (30 MHz, then 60 MHz).
2. **Move the qubit into the window.** If the qubit lies outside the target frequency window, move `flux.z` along the measured slope. The first move is a small probe; after that each move is a secant step, clipped to a maximum step. Wait for the flux to settle, then measure again. Give up past a maximum total shift or a maximum number of moves.
3. **Write the point.** Write the reset point's `flux.z` and drive IF.
4. **Take the readout levels.** Run Rabi at the new reset point. A fit below its minimum R² (0.9) fails the step. Otherwise it writes `level_0` and `level_1` with their uncertainties; which level is |0⟩ is decided by the first point of the sweep.

The window, spans, step sizes, shift limit, move limit, settle time and R² thresholds are tree parameters with the script's values as defaults. Widening on failure stays inside this experiment. It is not a framework feature.

The script runs unattended once launched, but someone edits it before each run and takes over when it stops. Only the search logic is ported; each manual part becomes something Seqtante-Open already provides:

| In the script | In Seqtante-Open |
| --- | --- |
| Qubit and buses hard-coded (`q14`, `readout_q14`, `drive_q14`) | Targets come from the tree, and buses are mapped from target and role in one place (ADR-0001) |
| Runcard and calibration paths, sample, cooldown edited in the file | CLI arguments and the tree header |
| Resonator frequency table and drive LO as constants | The reset point's readout and drive LO/IF in the calibration file |
| Sweep window, target window and limits as constants | Tree parameters: `defaults` per experiment, `overrides` per target |
| Starting flux z is a constant edited by hand every day; the corrected value is written but never read back | The search starts from the reset point's stored `flux.z` and writes the result back, so the next run starts from it. The shift limit applies per run, so slow daily drift is followed, and a large jump still fails on purpose, since it points to a setup problem |
| On failure it zeroes the bias, disconnects and exits; someone reads the log and decides what to do | An analysis error fails the step for that target only; the rest of its pipeline is skipped, other targets and pipelines continue, and the report records the reason, the move log and the plots |
| Levels and flux written to a separate `calibration.yml`, with a warning when a `landau_zener` section shadows them | One calibration file; the LZ section names the `reset` point instead of copying its values, so nothing can shadow them (D1, D2) |

## Scope

**In for December:** ADRs 0001, 0002, 0003, 0004 and 0006, plus the per-target resolution half of 0005. The experiments in scope:

- `two_tone_vs_flux`
- `two_tone` (sweet spot, and the reset-point search)
- `rabi` (readout levels, pi amplitude)
- `t1`
- `t2`
- `landau_zener`

**After December:**

- Pydantic settings and parse-time validation (ADR-0005)
- the `check` command
- the refusal to start when nothing provides an operating point
- porting `single_tone` and `offset_calibration`, then removing the legacy registry
- parallel execution
- single-shot readout calibration

## Acceptance criteria check

| Criterion | Covered by | Risk |
| --- | --- | --- |
| Full tree runs unattended; every experiment implemented in Seqtante-Open | Tasks 6–12 | Tasks 6–12 port code that is still on Qilitools branches (see Code still on Qilitools branches); T2 is first to drop if the critical path slips |
| Reset point recalibrated, LZ via QiliSDK as good as the calibration script | Tasks 9, 10, 15 | Needs the qili-experiments contract change and a hardware comparison against the script on the same qubit and cooldown. Dropping the full reset-point search puts this criterion at risk |
| People can write and read the tree file | ADR-0003, task 3 | Readability judged by operators on the EuroQCS tree (task 13) |
| A new branch such as T2 needs no code changes | ADR-0003 (a pipeline with `after`) | Only if T2's experiment already exists |
| Operating points stored once and read from there | ADR-0001, tasks 2, 15 | qili-experiments contract |
| Recalibrations update the file, benchmarks don't | ADR-0006, task 5 | — |
| A failing step skips only that qubit's later steps | ADR-0003, task 4 | — |
| Thresholds and verdicts in the tree | ADR-0002, task 5 | LZ verdict blocked on O2 and O3: "Not evaluated" until then, and on the drop list |
| PDF report, can be turned off, with timestamp and versions, saved to the configured folder with the timestamp in the name, benchmark value ± σ, plot, verdict, threshold, failed and skipped steps | ADR-0002, task 14 | Report ADR still to write; the report is an optional install extra |
| One full run on the EuroQCS qubits, reviewed by experimentalists | Tasks 16, 17 | Fridge time in the window; qililab at BSC upgraded first |

## Tasks

Sizes are first guesses (S ≈ 1–2 days, M ≈ 3–5, L ≈ 1–2 weeks) and should be re-estimated when the tickets are filed. They add up to well over the 5–18 November window, so the critical path and the drop list below come first.

| # | Task | Size | Depends on | Ticket |
| --- | --- | --- | --- | --- |
| 0 | Fix the `DatabaseManager` file name that collides when the same experiment runs on another loop of the same qubit; own PR ahead of the redesign | S | — | to file |
| 1 | Agree on the pulse naming convention in `Calibration.waveforms` and the section format with the qili-experiments and qili-runtime developers | S | — | to file |
| 2 | Calibration-file model: `operating_points` (version key, `sweet_spot`, `reset`, LO/IF, attenuation, readout levels), named pulses, sections naming point and pulses; read/write helpers; migration from the QHC-1608 format | M | 1 | QHC-1608 (rework) |
| 3 | Tree parser: pipelines of named steps, pipeline `after`, `enabled`, `--pipelines`, target narrowing (`all_q`/`all_c`), `defaults` → step → `overrides`, `loop`, `pulses`, thresholds; exhaustive unit tests; rewrite the example tree | L | — | to file |
| 4 | Graph: edges from `after` via `add_dependency`, per-pipeline failure skipping per target, dependencies on unselected pipelines read from the calibration file | M | 3 | to file |
| 5 | `CalibrationNode` base: `NODE_REGISTRY` with legacy support, run context (one `Calibration`, `--calibration_path`), operating-point helper with LO placement, restore context manager, execute/analyze/apply, analysis error, benchmark results and verdicts | L | 2 | to file |
| 6 | First slice: T1 at the sweet spot as a node, with call-assertion test | M | 4, 5 | QHC-1596 (restart) |
| 7 | `two_tone_vs_flux` as a `FluxSweepNode` (`sweet_spot_flux`, writes the crosstalk offset) | M | 5 | to file |
| 8 | `two_tone` at fixed flux (`lorentzian_peak`, sweet-spot drive IF) | S | 5 | to file |
| 9 | Reset-point search analysis on `two_tone`, porting the calibration script | M | 8 | to file |
| 10 | `rabi` node with `readout_levels` and `pi_amplitude` analyses | M | 5 | to file |
| 11 | `landau_zener` node; verdict "Not evaluated" until O2/O3 | M | 10 | to file |
| 12 | `t2` node | S | 10 | to file |
| 13 | EuroQCS calibration tree file, reviewed by an operator | S | 3, 6–12 | to file |
| 14 | Report ADR, then the report as an optional install extra (`seqtante_open[report]`): launch flag to turn it off, configured folder, timestamped name, versions, operating points before/after, fit plots, benchmark table, failed and skipped steps | ADR S + M | 5 | to file |
| 15 | qili-experiments reads operating points from the target level and pulses by name. Owner: to name | — | 1, 2 | qili-experiments, to create |
| 16 | Full run on the EuroQCS qubits; Landau-Zener through QiliSDK compared against the calibration script; experimentalists review the report | M | 13, 14, 15, 17 | to file |
| 17 | Upgrade qililab at BSC from 0.33.0 to ≥ 0.35.2 | S | — | to file |

Tasks 0, 1, 3 and 17 can start straight away and in parallel.

### Critical path

**1 → 2 → 5 → 6**, with **3 → 4** in parallel and joining at 6. This path lands first: it gives a T1 run at the sweet spot through the new calibration file, tree, graph and node base. Task 5 is the bottleneck, because every node from task 6 on builds on it.

If the path slips, we drop in this order, and the dropped work moves to after December:

1. T2 (task 12, and the `t2` pipeline in the tree).
2. The Landau-Zener verdict: task 11 ships the node and the verdict stays "Not evaluated".
3. The full reset-point search: task 9 runs a fixed wide two-tone sweep with the R² check instead of porting the script's widening and flux moves. This puts the "LZ as good as the calibration script" criterion at risk.

### Code still on Qilitools branches

Seqtante-Open goes ahead before the Qilitools merges, which are all Blocked. A task ports the experiment from its Qilitools branch. A fix made while porting goes back to that branch through the merge ticket, by the ticket's assignee, so the Qilitools merge carries it.

| Task | Experiment on a Qilitools branch | Merge ticket | Brings fixes back |
| --- | --- | --- | --- |
| 6 | `t1__saturation` | QHC-1667 | Ely Farrés |
| 7 | `two_tone_vs_flux__saturation_ramps` | QHC-1666 | Ely Farrés |
| 8, 9 | `two_tone__frequency_pulsed_saturation` | QHC-1665 | Ely Farrés |
| 8, 9 | `two_tone__frequency_pulsed_drag` | QHC-1669 | Juanjo Martinez |
| 10 | `rabi__amplitude_drag_drive` | QHC-1668 | Ely Farrés |
| 11 | `LZ_transition_same_RR_reset__custom_linear_ramp` | QHC-1670 | Juanjo Martinez |
| 12 | `t2__drag` | QHC-1683 | Juanjo Martinez |

## Open questions

None blocks the approval. Each has one owner, a date it is needed by and the task it blocks.

| Question | Owner | Needed by | Blocks |
| --- | --- | --- | --- |
| Pulse naming convention in `Calibration.waveforms` and the section format (task 1) | To name (qili-experiments and qili-runtime developers, QHC) | 4 Nov | Task 2, and so the whole critical path |
| O1: may an operating point park neighbours' fluxes, such as the couplers around q4? | To name (experimentalists) | 5 Nov | Task 2 (operating-point model) |
| D10: does sweet-spot recalibration correct the crosstalk offset rather than store the absolute flux? | To name (experimentalists) | 9 Nov | Task 7 (`sweet_spot_flux` writes the offset) |
| O2: where the Landau-Zener theory curve comes from | To name (experimentalists, theory team) | 12 Nov | Task 11 verdict; "Not evaluated" if unanswered |
| O3: which number measures Landau-Zener agreement with theory | To name (team, theory team) | 12 Nov | Task 11 verdict; "Not evaluated" if unanswered |
| Whether to rename "calibration tree" to something operators find more descriptive | To name (team) | 12 Nov | Task 13 (the EuroQCS tree file operators review) |
