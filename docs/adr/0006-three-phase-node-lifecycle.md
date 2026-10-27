---
status: proposed
---

# A node runs in three phases: execute, analyze, apply

Each experiment function today measures, fits, plots, writes fitted values to the platform or the crosstalk matrix, and saves the runcard or the calibration file, all in one body. The saving sits in a `finally`, so a failure on the third qubit still saves what the first two wrote, and old data cannot be re-fitted without re-measuring. The pieces need different things: an analysis (a `FittingClass` in today's code) loads its measurement from the database by id and never touches the platform, while writing results needs the platform and the calibration file. ADR-0003 also lets one measurement feed different analyses (Rabi for readout levels or for a pi amplitude), and ADR-0002 makes benchmarks write nothing back. We decided that every node implements three phases, which `CalibrationNode.run` calls in order:

- `execute_node(targets)`: hardware only. It returns measurement ids and restores any instrument state it changed (ADR-0004).
- `analyze`: runs the step's `analysis`, or the experiment's default, on the measurement and produces results and plots. It is read-only.
- `apply`: writes the results into the step's operating point, into an experiment section when the tree says so, or into the device-level crosstalk offsets for sweet-spot recalibration (ADR-0001), then saves the calibration file. It runs only for targets whose `analyze` succeeded. A benchmark's `apply` writes nothing; its value, uncertainty and verdict go to the run's results and the report (ADR-0002).

## Considered options

- **One `execute` doing everything**: no change, and readable top to bottom, but the analysis can be neither re-run on its own nor chosen per step, and partial results are saved on failure.
- **Two phases, with `analyze(apply=True)`**: fewer methods, but every subclass has to honour the flag, and one that forgets turns a read-only re-analysis into a write to the calibration file.

## Consequences

- A recalibration and a benchmark differ only in what `apply` does.
- Re-analysing a stored measurement goes straight through the analysis and cannot overwrite calibrated values.
- The `save_platform` calls in `single_tone` and `two_tone` go away: analysis results land in the calibration file, and the runcard keeps only the baseline (ADR-0001).
- Three methods per node instead of one. Inside a run they still execute back to back, because the next step depends on applied values; the split buys structure and standalone analysis, not different in-run behaviour.
- What counts as a failed analysis (a fit that does not converge, or falls below its quality threshold) is left to its own ADR.
