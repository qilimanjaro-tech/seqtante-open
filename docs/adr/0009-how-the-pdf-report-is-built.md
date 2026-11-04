---
status: accepted
---

# The PDF report is built in-process from the run's results, behind an optional install extra

ADR-0002 fixed what the report contains and deferred how it is built. Three shapes were on the table: a library the run calls at the end, a separate CLI that rebuilds the report from the database by run id, or folding it into the `outputs` service. By the end of a run the graph and its nodes already hold everything the report needs — each target's operating points before and after, every fit plot, each benchmark's value ± uncertainty, verdict and threshold, the steps that failed or were skipped with the reason, and the reset-point move log (ADR-0001, ADR-0006) — alongside the timestamp, the Seqtante-Open and qililab versions, and the run id the `outputs` service opened. We decided to **build the report in-process, at the end of the run, from those in-memory results**, in a module separate from the always-imported `outputs` service, pulled in only when an **optional install extra `seqtante_open[report]`** is present, and turned off by a launch flag. The report reads nothing back from the database.

- **A separate module, lazily imported.** The report code lives in its own module (for example `seqtante_open.report`), which `outputs` never imports. `CalibrationRun` imports it only in `_end_calibration`, after `end_run` and after the calibration file is saved, so a report failure can never cost a run its results. The `outputs` service stays free of PDF dependencies because it is imported on every run; the report is not.
- **The install extra.** `seqtante_open[report]` carries the PDF-assembly dependency, as agreed on 22 October 2026, so running calibrations does not pull in the report dependencies. The plotting stack (`plotly`, `kaleido`) is already a core dependency, so the extra is small. The exact assembly library is left to the implementation (task 14, second half, QHC-1810).
- **When the extra is not installed.** The lazy import fails, the run logs a warning that names `pip install "seqtante_open[report]"`, and finishes normally: the database entry and the calibration file are already written. A missing report is never a failed run.
- **The launch flag.** The report is on by default, and `--no-report` turns it off for quick debugging runs (ADR-0002). With `--no-report` the module is never imported, so that run does not need the extra installed at all.
- **Where the data comes from.** The report consumes the run's in-memory results, the same ones the nodes hand to the database and the verdicts ADR-0006 puts on the run. It does not query the database or read fitted parameters back, so it does **not** depend on QHC-1791 (unified `add_fitting`) or on the fitted-parameter names in QHC-1497. It records the run id so the PDF cross-references the database, but never reads values from it.
- **Where it is saved.** To the configured folder, the per-run `data/<date>/run_<timestamp>/` the `outputs` service already created, with the timestamp in the filename (ADR-0002), so it never overwrites an older one.

## Considered options

- **A CLI that rebuilds the report from the database by run id.** Rejected for the December delivery. It couples the report to the database schema and makes it depend on QHC-1791 and QHC-1497, which the delivery plan keeps off the report's critical path (QHC-1810). It also cannot see what the database does not store in a stable, named form — skipped steps, the reset-point move log, the plot artifacts — and needs every value persisted and named before it can run at all. Its one advantage, regenerating a report from a past run without re-measuring, is a later need, not a December one, and building in-process does not preclude adding such a command on top of the same report module when it is wanted.
- **Folding the report into the `outputs` service.** Rejected. `outputs` owns the per-run folder, the database run entry and the logger, and is imported on every run; putting the PDF code there would pull the report dependencies into every calibration and defeat the optional extra. The report reuses what `outputs` already exposes (the storage path and the run id) from the outside instead.

## Consequences

- The report is decoupled from the database work: task 14's second half (QHC-1810) can start as soon as the run exposes its results, without waiting on QHC-1791 or QHC-1497.
- The run must collect its results into an in-memory record the report can read. ADR-0006 already routes benchmark values, uncertainties and verdicts to the run's results; the operating points before and after, the fit plots, the failed and skipped steps with their reason, and the reset-point move log join the same record.
- The entrypoint grows a CLI surface beyond `--platform_path` and `--config_path`, starting with `--no-report`.
- A report can only be produced while the run's process is alive. Regenerating one from a past run is a separate feature, to be designed as the CLI option above if it is ever needed.
- Both paths need testing: a run with the extra installed, and a run without it, where the warning fires and the run still completes.
