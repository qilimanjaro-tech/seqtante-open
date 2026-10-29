---
status: accepted
---

# Benchmark thresholds live in the calibration tree, and benchmarks write nothing back

The EuroQCS experiment proposal put the Passed/Warning/Failed thresholds "in the calibration file". We put them in the **calibration tree** instead: a threshold is an acceptance policy for a run, not something calibration learns about the chip, and the tree already lets any value be overridden per target. Benchmarks such as T1, T2 and Landau-Zener write nothing to the calibration file. Their value, uncertainty and verdict go into the run's results and into a PDF report generated at the end of every run.

- A Failed verdict is a result, not a step failure: it does not skip anything that comes after the benchmark. A benchmark whose analysis fails (ADR-0006) is a step failure, and the report says why.
- The report can be turned off when the run is launched, for quick debugging runs.
- It is saved to the folder set in the configuration, with the timestamp in the filename, so it never overwrites an older one.
- It shows the timestamp, the Seqtante-Open version and the qililab version; per target, each operating point before and after the run and every fit plot; per benchmark, the value ± uncertainty when there is one, the plot, the verdict and the threshold used; and the steps that failed or were skipped, with the reason.
- PDF reporting is an optional install extra (`seqtante_open[report]`), as agreed on 22 October 2026, so running calibrations does not pull in the PDF dependencies.
- How the report is built (library, a CLI that reads fitted parameters back by calibration id, or the `outputs` service) gets its own ADR before it is implemented. That ADR records the install extra and what a run does when the extra is not installed.

## Open questions

- **Landau-Zener theory curve.** It is not in any repository; it comes from the theory team, or the experimentalists may already have it in a notebook. One candidate: a data file referenced from the calibration tree, with the verdict "Not evaluated" until it is configured. Owner: experimentalists, theory team.
- **Landau-Zener agreement metric.** Candidate: the reduced χ² of the normalised transition probability against the theory curve, with a binomial σ per point (√(p(1−p)/N), N the number of averages) and thresholds on χ²ᵣ in the calibration tree, reporting the largest single-point deviation in σ alongside. Owner: team, theory team.
