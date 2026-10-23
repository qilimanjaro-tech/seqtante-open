---
status: proposed
---

# Benchmark thresholds live in the calibration tree, and benchmarks write nothing back

The EuroQCS experiment proposal put the Passed/Warning/Failed thresholds "in the calibration file". We put them in the **calibration tree** instead: a threshold is an acceptance policy for a run, not something calibration learns about the chip, and the tree already lets any value be overridden per target. Benchmarks such as T1 and Landau-Zener write nothing to the calibration file. Their value, uncertainty and verdict go into the run's results and into a PDF report generated at the end of every run, which also shows each operating point before and after the run, every fit plot, and the steps that failed or were skipped, with the reason.

## Open questions

- **Landau-Zener theory curve.** It is not in any repository; it comes from the theory team, or the experimentalists may already have it in a notebook. How it enters Seqtante-Open is to be determined. One candidate: a data file referenced from the calibration tree, with the verdict "Not evaluated" until it is configured.
- **Landau-Zener agreement metric.** Candidate: the reduced χ² of the normalised transition probability against the theory curve, with a binomial σ per point (√(p(1−p)/N), N the number of averages) and thresholds on χ²ᵣ in the calibration tree, reporting the largest single-point deviation in σ alongside. Others may agree or propose a different metric.
