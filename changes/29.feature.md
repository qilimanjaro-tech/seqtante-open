Benchmark `thresholds` in the calibration tree carry their unit (ADR-0002), and the parser converts them to the unit the benchmark reports.

- `thresholds` resolve per target like any other parameter, from `defaults.<experiment>`, the step and its `overrides`. An override replaces the whole `thresholds`, so it sets both `passed` and `warning`.
- Each resolved value is a `Thresholds(passed, warning, unit, direction)`. A benchmark step without `thresholds` is accepted.
- Time thresholds take `ns`, `us` (or `µs`) and `ms`. A dimensionless benchmark such as Landau-Zener takes a plain number.
- These raise a `TreeParseError`: `thresholds` on an experiment that is not a benchmark, a missing or extra key, a missing or unknown unit, a unit on a dimensionless benchmark, `passed` equal to `warning`, and `passed` and `warning` in the wrong order for the benchmark's direction.

## **Example**
```yml
defaults:
  t1: {thresholds: {passed: 50 us, warning: 20 us}}
pipelines:
  t1:
    steps:
      t1_sweet_spot:
        experiment: t1
        operating_point: sweet_spot
        overrides:
          - targets: [q14]
            thresholds: {passed: 30 us, warning: 10 us}
```
