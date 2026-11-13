# seqtante-open coding standards

Shared core v0.2 (Qilimanjaro Software), a draft until the software leads agree it. Copied verbatim, followed by seqtante-open additions. Mechanical rules live in the tools configured in `pyproject.toml` and in CI; this file holds the judgement they cannot make. The review contract is `REVIEW.md`; findings cite the section they apply.

## Tests

Tests are behaviour tests: they drive the public interface at an agreed seam and assert what a caller observes.

- **Seams.** Test at the seams listed in the additions below. A new seam is named in the PR description before tests target it.
- **Observable outcomes.** Assert returned values, raised errors, emitted programs and waveforms, and results read back through the public reader.
- **Independent expected values.** Take expected values from a literal, a worked example, the specification, a published reference or a hardware-validated fixture, never recomputed the way the code computes them.
- **Boundary mocks.** Mock instrument drivers and hardware execution, network, databases, the file system when needed, time and randomness. The package's own classes and functions run for real.
- **Public reach.** When a behaviour is visible only through a `_private` name, add a public accessor or move the test to the seam above.
- **Regressions.** A bug fix starts with a test that fails on the bug.
- **Coverage is evidence.** A line counts as covered when a behaviour test exercises it. Hardware-only code goes in the coverage `omit` list with a reason.

## Code

- **Design.** Prefer deep modules: a small interface over a lot of behaviour. Name things with the terms in `GLOSSARY.md`; a new concept gets a glossary entry first.
- **Units.** Physical quantities carry their unit in the name, a type alias or the docstring (Hz, ns, V, rad, dBm), fixtures included, in the convention the additions state.
- **Suppressions.** Every suppression (`# noqa`, `# type: ignore`, `# ty: ignore`, `# pragma: no cover`, a coverage `omit`) names its rule code and a one-line reason.
- **Comments.** Explain why. A comment that excuses a workaround becomes a type, a lint or a test.
- **Dependencies.** A dependency on another Qilimanjaro package names the released version that carries the needed behaviour.
- **Compatibility.** Public API changes go through a deprecation release unless the PR links an accepted decision to break compatibility.

## Change hygiene

- User-facing changes add a towncrier fragment in `changes/`, in repos that use towncrier.
- Interface or architecture decisions link their decision record; proposals stay labelled as proposals until accepted.

## seqtante-open additions

- **Seams:** fits through `FittingTestCase` in `tests/experiments/fitting/harness.py`, from a stored measurement through `fit()` and `plot()` (`tests/experiments/fitting/README.md`); nodes through the node function, on a real offline `Platform` built from `tests/runcards/test_AQPU_runcard.yml`, asserting the calls the experiment function receives; run records through `Outputs` (`tests/outputs/test_outputs.py`). The calibration tree parser and graph have no tests yet; ADR-0007 asks for exhaustive parser tests. TODO(human): agree the parser and graph seams.
- **Boundary mocks:** `mock_db_manager` and `mock_recorder` in `tests/experiments/conftest.py` (`mock_recorder` autospecs, so a renamed argument fails), and the harness's fake measurement. Fits that run `differential_evolution` draw from the global numpy RNG, so test cases seed it with `SEED`.
- **Units:** durations are integer ns: QProgram pulse and wait durations (`relax_duration` in `src/seqtante_open/experiments/qprogram/two_tone.py`) and the QDAC settle times from `qdac_step_timings` (`src/seqtante_open/experiments/utils/qdac_utils.py`). Frequencies, IFs and LOs are Hz (`src/seqtante_open/experiments/fitting/two_tone_fit.py`), converted to MHz or GHz only for plot labels. Flux sweeps are recorded in V for `Parameter.VOLTAGE` and phi_0 for `Parameter.FLUX` (`src/seqtante_open/experiments/experiment_classes/single_tone.py`). Exceptions at the QDAC edge, in `src/seqtante_open/experiments/qprogram/spectroscopy_vs_flux.py`: `qdac.play` takes `dwell` in µs and `set_trigger` takes `duration` in s. Benchmark thresholds in the calibration tree carry their unit (`50 us`) and the parser converts them (ADR-0003). Pulse amplitudes and gains go to qililab as given; their range is not documented. TODO(human): document the amplitude and gain convention.
- **Coverage:** no `fail_under` in `pyproject.toml`. TODO(human): agree a coverage floor.
