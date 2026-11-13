# seqtante-open review contract

Shared core v0.2 (Qilimanjaro Software), a draft until the software leads agree it. Copied verbatim, followed by seqtante-open additions. Applies to humans and agents. AI review is advisory: a qualified human reads every line of the same revision before merge.

## Findings
- Each finding gives `path:line` and quotes the line, then the trigger, the consequence, the evidence and a concrete next step. No citation, no finding.
- Separate confirmed defects from questions and missing evidence. Recheck earlier findings on the current revision and state the inspected scope.
- Review on two axes: standards (`CODING_STANDARDS.md`, citing the section) and spec (the linked ticket, PRD or TRD: missing, extra, wrong; quote the spec line).
- Green CI is evidence, not a verdict: say what it covers and what it misses.

## Severity
Lead with the highest.
- **blocker**: wrong results, data loss, a security hole, or a broken invariant from AGENTS.md. Fix before merge.
- **major**: a likely bug, or a design problem the next change will trip over. Fix, or justify in the PR.
- **minor**: worth fixing, not blocking.
- **nit**: taste. At most 3 per review; drop the rest.

## seqtante-open additions
- **Decisions.** Check the change against the ADRs in `docs/adr/` that touch its area, and quote the ADR line it follows or departs from. A departure links a new or superseding ADR.
- **Calibration file.** A change to what a run reads from or writes to the calibration file (operating points, pulses, experiment sections, crosstalk) follows ADR-0001 and ADR-0008; a format change is a blocker without an accepted ADR, because other repositories read the file.
- **Writes.** Only a recalibration step writes to the calibration file; a benchmark writes nothing back (ADR-0002, `GLOSSARY.md`). For node classes, hardware stays in `execute_node`, analysis is read-only and only `apply` writes (ADR-0006). TODO(human): confirm.
- **Parameter mapping.** A change to how tree or runcard values reach an experiment comes with a test that asserts the arguments the hardware call receives, using `mock_recorder` from `tests/experiments/conftest.py` (ADR-0007).
- **Measured values.** A change that alters measured or fitted values is its own PR, not part of a refactor, and the PR names the hardware test it needs (ADR-0007).
- **Public repository.** No internal links, hostnames, customer data or measurement data in the diff, the PR text or the commit messages.
- **Units.** Values follow the convention in CODING_STANDARDS.md additions; check every conversion where a value leaves this repo (compiler, driver, instrument).
- **Parity.** Any numeric change ships before-and-after output, with each difference explained: from `tests/experiments/fitting/data/*.h5` where it reaches the changed code, otherwise from a synthetic input that does (for a QProgram builder, the emitted program built on the platform from `tests/runcards/test_AQPU_runcard.yml`).
