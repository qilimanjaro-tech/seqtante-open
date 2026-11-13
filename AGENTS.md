# seqtante-open

Seqtante-Open characterises and calibrates qubits and couplers automatically: a calibration tree declares the steps, and a run executes them on a qililab platform. QHC maintains it.

## Navigate by task
- Run entry point and CLI: `src/seqtante_open/calibration_run.py`, `src/seqtante_open/autocalibration.py`.
- Calibration tree parsing and the run graph: `src/seqtante_open/controllers/`.
- Experiments: nodes in `src/seqtante_open/experiments/nodes/`, QPrograms in `src/seqtante_open/experiments/qprogram/`, fits in `src/seqtante_open/experiments/fitting/`, plots in `src/seqtante_open/experiments/plotting/`.
- Run records, data folders and the database: `src/seqtante_open/outputs/`.
- Tests: `tests/`, mirroring `src/`. Fit tests use the harness in `tests/experiments/fitting/` (read `tests/experiments/fitting/README.md`); the offline runcard is `tests/runcards/test_AQPU_runcard.yml`.
- Before writing or reviewing code or tests: read `CODING_STANDARDS.md` and `REVIEW.md`.
- Domain vocabulary: `GLOSSARY.md`; name code, tests and PRs with its terms. Decisions: `docs/adr/`. Tracker and triage: `docs/agents/`.
- EuroQCS December delivery plan and task breakdown: `docs/euroqcs-december-delivery.md`.
- Project knowledge base: TODO(human): link the domain wiki.

## Commands
- **Verify (run before calling work done):** `uv run ruff check . && uv run ruff format --check . && uv run ty check --python-version 3.11 ./src/seqtante_open && uv run ty check --python-version 3.12 ./src/seqtante_open && uv run ty check --python-version 3.13 ./src/seqtante_open && uv run pytest -n auto --dist loadfile tests`
  (the gates of the Code Quality and Tests workflows).
- Focused test: `uv run pytest tests/experiments/fitting/test_two_tone_fit.py`

<important if="you set up the environment, install dependencies or hit a dependency-resolution error">
- Setup: `uv sync --locked --group dev --all-extras`. qililab and qpysequence resolve only from the private `qilimanjaro` CodeArtifact index (`[[tool.uv.index]]` in `pyproject.toml`): authenticate uv to it first. CI does it with OIDC in `.github/workflows/tests.yml`.
</important>

<important if="you change a builder of the committed fit test data">
- Rewrite the committed fit test data: `uv run python -m tests.experiments.fitting.data.make_data`
</important>

## Boundaries
- Validate with synthetic inputs and offline tests; hardware, lab databases and production runcards are in scope only when the task names them.
- Report every gate you could not run and why.
- Keep credentials, runcards with real addresses and measurement data out of the repository and out of agent instructions.

## Critical invariants
- The repository is public: keep internal links, hostnames and customer data out of code, docs, commits and PRs, because anyone can read them.

<important if="you change the calibration-file format or the code that reads or writes it">
- The calibration-file format changes only through an accepted ADR (ADR-0001, ADR-0008), because qili-experiments and qili-runtime read the file and code against it.
</important>

<important if="a change alters measured or fitted values">
- A fix that changes measured values lands in its own PR, never inside a refactor, because it needs hardware sign-off (ADR-0007).
</important>

<important if="you add or change how an experiment measures on hardware">
- Experiments measure through qililab's QProgram and platform APIs (`platform.execute_qprogram`), because qililab owns the hardware contract.
</important>

<important if="you touch the package version, release tags or the publish workflow">
- The version comes from git tags (hatch-vcs): never write one into `pyproject.toml`, and release tags are a bare `MAJOR.MINOR.PATCH`, because `.github/workflows/publish.yml` fails on any other built version.
</important>

<important if="you add, remove or upgrade dependencies or change their sources">
- qililab and qpysequence stay pinned to the `qilimanjaro` index in `[tool.uv.sources]`, because builds must not resolve them from public mirrors.
</important>

<important if="you touch numeric code, units, golden values or tolerances">
- Units: durations are integer ns and frequencies Hz; flux sweeps are V or phi_0, depending on the flux parameter; the QDAC play dwell is µs and the QDAC trigger duration is s. The full convention and its exceptions are in CODING_STANDARDS.md additions.
- Every numeric change ships with the before-and-after evidence REVIEW.md asks for.
</important>

## Branches and PRs
<important if="you create a branch, commit, or open or update a pull request">
Decision 2 (Qilimanjaro Software branch conventions):
- Branch from `main`, named after the ticket: `qhc-####-<topic>`. PR title `[QHC-####] Name`, with the Linear ticket linked.
- Experiments: `exp/<topic>`, no ticket key. Deliveries: `delivery/<project>-<topic>`.
- A PR with a user-facing change adds a towncrier fragment `changes/<PR number>.<type>.md`; the types are in `[tool.towncrier]` in `pyproject.toml`.
</important>

<important if="you work on a EuroQCS December ticket">
- EuroQCS December tickets branch from and target `delivery/seqtante-open-euroqcs-dec` until it merges into `main` (ADR-0007).
</important>

## Agent workflow
- Gardener: TODO(human) (owns the environment-change list after each retro)
- Trust rung: 1 (Pair), proposed: the human drives and reads every diff live. The code drives hardware.
- Scratch output (research, specs, evidence) goes to `.agents/artifacts/`, which is gitignored.

## Agent skills

### Issue tracker

<important if="you read, file or triage issues">
Work is tracked in Linear, team QHC (`QHC-####`); outside users may open GitHub issues. See `docs/agents/issue-tracker.md`.
</important>

### Triage labels

The five default triage labels, unchanged. See `docs/agents/triage-labels.md`.

### Domain docs

<important if="you name code, tests or PRs, or record an architectural decision">
Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
</important>

## Completion
Work is ready for review when the verify command passes and its output is attached to the PR. The verify command runs the gates of the workflows that run on pull requests.
