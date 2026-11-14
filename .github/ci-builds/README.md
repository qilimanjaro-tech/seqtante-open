# CI source-build policy

Tests, Code Quality and the publish build use uv 0.13.8 with the experimental
`build-dependency-hashes` feature. The project requires uv >=0.13.8 so older
clients cannot silently ignore the hashed build configuration.

CI denies third-party source builds except `qblox-instruments==1.0.3`. Its
published artifact is source-only. Before authentication or installation,
`check_policy.py` verifies its exact registry, source URL, version and archive
hash against the reviewed lock. Another source-only or non-registry dependency
requires a policy review. The root editable project is the intentional
first-party build.

`pyproject.toml` contains the complete pinned, wheel-hashed build dependency
closure for Hatchling, hatch-vcs, its editable build helper and Qblox's setuptools
backend. `require-build-hashes` also requires hashes for transitive and dynamically
requested build dependencies. A newly requested dependency fails until reviewed.
These constraints are recorded in the lock without changing application package
versions, artifact hashes or private index mappings.

The restricted CI installs and publish build use fresh caches because uv does not recheck build
dependency hashes in already-built wheels or installed environments. The Tests
matrix checks Python 3.11, 3.12 and 3.13 on PRs as well as main, so every supported
version verifies a clean install. First-party builds remain enabled.

The publish build still checks out its selected source ref, with full Git history
and tags. Build constraints belong to that source ref. An older ref that predates
the hash configuration retains its historical build configuration, while the
workflow still prohibits fresh third-party source builds. No fallback to another
source ref or version scheme is introduced.

To verify the policy without private dependencies or services:

```sh
uv run --no-project python .github/ci-builds/check_policy.py
uv run --no-project python -m unittest discover -s .github/ci-builds -v
```

The regressions build harmless fixtures using a loopback index. They prove
complete hash graphs succeed, missing transitive or changed hashes fail before
the backend runs, unlisted source builds are denied, and changed exception
versions/hashes or local sources fail the guard. They never access instruments,
databases, cloud credentials or upload endpoints.

When changing a build backend or the Qblox exception, resolve its full build
dependency closure from approved wheels, including requirements requested by
the backend during editable/metadata builds. Review every exact version and
hash, update the project build constraints and corresponding lock manifest, and
repeat clean-cache builds and the supported Python matrix. Keep unrelated
application dependency updates in their own changes. A runtime lock alone does
not lock isolated build dependencies.
