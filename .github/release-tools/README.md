# Release tools

This non-package project locks Twine and its full dependency graph separately from
Seqtante-open. It requires uv 0.13.8 and Python 3.13. No third-party source builds
are allowed when installing it.

The publish job checks out these files at `github.workflow_sha`, the immutable
revision of the running workflow. The artifact build keeps its independently
selected ref. This allows a manual run to build an older tag that predates these
files without falling back to floating tool versions.

Install once, then invoke the same environment for check and upload. Calling the
environment's Python directly avoids another sync or dependency resolution.
The check still runs on dry runs. The upload-token and endpoint requests, and the
upload itself, remain behind the existing dry-run condition.

To verify locally, without AWS or a real upload:

```sh
uv sync --project .github/release-tools --locked --no-build --no-default-groups --group test
.github/release-tools/.venv/bin/python -m unittest discover -s .github/release-tools -v
```

The tests use a loopback fixture index, fresh temporary caches and stubbed
publication commands. Real Twine checks only local fixture distributions.
The Release tools workflow runs them independently of private application
dependencies, with the same uv and Python versions as the publishing environment.

To upgrade tools, update the exact requirement, regenerate the lock with the
selected uv version and `uv lock --project .github/release-tools --no-build`, then
review the entire transitive diff. Keep the uv version in the manifest and both
workflow setup steps aligned. Re-run clean-cache Linux verification and the
regression suite before accepting the new lock. Never add a source-build
exception to this publishing environment.

This environment does not constrain the application's CI installs or its
first-party build backend dependencies. Those policies require their own
compatibility checks.
