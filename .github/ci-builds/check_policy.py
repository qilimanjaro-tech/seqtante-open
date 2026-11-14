# Copyright 2026 Qilimanjaro Quantum Tech
# ruff: file-ignore[implicit-namespace-package] -- Standalone CI command.
"""Reject changes to the exact third-party source-build exception."""

import sys
import tomllib
from pathlib import Path

QBLOX_SOURCE = {
    "url": "https://files.pythonhosted.org/packages/3f/c5/92860233ea0d7f14df45df10633682f40d86ee03d2667b423c8f3c574307/qblox_instruments-1.0.3.tar.gz",
    "hash": "sha256:798528d69253c64f6e70b283c623a7cb6b5265806320bd90cec424a9f384f5ff",
}


def check_lock(path: Path) -> None:
    """Validate the source exception before CI installs or builds anything."""
    lock = tomllib.loads(path.read_text())
    exceptions = []
    for package in lock["package"]:
        source = package["source"]
        if package["name"] == "seqtante-open" and source == {"editable": "."}:
            continue
        if "registry" not in source:
            raise SystemExit(f"Unreviewed non-registry source: {package['name']}")
        if package["name"] == "qblox-instruments":
            exceptions.append(package)
        elif not package.get("wheels"):
            raise SystemExit(f"Unreviewed source-only package: {package['name']}")
    if len(exceptions) != 1:
        raise SystemExit("Expected exactly one reviewed Qblox source exception")
    qblox = exceptions[0]
    source = qblox.get("sdist", {})
    if (
        qblox["version"] != "1.0.3"
        or qblox["source"] != {"registry": "https://pypi.org/simple"}
        or any(source.get(key) != value for key, value in QBLOX_SOURCE.items())
    ):
        raise SystemExit("Qblox version, source or hash changed, review the CI build exception first")


if __name__ == "__main__":
    check_lock(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("uv.lock"))
