# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rewrite every committed ``.h5`` in this folder from its builder.

    python -m tests.experiments.fitting.data.make_data

The builders themselves live next to the test case that uses them, in the
mirrored test module for the fit class. This script only finds them: it imports
every test module, then asks each :class:`FittingTestCase` with a committed
``DATA`` file and a ``BUILDER`` to regenerate itself.

A case whose ``DATA`` is a callable builds its data at test time and writes
nothing here. A committed file with no ``BUILDER`` cannot be regenerated, and is
reported so it does not quietly become an orphan.
"""

from __future__ import annotations

from loguru import logger

from tests.experiments.fitting.harness import DATA_DIR, FittingTestCase, import_test_modules


def all_cases() -> list[type[FittingTestCase]]:
    """Every declared test case, in a stable order."""
    import_test_modules()
    return sorted(FittingTestCase.__subclasses__(), key=lambda case: case.__name__)


def main() -> None:
    """Regenerate committed data files, and report anything that could not be."""
    written = set()
    for case in all_cases():
        path = case.regenerate()
        if path is not None:
            written.add(path.name)
            logger.info(f"wrote {path.relative_to(DATA_DIR.parent)}")
        elif not callable(case.DATA):
            logger.info(f"SKIPPED {case.DATA}: {case.__name__} declares no BUILDER")

    orphans = {path.name for path in DATA_DIR.glob("*.h5")} - written
    for name in sorted(orphans):
        logger.info(f"ORPHAN {name}: no test case regenerates this file")


if __name__ == "__main__":
    main()
