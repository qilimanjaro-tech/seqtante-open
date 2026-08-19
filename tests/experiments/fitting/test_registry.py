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

"""Every fit class has a test case, and every case sits where it belongs.

Both sides are discovered rather than listed. Subclassing registers a class, so
importing the package yields every :class:`FittingClass` and importing the test
tree yields every :class:`FittingTestCase`; matching the two needs no hand-written
inventory to keep in step. A fit class landing without a case fails here on its
own, which a list could only manage if someone remembered to update it.

Consequence worth knowing: an intermediate abstract fit class, one that exists to
be subclassed rather than used, would be asked for a case like any other. There
are none today. Give it a case, or teach discovery to skip it, when the first one
appears — do not reintroduce a list of names.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from seqtante_open.experiments.fitting.fit_base import FittingClass
from tests.experiments.fitting.harness import (
    SRC_ROOT,
    FittingTestCase,
    all_subclasses,
    import_source_modules,
    import_test_modules,
)


@pytest.fixture(scope="module")
def fit_classes() -> set[type]:
    """Every ``FittingClass`` subclass in the package."""
    import_source_modules()
    return all_subclasses(FittingClass)


@pytest.fixture(scope="module")
def cases() -> dict[type, type[FittingTestCase]]:
    """``{fit class: test case}`` for every case declared anywhere under ``tests/``."""
    import_test_modules()
    return {case.FIT_CLASS: case for case in all_subclasses(FittingTestCase)}


def test_every_fit_class_has_a_test_case(fit_classes, cases) -> None:
    """A fit class with no case is a fit class nothing checks."""
    missing = {cls.__name__ for cls in fit_classes - set(cases)}
    assert not missing, (
        f"no FittingTestCase for {sorted(missing)}. Write one at the mirror of the fit class's "
        f"module, see tests/experiments/fitting/README.md."
    )


def test_every_case_tests_a_fit_class(fit_classes, cases) -> None:
    """A case pointed at something that is not a fit class in this package is a dead case.

    Catches a ``FIT_CLASS`` left behind by a class that was deleted, moved out of
    the package, or that never inherited from ``FittingClass`` to begin with.
    """
    orphaned = {
        case.__name__: case.FIT_CLASS for case in cases.values() if case.FIT_CLASS not in fit_classes
    }
    assert not orphaned, (
        f"these cases test something that is not a FittingClass subclass in the package: {orphaned}"
    )


def test_test_cases_mirror_the_source_layout(cases) -> None:
    """A case lives at the mirror of its fit class's module, so tests stay findable."""
    misplaced = {}
    for fit_class, case in cases.items():
        source = Path(importlib.import_module(fit_class.__module__).__file__)
        expected = Path("tests") / source.parent.relative_to(SRC_ROOT) / f"test_{source.stem}.py"
        actual = Path(importlib.import_module(case.__module__).__file__)
        if actual.parts[-len(expected.parts) :] != expected.parts:
            misplaced[case.__name__] = f"expected .../{expected}, found {actual}"

    assert not misplaced, f"test cases in the wrong place: {misplaced}"
