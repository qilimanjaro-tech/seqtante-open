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

"""Boiler-plate for testing :class:`~seqtante_open.experiments.fitting.FittingClass` subclasses.

Subclass :class:`FittingTestCase`, declare four things, and you inherit a full
test suite for ``__init__``, ``fit`` and ``plot``::

    from tests.experiments.fitting.harness import FittingTestCase


    class TestT1Fit(FittingTestCase):
        FIT_CLASS = T1Fit
        DATA = "t1_fit.h5"
        INIT = {"qubit_idx": 0, "measurement_id": 1}
        EXPECTED = {"optimized_params.thresh.1": pytest.approx(-8.06e-5, rel=0.05)}

The test module must live at the mirror of the fit class's own module. ``T1Fit``
lives in ``src/seqtante_open/experiments/transmons/single_qubit_gates/fit/t1_fit.py``,
so its case lives in ``tests/experiments/transmons/single_qubit_gates/fit/test_t1_fit.py``.
``test_registry.py`` enforces that, and that no fit class goes without a case.

See ``README.md`` in this folder for the copy-paste template.
"""

from __future__ import annotations

import copy
import functools
import importlib
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import h5py
import matplotlib as mpl
import numpy as np
import pytest
import seqtante_open
from qililab.result.result_management import load_results
from ruamel.yaml import YAML

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

mpl.use("Agg")

import matplotlib.pyplot as plt
import plotly.graph_objects as go

DATA_DIR = Path(__file__).resolve().parent / "data"
"""Every ``.h5`` fixture lives here, whatever the depth of the test that uses it."""

RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

TESTS_ROOT = Path(__file__).resolve().parents[2]

SRC_ROOT = Path(seqtante_open.__file__).parent
"""Root of the installed package, the mirror the test tree is checked against."""


def import_test_modules() -> list[ModuleType]:
    """Import every ``test_*.py`` under ``tests/``, so subclass lookups are complete.

    ``FittingTestCase.__subclasses__()`` only sees modules that happen to have been
    imported already, which under ``-k`` filtering or xdist is an arbitrary subset.
    The registry test and the data regeneration script both need all of them.
    """
    modules = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        dotted = ".".join(path.relative_to(TESTS_ROOT.parent).with_suffix("").parts)
        modules.append(importlib.import_module(dotted))
    return modules


def import_source_modules() -> list[ModuleType]:
    """Import every module of the package, so ``FittingClass.__subclasses__()`` is complete.

    A class registers as a subclass when its module is imported and not before, so
    discovering the fit classes means importing all of them. Import errors are left
    to propagate: a module that cannot be imported is a fit class that cannot be
    discovered, and that has to fail loudly rather than quietly shrink the registry.
    """
    modules = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        parts = path.relative_to(SRC_ROOT.parent).with_suffix("").parts
        dotted = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules.append(importlib.import_module(dotted))
    return modules


def all_subclasses(cls: type) -> set[type]:
    """Every subclass of ``cls``, transitively.

    ``__subclasses__`` is direct children only, so a fit class that specialises
    another fit class would otherwise go unseen.
    """
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= all_subclasses(sub)
    return found


@functools.lru_cache(maxsize=1)
def default_platform_before() -> dict:
    """The test runcard as a dict, used as the default ``measurement.platform_before``.

    Fits that need instrument settings (``TwoToneFluxFit``, ``CZConditionalAmpFit``,
    ``FluxoniumSingleToneFluxModel``) call ``build_platform`` on this, so the bus
    aliases a case's loop metadata refers to must exist in the runcard.
    """
    return YAML(typ="safe").load(RUNCARD_PATH.read_text(encoding="utf-8"))


def loop(array: np.ndarray, units: str = "", bus: str = "", parameter: str = "") -> dict:
    """One entry of the ``loops`` mapping that ``qililab.load_results`` yields."""
    return {"array": np.asarray(array), "units": units, "bus": bus, "parameter": parameter}


def write_h5(path: Path, results: np.ndarray, loops: dict[str, dict]) -> Path:
    """Write ``results``/``loops`` in the layout ``qililab.load_results`` reads back.

    The ``loops`` group is created with ``track_order`` so the sweep order a
    generator declares survives the round-trip. Several fits index dimensions
    positionally, and h5py would otherwise hand them back alphabetically.
    """
    with h5py.File(path, "w") as hf:
        group = hf.create_group("loops", track_order=True)
        for name, meta in loops.items():
            dataset = group.create_dataset(name, data=np.asarray(meta["array"]))
            for key in ("units", "bus", "parameter"):
                dataset.attrs[key] = meta.get(key, "")
        hf.create_dataset("results", data=np.asarray(results))
    return path


# --------------------------------------------------------------------------------------
# Fake-data helpers.
#
# Deliberately thin: build the *shape* of a signal with the model functions already on
# FittingClass (exponential, sinus, two_gaussians, ...) and use these to add noise and to
# package the result the way the acquisition writes it.
# --------------------------------------------------------------------------------------


def add_noise(signal: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Additive white gaussian noise, scale given in signal units."""
    return signal + rng.normal(0.0, sigma, size=np.shape(signal))


def as_iq(
    signal: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.0,
    angle: float = 0.7,
    offset: complex = 0j,
) -> np.ndarray:
    """Package a real signal as the ``[..., I, Q]`` pair the fits unpack.

    The signal is placed along a rotated axis in the IQ plane and independent
    noise is added to both quadratures, which is what ``FittingClass.rotate_iq``
    exists to undo. ``angle`` being non-zero is the point: a fit that forgets to
    rotate sees a scaled-down signal and fails.
    """
    s21 = (np.asarray(signal) * np.exp(1j * angle)) + offset
    i = add_noise(np.real(s21), rng, sigma)
    q = add_noise(np.imag(s21), rng, sigma)
    return np.stack([i, q], axis=-1)


# --------------------------------------------------------------------------------------
# Path resolution for EXPECTED
# --------------------------------------------------------------------------------------


def resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dotted ``EXPECTED`` key through attributes, dict keys and indices.

    ``"optimized_params.rot.1"`` resolves to ``obj.optimized_params["rot"][1]``.
    """
    current = obj
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple, np.ndarray)) and part.lstrip("-").isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def is_unpopulated(value: Any) -> bool:
    """Whether a fit output still looks untouched: ``None``, or an all-``None``/empty container."""
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, dict):
        return not value or all(v is None for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return not value or all(v is None for v in value)
    return False


def added_public_methods(fit_class: type) -> set[str]:
    """Public methods ``fit_class`` itself defines, beyond ``fit`` and ``plot``.

    Reads ``vars(fit_class)`` rather than ``dir``, so it catches both brand-new
    methods and re-implementations of a base method. ``DragFit.fit_drag`` shadows
    ``FittingClass.fit_drag``: new code, and it needs its own test.
    """
    names = {
        name
        for name, member in vars(fit_class).items()
        if not name.startswith("_") and callable(getattr(member, "__func__", member))
    }
    return names - {"fit", "plot"}


class FittingTestCase:
    """Inherited test suite for a single fit class.

    Declare:

    ``FIT_CLASS``
        The class under test. Required.
    ``INIT``
        Keyword arguments it is constructed with. Required.
    ``DATA``
        Either the name of an ``.h5`` in ``data/``, or a callable
        ``(rng) -> (results, loops)`` that builds it at test time. Required.
    ``BUILDER``
        The ``(rng) -> (results, loops)`` builder that produced a committed
        ``DATA`` file. Declare it next to the case, so the recipe for a fit
        class's data sits with the fit class's test. Only used by
        ``python -m tests.experiments.fitting.data.make_data``, which rewrites
        every committed file from its builder.
    ``EXPECTED``
        Dotted attribute path mapped to an expected value, compared with ``==``
        (so ``pytest.approx`` works) or, if callable, used as a predicate.
        Required unless the class does not override ``fit``.

    Optional: ``MEASUREMENT`` to override attributes of the fake measurement
    (``platform_before``, ``calibration``, ``data_shape``, ...), ``PLOTS`` to pin
    the exact filenames ``plot`` must produce, ``WAIVED`` to justify an added
    method that needs no test of its own, and ``SEED`` for a ``DATA`` callable.
    """

    FIT_CLASS: FittingClass
    INIT: dict[str, Any]
    DATA: str | Callable[[np.random.Generator], tuple[np.ndarray, dict]]
    BUILDER: Callable[[np.random.Generator], tuple[np.ndarray, dict]] | None = None
    EXPECTED: dict[str, Any] = {}
    MEASUREMENT: dict[str, Any] = {}
    PLOTS: list[str] = []
    WAIVED: dict[str, str] = {}
    SEED: int = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a misconfigured case at import time rather than as a confusing failure."""
        super().__init_subclass__(**kwargs)
        for name in ("DATA", "BUILDER"):
            declared = cls.__dict__.get(name)
            if callable(declared):
                # A plain function assigned in a class body would bind as a method.
                setattr(cls, name, staticmethod(declared))
        for required in ("FIT_CLASS", "INIT", "DATA"):
            if getattr(cls, required, None) is None:
                raise TypeError(f"{cls.__name__} must declare {required}")
        if cls.overrides_fit() and not cls.EXPECTED:
            raise TypeError(
                f"{cls.__name__}: {cls.FIT_CLASS.__name__} overrides fit(), so EXPECTED must say "
                f"what fit() is supposed to produce. A test that asserts nothing is worse than no test."
            )
        stale = set(cls.WAIVED) - added_public_methods(cls.FIT_CLASS)
        if stale:
            raise TypeError(f"{cls.__name__}: WAIVED names methods {sorted(stale)} does not define")

    @classmethod
    def overrides_fit(cls) -> bool:
        """Whether the class under test implements its own ``fit``. ``AllXYModel`` does not."""
        return cls.FIT_CLASS.fit is not FittingClass.fit

    @classmethod
    def regenerate(cls) -> Path | None:
        """Rewrite this case's committed ``.h5`` from its ``BUILDER``.

        Returns the path written, or ``None`` for a case that builds its data at
        test time and has nothing checked in.
        """
        if callable(cls.DATA) or cls.BUILDER is None:
            return None
        results, loops = cls.BUILDER(np.random.default_rng(cls.SEED))
        return write_h5(DATA_DIR / cls.DATA, results, loops)

    # -------------------------------- fixtures --------------------------------

    @pytest.fixture(scope="class")
    def data(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, dict]:
        """The ``(results, loops)`` pair, always round-tripped through a real ``.h5``.

        A committed fixture and a ``DATA`` callable therefore reach the fit class
        by an identical path, and a generated case can be frozen into ``data/``
        later without touching the test.
        """
        if callable(self.DATA):
            results, loops = self.DATA(np.random.default_rng(self.SEED))
            path = tmp_path_factory.mktemp("fitting-data") / f"{self.FIT_CLASS.__name__}.h5"
            write_h5(path, results, loops)
        else:
            path = DATA_DIR / self.DATA
            if not path.exists():
                raise pytest.UsageError(
                    f"{type(self).__name__}: DATA file {path} not found. Generate it with "
                    f"`python -m tests.experiments.fitting.data.make_data`, or set DATA to a callable."
                )
        return load_results(str(path))

    @pytest.fixture(scope="class")
    def environment(self, data: tuple[np.ndarray, dict]) -> Iterator[MagicMock]:
        """Patch the boundaries every fit class touches, for the whole test class.

        ``db_manager`` is a MagicMock whose ``load_calibration_by_id`` returns a
        stand-in measurement. ``store_parameter`` is replaced by a no-op: it is on
        its way out, and untouched it raises ``AttributeError`` (``Outputs.storage_conf``
        is only ever set by ``reset()``), so seven fit classes could not run at all.
        """
        results, loops = data
        attrs = {
            "load_h5": lambda: (results, loops),
            "load_old_h5": lambda: (results, loops),
            "platform_before": default_platform_before(),
            "data_shape": np.shape(results),
            "calibration": None,
            "experiment_name": self.FIT_CLASS.__name__,
            **self.MEASUREMENT,
        }
        measurement = SimpleNamespace(**attrs)

        db_manager = MagicMock(name="db_manager")
        db_manager.load_calibration_by_id.return_value = measurement

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(output_controller, "_db_manager", db_manager)
        monkeypatch.setattr(output_controller, "store_parameter", lambda **_: None)
        yield db_manager
        monkeypatch.undo()

    @pytest.fixture
    def fit_obj(self, environment: MagicMock) -> Any:
        """A freshly constructed, un-fitted instance."""
        return self.FIT_CLASS(**self.INIT)

    @pytest.fixture(scope="class")
    def _fitted_once(self, environment: MagicMock) -> Any:
        """``fit()`` run once per test class, because several of these fits are slow."""
        obj = self.FIT_CLASS(**self.INIT)
        obj.fit()
        return obj

    @pytest.fixture
    def fitted(self, _fitted_once: Any) -> Any:
        """A private copy of the fitted object, so a test may mutate it freely."""
        return copy.deepcopy(_fitted_once)

    @pytest.fixture
    def shows(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """Count calls to both display entry points, matplotlib's and plotly's."""
        counts = {"matplotlib": 0, "plotly": 0}
        monkeypatch.setattr(plt, "show", lambda *a, **k: counts.__setitem__("matplotlib", counts["matplotlib"] + 1))
        monkeypatch.setattr(
            go.Figure, "show", lambda *a, **k: counts.__setitem__("plotly", counts["plotly"] + 1), raising=False
        )
        return counts

    # ------------------------------------------------------------------- tests

    def test_init(self, fit_obj: Any) -> None:
        """Constructs from the data, and leaves every declared output unpopulated."""
        for path in self.EXPECTED:
            name = path.split(".")[0]
            if not hasattr(fit_obj, name):
                continue  # not initialised at all is fine, it certainly is not populated
            value = getattr(fit_obj, name)
            assert is_unpopulated(value), (
                f"{self.FIT_CLASS.__name__}.{name} is already {value!r} before fit() runs, "
                f"so test_fit would pass without fit() doing anything"
            )

    def test_fit(self, fitted: Any) -> None:
        """Every declared ``EXPECTED`` path holds after ``fit()``."""
        if not self.overrides_fit():
            pytest.skip(f"{self.FIT_CLASS.__name__} does not override fit(), nothing to assert")
        for path, expected in self.EXPECTED.items():
            actual = resolve_path(fitted, path)
            if callable(expected):
                assert expected(actual), f"{path} = {actual!r} rejected by the declared predicate"
            else:
                assert actual == expected, f"{path} = {actual!r}, expected {expected!r}"

    def test_plot_saves_image_to_path(self, fitted: Any, tmp_path: Path) -> None:
        """With a path set, ``plot()`` writes an image there and leaves no figure open."""
        fitted.path = str(tmp_path)
        fitted.plot()

        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert written, f"{self.FIT_CLASS.__name__}.plot() wrote nothing to {tmp_path}"
        assert all(p.stat().st_size > 0 for p in written), f"empty image written: {written}"
        if self.PLOTS:
            assert sorted(p.name for p in written) == sorted(self.PLOTS)
        assert not plt.get_fignums(), (
            f"{self.FIT_CLASS.__name__}.plot() left matplotlib figures open. "
            f"Call save_plot() or plt.close() before returning."
        )

    def test_plot_shows_when_no_path(self, fitted: Any, shows: dict[str, int]) -> None:
        """With no path, ``plot()`` displays instead of saving, via matplotlib or plotly."""
        fitted.path = None
        fitted.plot()
        assert sum(shows.values()) == 1, (
            f"expected {self.FIT_CLASS.__name__}.plot() to display exactly once when path is None, got {shows}"
        )

    def test_added_methods_are_covered(self) -> None:
        """Every public method this fit class adds or re-implements has its own test."""
        added = added_public_methods(self.FIT_CLASS)
        missing = {name for name in added if not hasattr(self, f"test_{name}") and name not in self.WAIVED}
        assert not missing, (
            f"{self.FIT_CLASS.__name__} defines {sorted(missing)} beyond fit/plot. "
            f"Add a test_<name> method to {type(self).__name__}, or a WAIVED entry saying why it needs none."
        )

    def test_init_signature_matches_declared_kwargs(self) -> None:
        """``INIT`` is accepted by the constructor, so a renamed argument fails here and loudly."""
        signature = inspect.signature(self.FIT_CLASS.__init__)
        signature.bind(None, **self.INIT)
