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

"""Tests for how ``FittingClass`` persists fitted parameters and plot paths.

``get_db_manager`` is patched to return a ``FakeDBManager``, which records every
``add_fitting`` call, so the tests assert on what would reach the database. The
fits are bare ``FittingClass`` instances built without ``__init__``, which would
load a measurement from the database. ``go.Figure.write_image`` is replaced by a
plain file write, so no image renderer runs.
"""

import os
import warnings

import numpy as np
import plotly.graph_objects as go
import pytest

import seqtante_open.experiments.fitting.fit_base as fit_base_module
import seqtante_open.outputs.outputs as outputs_module
from seqtante_open.experiments.fitting.fit_base import RENDER_FORMAT, FittingClass
from seqtante_open.outputs import FittedParameter, output_controller


class FakeDBManager:
    """Records the arguments passed to ``add_fitting`` instead of touching a real database."""

    def __init__(self, database_schema="autocal"):
        self.database_schema = database_schema
        self.fitting_calls = []

    def add_fitting(self, id, path=None, parameters=None):
        self.fitting_calls.append({"id": id, "path": path, "parameters": parameters})


@pytest.fixture(autouse=True)
def reset_output_controller():
    """Give every test the unconfigured (freshly initialised) singleton, and leave it that way."""
    output_controller._initialize()
    yield
    output_controller._initialize()


@pytest.fixture(autouse=True)
def write_image(monkeypatch):
    monkeypatch.setattr(go.Figure, "write_image", lambda self, path, **kwargs: open(path, "wb").close())


@pytest.fixture
def fake_db(monkeypatch):
    """An initialised controller whose ``db_manager`` records calls instead of writing.

    ``_initialized`` is set by hand: the ``db_manager`` property is guarded and
    raises without it. ``reset()`` picks the fake up through ``get_db_manager``.
    """
    fake = FakeDBManager()
    monkeypatch.setattr(outputs_module, "get_db_manager", lambda *args, **kwargs: fake)
    monkeypatch.setattr(output_controller, "_db_manager", fake)
    monkeypatch.setattr(output_controller, "_initialized", True)
    return fake


@pytest.fixture
def configure(monkeypatch):
    """Set the controller's fitting config for one test."""

    def _configure(save_to_database, parameters_to_save=None):
        monkeypatch.setattr(output_controller, "save_to_database", save_to_database)
        monkeypatch.setattr(output_controller, "parameters_to_save", parameters_to_save)

    return _configure


def make_fit(path=None, fitting=None):
    """Build a bare ``FittingClass`` without hitting the database in ``__init__``."""
    fit = FittingClass.__new__(FittingClass)
    fit.id = 42
    fit.fitting = {} if fitting is None else dict(fitting)
    fit.path = path
    fit._results = None
    return fit


# --------------------------------- results ----------------------------------


def test_results_refuse_to_be_read_until_a_fit_assigns_them():
    """Every fitted value is read through ``results``, so a read before ``fit()`` raises without a per-method check."""
    fit = make_fit()
    with pytest.raises(RuntimeError, match="FittingClass has no fit result"):
        _ = fit.results

    fit.results = {"center": 0.1}

    assert fit.results == {"center": 0.1}


# --------------------------- add_fitted_parameter ---------------------------


def test_add_fitted_parameter_stores_the_plain_key_and_rounds_to_six_decimals():
    fit = make_fit()
    fit.add_fitted_parameter(FittedParameter.T1, 1.23456789)
    assert fit.fitting == {"t1": 1.234568}
    assert type(next(iter(fit.fitting))) is str


def test_add_fitted_parameter_keeps_fidelity_resolution():
    fit = make_fit()
    fit.add_fitted_parameter(FittedParameter.READOUT_FIDELITY, 0.99987654)
    assert fit.fitting["readout_fidelity"] == pytest.approx(0.999877, abs=0)


def test_add_fitted_parameter_accumulates_multiple_parameters():
    fit = make_fit()
    fit.add_fitted_parameter(FittedParameter.T1, 10.0)
    fit.add_fitted_parameter(FittedParameter.T2, 20.0)
    assert fit.fitting == {"t1": 10.0, "t2": 20.0}


# ------------------------- save_fitting_to_database -------------------------


def test_save_fitting_warns_and_skips_when_unset(fake_db, configure):
    configure(None)
    fit = make_fit(fitting={"t1": 1.0})
    with pytest.warns(RuntimeWarning):
        fit.save_fitting_to_database()
    assert fake_db.fitting_calls == []


def test_save_fitting_skips_silently_when_disabled(fake_db, configure):
    configure(False)
    fit = make_fit(fitting={"t1": 1.0})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit.save_fitting_to_database()
    assert fake_db.fitting_calls == []


def test_unset_saving_warns_only_on_first_use(fake_db, configure, tmp_path):
    configure(None)
    fit = make_fit(path=str(tmp_path), fitting={"t1": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit.save_fitting_to_database()
        fit.save_plot(go.Figure(), "myplot", save_to_db=True)
        make_fit(fitting={"t2": 2.0}).save_fitting_to_database()
    assert [w.category for w in caught] == [RuntimeWarning]
    assert fake_db.fitting_calls == []


@pytest.mark.parametrize("schema", ["autocal", "measurements"])
def test_save_fitting_writes_all_parameters_without_path(fake_db, configure, schema):
    fake_db.database_schema = schema
    configure(True)
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0, "t2": 2.0}}]


def test_save_fitting_filters_by_parameters_to_save(fake_db, configure):
    configure(True, {"t1"})
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0}}]


@pytest.mark.parametrize(("fitting", "parameters_to_save"), [({}, None), ({"t2": 2.0}, {"t1"})])
def test_save_fitting_skips_when_nothing_to_write(fake_db, configure, fitting, parameters_to_save):
    configure(True, parameters_to_save)
    fit = make_fit(fitting=fitting)
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == []


# ------------------------------- save_plot -------------------------------


def test_save_plot_writes_plot_path_when_enabled(fake_db, configure, tmp_path):
    configure(True)
    fit = make_fit(path=str(tmp_path), fitting={"t1": 1.0})
    filepath = fit.save_plot(go.Figure(), "myplot", save_to_db=True)
    expected = os.path.join(str(tmp_path), f"myplot.{RENDER_FORMAT}")
    assert filepath == expected
    assert fake_db.fitting_calls == [{"id": 42, "path": expected, "parameters": None}]
    assert os.path.isfile(expected)


def test_save_plot_stores_the_path_by_default(fake_db, configure, tmp_path):
    """Every fit stores its plot's path on the measurement entry unless it opts out (QHC-1932)."""
    configure(True)
    fit = make_fit(path=str(tmp_path))
    filepath = fit.save_plot(go.Figure(), "myplot")
    assert [call["path"] for call in fake_db.fitting_calls] == [filepath]


def test_save_plot_skips_db_when_asked(fake_db, configure, tmp_path):
    configure(True)
    fit = make_fit(path=str(tmp_path))
    fit.save_plot(go.Figure(), "myplot", save_to_db=False)
    assert fake_db.fitting_calls == []


def test_save_plot_skips_db_when_saving_is_off(fake_db, configure, tmp_path):
    configure(False)
    fit = make_fit(path=str(tmp_path))
    fit.save_plot(go.Figure(), "myplot", save_to_db=True)
    assert fake_db.fitting_calls == []


def test_save_plot_warns_and_skips_db_when_saving_is_unset(fake_db, configure, tmp_path):
    configure(None)
    fit = make_fit(path=str(tmp_path))
    with pytest.warns(RuntimeWarning, match="save_to_database is unset"):
        fit.save_plot(go.Figure(), "myplot", save_to_db=True)
    assert fake_db.fitting_calls == []


def test_save_plot_skips_db_for_interactive_plot(fake_db, configure, monkeypatch):
    configure(True)
    monkeypatch.setattr(go.Figure, "show", lambda *args, **kwargs: None)
    fit = make_fit(path=None)
    assert fit.save_plot(go.Figure(), "myplot", save_to_db=True) is None
    assert fake_db.fitting_calls == []


# ------------------------- storage config end to end -------------------------


def test_storage_conf_can_disable_saving(fake_db, tmp_path):
    output_controller.reset({"root_dir": str(tmp_path), "save_to_database": False})
    fit = make_fit(path=str(tmp_path), fitting={"t1": 1.0})
    fit.save_fitting_to_database()
    fit.save_plot(go.Figure(), "myplot", save_to_db=True)
    assert fake_db.fitting_calls == []


def test_storage_conf_parameters_to_save_filters_what_is_saved(fake_db, tmp_path):
    output_controller.reset({"root_dir": str(tmp_path), "parameters_to_save": ["t1"]})
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0}}]


@pytest.mark.parametrize("free_phase", [False, True])
def test_oscillation_initial_guess_resolves_a_trace_of_under_two_periods(free_phase):
    """~1.4 periods over 40 points: the FFT bins sit 1.25 periods per unit apart, the scan resolves it."""
    x = np.linspace(0.02, 0.8, 40)
    omega, phase = np.pi / 0.28, (0.7 if free_phase else 0.0)
    y = 0.2 - 0.9 * np.cos(omega * x + phase)

    guess = FittingClass.oscillation_initial_guess(x, y, free_phase=free_phase)

    assert guess["omega"] == pytest.approx(omega, rel=0.02)
    assert guess["offset"] == pytest.approx(0.2, abs=0.05)
    model = FittingClass.cosfunc(x, guess["a"], guess["omega"], guess["offset"], guess["phase_offset"])
    np.testing.assert_allclose(model, y, atol=0.1)
    if not free_phase:
        assert guess["phase_offset"] == pytest.approx(0.0)
        assert guess["a"] == pytest.approx(-0.9, abs=0.05)  # signed, no phase to absorb it


def test_oscillation_initial_guess_of_a_window_with_no_range_has_no_frequency():
    guess = FittingClass.oscillation_initial_guess(np.full(5, 0.3), np.arange(5.0))

    assert guess == {"a": 0.0, "omega": 0.0, "offset": 2.0, "phase_offset": 0.0}


# ------------------------------ fit_lorentzian ------------------------------


@pytest.mark.parametrize("height", [1.0, -1.0], ids=["peak", "dip"])
def test_the_lorentzian_start_is_read_off_the_trace(height):
    x = np.linspace(-5e6, 5e6, 101)
    guess = FittingClass.lorentzian_initial_guess(x, FittingClass.lorentzian(x, height, 1e6, 0.8e6, 0.5))

    assert guess["center"] == pytest.approx(1e6)
    assert guess["offset"] == pytest.approx(0.5, abs=0.05)
    assert guess["amplitude"] == pytest.approx(height, rel=0.05)
    assert guess["width"] == pytest.approx(0.8e6, rel=0.25)


def test_fit_lorentzian_reports_the_line():
    x = 7e6 + np.linspace(-5e6, 5e6, 101)
    y = FittingClass.lorentzian(x, -2.0, 6.2e6, 0.5e6, 3.0) + np.random.default_rng(1).normal(0, 0.02, x.size)

    fit = FittingClass.fit_lorentzian(x, y)

    assert fit["center"] == pytest.approx(6.2e6, abs=20e3)
    assert fit["width"] == pytest.approx(0.5e6, rel=0.1)
    assert fit["amplitude"] == pytest.approx(-2.0, rel=0.1)
    assert fit["r_squared"] > 0.95


@pytest.mark.parametrize("center", [8.5e6, 9.5e6])
def test_fit_lorentzian_finds_a_line_at_the_window_edge(center):
    """A line at the edge pulls the edge baseline, so the trace's mean picks the wrong sign; both are fit."""
    x = np.linspace(-10e6, 10e6, 51)
    y = FittingClass.lorentzian(x, 1.0, center, 1.5e6, 0.0) + np.random.default_rng(3).normal(0, 0.05, x.size)
    assert FittingClass.lorentzian_initial_guess(x, y)["amplitude"] < 0  # the mean alone calls it a dip

    fit = FittingClass.fit_lorentzian(x, y)

    assert fit["center"] == pytest.approx(center, abs=0.3e6)
    assert fit["amplitude"] > 0
    assert fit["r_squared"] > 0.8


def test_the_lorentzian_start_takes_the_sign_it_is_given():
    x = np.linspace(-5e6, 5e6, 101)
    y = FittingClass.lorentzian(x, 1.0, 1e6, 0.8e6, 0.5)

    assert FittingClass.lorentzian_initial_guess(x, y, dip=False)["center"] == pytest.approx(1e6)
    assert FittingClass.lorentzian_initial_guess(x, y, dip=True)["amplitude"] <= 0


def test_fit_lorentzian_drops_a_fit_that_did_not_converge(monkeypatch):
    """lmfit's least_squares returns its last iterate with success False at its evaluation limit."""
    original = fit_base_module.Model.fit

    def unconverged(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.success = False
        return result

    monkeypatch.setattr(fit_base_module.Model, "fit", unconverged)
    x = np.linspace(-5e6, 5e6, 101)

    fit = FittingClass.fit_lorentzian(x, FittingClass.lorentzian(x, 1.0, 1e6, 0.8e6, 0.5))

    assert all(np.isnan(value) for value in fit.values())


def test_fit_lorentzian_of_a_flat_trace_is_nan():
    fit = FittingClass.fit_lorentzian(np.linspace(0, 1, 11), np.ones(11))

    assert all(np.isnan(value) for value in fit.values())
