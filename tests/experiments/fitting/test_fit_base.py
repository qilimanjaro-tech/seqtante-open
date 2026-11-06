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

import plotly.graph_objects as go
import pytest

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

    def _configure(save_to_database, specify_parameters=None):
        monkeypatch.setattr(output_controller, "save_to_database", save_to_database)
        monkeypatch.setattr(output_controller, "specify_parameters", specify_parameters)

    return _configure


def make_fit(path=None, fitting=None):
    """Build a bare ``FittingClass`` without hitting the database in ``__init__``."""
    fit = FittingClass.__new__(FittingClass)
    fit.id = 42
    fit.fitting = {} if fitting is None else dict(fitting)
    fit.path = path
    return fit


# ------------------------------- add_fitting -------------------------------


def test_add_fitting_stores_the_plain_key_and_rounds_to_six_decimals():
    fit = make_fit()
    fit.add_fitting(FittedParameter.T1, 1.23456789)
    assert fit.fitting == {"t1": 1.234568}
    assert type(next(iter(fit.fitting))) is str


def test_add_fitting_keeps_fidelity_resolution():
    fit = make_fit()
    fit.add_fitting(FittedParameter.READOUT_FIDELITY, 0.99987654)
    assert fit.fitting["readout_fidelity"] == pytest.approx(0.999877, abs=0)


def test_add_fitting_accumulates_multiple_parameters():
    fit = make_fit()
    fit.add_fitting(FittedParameter.T1, 10.0)
    fit.add_fitting(FittedParameter.T2, 20.0)
    assert fit.fitting == {"t1": 10.0, "t2": 20.0}


# ------------------------- save_fitting_to_database -------------------------


def test_save_fitting_warns_and_skips_when_unset(fake_db, configure):
    configure(None)
    fit = make_fit(fitting={"t1": 1.0})
    with pytest.warns(RuntimeWarning):
        fit.save_fitting_to_database()
    assert fake_db.fitting_calls == []


def test_save_fitting_skips_when_disabled(fake_db, configure):
    configure(False)
    fit = make_fit(fitting={"t1": 1.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == []


@pytest.mark.parametrize("schema", ["autocal", "measurements"])
def test_save_fitting_writes_all_parameters_without_path(fake_db, configure, schema):
    fake_db.database_schema = schema
    configure(True)
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0, "t2": 2.0}}]


def test_save_fitting_filters_by_specify_parameters(fake_db, configure):
    configure(True, {"t1"})
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0}}]


@pytest.mark.parametrize(("fitting", "specify_parameters"), [({}, None), ({"t2": 2.0}, {"t1"})])
def test_save_fitting_skips_when_nothing_to_write(fake_db, configure, fitting, specify_parameters):
    configure(True, specify_parameters)
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


def test_save_plot_skips_db_by_default(fake_db, configure, tmp_path):
    configure(True)
    fit = make_fit(path=str(tmp_path))
    fit.save_plot(go.Figure(), "myplot")
    assert fake_db.fitting_calls == []


@pytest.mark.parametrize("save_to_database", [False, None])
def test_save_plot_skips_db_when_saving_is_off_or_unset(fake_db, configure, tmp_path, save_to_database):
    configure(save_to_database)
    fit = make_fit(path=str(tmp_path))
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


def test_storage_conf_specify_parameters_filters_what_is_saved(fake_db, tmp_path):
    output_controller.reset({"root_dir": str(tmp_path), "specify_parameters": ["t1"]})
    fit = make_fit(fitting={"t1": 1.0, "t2": 2.0})
    fit.save_fitting_to_database()
    assert fake_db.fitting_calls == [{"id": 42, "path": None, "parameters": {"t1": 1.0}}]
