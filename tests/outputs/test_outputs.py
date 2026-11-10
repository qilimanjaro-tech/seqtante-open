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

"""Tests for ``Outputs``, the class behind the ``output_controller`` singleton.

Each test builds its own ``Outputs`` and resets it with ``root_dir`` in
``tmp_path``, so the singleton is never touched. ``get_db_manager`` is patched
to return a MagicMock with the requested ``database_schema``, so nothing hits a
database. The runs the mock manager returns are real ``SequenceRun`` and
``CalibrationRun`` instances, since ``end_run`` and ``run_id`` dispatch on their
type.
"""

import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from qililab.result.database import CalibrationRun, SequenceRun

from seqtante_open.outputs.outputs import DB_MANAGER_KEYS, Outputs

TREE = {"sample": "chip", "node_types": ["1q"]}
SEQUENCE_ID = 7
CALIBRATION_ID = 9


def make_db_manager(database_schema: str) -> MagicMock:
    db_manager = MagicMock(name="db_manager", database_schema=database_schema)
    db_manager.add_sequence_run.return_value = SequenceRun(sequence_id=SEQUENCE_ID)
    db_manager.add_calibration_run.return_value = CalibrationRun(calibration_id=CALIBRATION_ID)
    return db_manager


@pytest.fixture
def get_db_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``get_db_manager`` where ``outputs`` looks it up.

    Without a ``database_schema`` the mock falls back to ``"measurements"``,
    like qililab does when the ``.ini`` section doesn't declare one either.
    """
    mock = MagicMock(side_effect=lambda **kwargs: make_db_manager(kwargs["database_schema"] or "measurements"))
    monkeypatch.setattr("seqtante_open.outputs.outputs.get_db_manager", mock)
    return mock


@pytest.fixture
def make_outputs(tmp_path: Path, get_db_manager: MagicMock) -> Callable[..., Outputs]:
    """Build an ``Outputs`` reset with ``root_dir`` in ``tmp_path``, the given ``db_manager`` config and any other
    ``storage`` keys."""

    def _make(db_conf: dict | None = None, **storage: Any) -> Outputs:
        outputs = Outputs()
        storage_conf: dict = {"root_dir": str(tmp_path), **storage}
        if db_conf is not None:
            storage_conf["db_manager"] = db_conf
        outputs.reset(storage_conf)
        return outputs

    return _make


# ------------------------------- Initialization -------------------------------


@pytest.mark.parametrize("prop", ["storage_path", "db_manager"])
def test_properties_require_reset(prop):
    out = Outputs()
    with pytest.raises(AttributeError, match="has not been initialized"):
        getattr(out, prop)


def test_reset_creates_run_folder_in_day_folder(make_outputs, tmp_path):
    outputs = make_outputs()
    storage_path = Path(outputs.storage_path)

    assert storage_path.is_dir()
    assert storage_path.parent == tmp_path / datetime.now().strftime("%Y-%m-%d")
    assert storage_path.name.startswith("run_")


def test_reset_clears_the_previous_run(make_outputs):
    outputs = make_outputs({"database_schema": "autocal"})
    outputs.start_run(TREE, sample_name="chip", cooldown="cd1", sequence_name=None)

    outputs.reset({"root_dir": outputs.storage_conf["root_dir"]})

    assert outputs.run is None
    assert outputs.run_id == -1


# ------------------------------- Fitting persistence -------------------------------


def test_fitting_config_is_unset_before_reset():
    outputs = Outputs()

    assert outputs.storage_conf == {}
    assert outputs.save_to_database is None
    assert outputs.parameters_to_save is None


def test_reset_enables_saving_to_database_by_default(make_outputs):
    outputs = make_outputs()

    assert outputs.save_to_database is True
    assert outputs.parameters_to_save is None


def test_reset_without_storage_conf_enables_saving_to_database(get_db_manager, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    outputs = Outputs()

    outputs.reset(None)

    assert outputs.save_to_database is True
    assert outputs.parameters_to_save is None


def test_reset_can_disable_saving_to_database(make_outputs):
    assert make_outputs(save_to_database=False).save_to_database is False


def test_fitting_saving_enabled_follows_save_to_database(make_outputs):
    assert make_outputs().fitting_saving_enabled is True
    assert make_outputs(save_to_database=False).fitting_saving_enabled is False


def test_fitting_saving_enabled_warns_once_while_unset():
    outputs = Outputs()

    with pytest.warns(RuntimeWarning, match="save_to_database is unset"):
        assert outputs.fitting_saving_enabled is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert outputs.fitting_saving_enabled is False


def test_parameters_to_save_is_stored_as_a_set(make_outputs):
    assert make_outputs(parameters_to_save=["t1", "t2", "t1"]).parameters_to_save == {"t1", "t2"}


# ------------------------------- Database manager -------------------------------


def test_db_manager_uses_defaults_without_config(make_outputs, get_db_manager):
    make_outputs()

    get_db_manager.assert_called_once_with(path="~/database.ini", database_name="postgresql", database_schema=None)


def test_db_manager_config_is_passed_to_get_db_manager(make_outputs, get_db_manager):
    outputs = make_outputs({"path": "db.ini", "database_name": "calibration", "database_schema": "autocal"})

    get_db_manager.assert_called_once_with(path="db.ini", database_name="calibration", database_schema="autocal")
    assert outputs.db_manager.database_schema == "autocal"


def test_db_manager_is_created_once(make_outputs, get_db_manager):
    outputs = make_outputs()

    assert outputs.db_manager is outputs.db_manager
    get_db_manager.assert_called_once()


def test_unknown_db_manager_key_warns(make_outputs, get_db_manager):
    with pytest.warns(RuntimeWarning, match=r"\['bad_key'\]") as record:
        make_outputs({"database_schema": "autocal", "bad_key": 1})

    assert str(DB_MANAGER_KEYS) in str(record[0].message)
    get_db_manager.assert_called_once()


def test_qaas_schema_raises(make_outputs):
    with pytest.raises(ValueError, match="'qaas'"):
        make_outputs({"database_schema": "qaas"})


# ------------------------------- Run entry -------------------------------


def test_run_id_is_minus_one_before_a_run_starts(make_outputs):
    assert make_outputs({"database_schema": "autocal"}).run_id == -1


@pytest.mark.parametrize("cooldown", ["cd1", None])
def test_measurements_schema_starts_a_sequence_run(make_outputs, cooldown):
    outputs = make_outputs({"database_schema": "measurements"})
    db_manager = outputs.db_manager

    outputs.start_run(TREE, sample_name="chip", cooldown=cooldown, sequence_name="seq")

    db_manager.set_sample_and_cooldown.assert_called_once_with("chip", cooldown)
    db_manager.add_sequence_run.assert_called_once_with(
        sequence_name="seq", sequence_tree=TREE, sample_name="chip", cooldown=cooldown
    )
    db_manager.add_calibration_run.assert_not_called()
    assert outputs.run_id == SEQUENCE_ID


def test_measurements_schema_requires_sequence_name(make_outputs):
    outputs = make_outputs({"database_schema": "measurements"})

    with pytest.raises(ValueError, match="'sequence_name' is required"):
        outputs.start_run(TREE, sample_name="chip", cooldown="cd1", sequence_name=None)

    assert outputs.db_manager.method_calls == []
    assert outputs.run_id == -1


@pytest.mark.parametrize("sequence_name", ["seq", None])
def test_autocal_schema_starts_a_calibration_run(make_outputs, sequence_name):
    outputs = make_outputs({"database_schema": "autocal"})
    db_manager = outputs.db_manager

    outputs.start_run(TREE, sample_name="chip", cooldown="cd1", sequence_name=sequence_name)

    db_manager.add_calibration_run.assert_called_once_with(calibration_tree=TREE, sample_name="chip", cooldown="cd1")
    db_manager.add_sequence_run.assert_not_called()
    db_manager.set_sample_and_cooldown.assert_not_called()
    assert outputs.run_id == CALIBRATION_ID


def test_autocal_schema_requires_cooldown(make_outputs):
    outputs = make_outputs({"database_schema": "autocal"})

    with pytest.raises(ValueError, match="'cooldown' is required"):
        outputs.start_run(TREE, sample_name="chip", cooldown=None, sequence_name="seq")

    assert outputs.db_manager.method_calls == []
    assert outputs.run_id == -1


@pytest.mark.parametrize(
    ("database_schema", "run_class", "end_method", "add_method"),
    [
        pytest.param("measurements", SequenceRun, "end_sequence", "add_sequence_run", id="measurements"),
        pytest.param("autocal", CalibrationRun, "end_calibration", "add_calibration_run", id="autocal"),
    ],
)
def test_end_run_ends_the_started_run(make_outputs, database_schema, run_class, end_method, add_method):
    outputs = make_outputs({"database_schema": database_schema})
    db_manager = outputs.db_manager
    run = MagicMock(spec=run_class)
    getattr(db_manager, add_method).return_value = run
    outputs.start_run(TREE, sample_name="chip", cooldown="cd1", sequence_name="seq")

    outputs.end_run()

    getattr(run, end_method).assert_called_once_with(db_manager.session)


def test_end_run_without_a_run_does_nothing(make_outputs):
    outputs = make_outputs({"database_schema": "autocal"})

    outputs.end_run()

    assert outputs.db_manager.method_calls == []


# ------------------------------- Logger -------------------------------


@pytest.fixture
def logger_configure(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch loguru's ``logger.configure`` so the global logger and its file sinks stay untouched."""
    mock = MagicMock(name="configure")
    monkeypatch.setattr("seqtante_open.outputs.outputs.logger.configure", mock)
    return mock


def test_setup_logger_binds_the_run_id(make_outputs, logger_configure):
    outputs = make_outputs({"database_schema": "autocal"})
    outputs.start_run(TREE, sample_name="chip", cooldown="cd1", sequence_name=None)

    outputs.setup_logger()

    extra = logger_configure.call_args.kwargs["extra"]
    assert extra["run_id"] == CALIBRATION_ID
    assert extra["app"] == "seqtante_open"
    assert "generated_uuid" in extra


def test_setup_logger_maps_folder_sinks_to_the_storage_path(make_outputs, logger_configure):
    outputs = make_outputs({"database_schema": "autocal"})

    outputs.setup_logger()

    file_sinks = [h["sink"] for h in logger_configure.call_args.kwargs["handlers"] if isinstance(h["sink"], str)]
    assert file_sinks
    assert all(sink.startswith(outputs.storage_path) for sink in file_sinks)
