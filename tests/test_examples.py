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

"""Smoke tests for the files in ``examples/``: each one loads, its tree parses and its storage builds."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from seqtante_open.calibration_run import load_config
from seqtante_open.controllers import CalibrationParser
from seqtante_open.outputs import output_controller

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted([*EXAMPLES_DIR.rglob("*.yml"), *EXAMPLES_DIR.rglob("*.yaml")])


@pytest.fixture
def platform() -> SimpleNamespace:
    """A platform holding only analog settings, so any other access by the parser fails."""
    return SimpleNamespace(analog_compilation_settings=SimpleNamespace(qubits=[1, 14], topology=[[1, 14]]))


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Keeps every folder storage creates under ``tmp_path`` and replaces the database manager with a mock."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    get_db_manager = MagicMock(name="get_db_manager", return_value=MagicMock(database_schema="autocal"))
    monkeypatch.setattr("seqtante_open.outputs.outputs.get_db_manager", get_db_manager)
    yield get_db_manager
    output_controller._initialize()


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(EXAMPLES_DIR).as_posix())
class TestExample:
    def test_tree_parses(self, path, platform):
        CalibrationParser(calibration_config=load_config(path), platform=platform).parse()

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_storage_builds(self, path, isolated_storage, tmp_path):
        output_controller.reset(storage_conf=load_config(path).get("storage", {}))
        assert Path(output_controller.storage_path).resolve().is_relative_to(tmp_path.resolve())
        isolated_storage.assert_called_once()
