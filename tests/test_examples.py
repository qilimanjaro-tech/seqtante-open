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

import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from seqtante_open.calibration_run import load_config
from seqtante_open.controllers import CalibrationParser
from seqtante_open.outputs import output_controller

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted([*EXAMPLES_DIR.rglob("*.yml"), *EXAMPLES_DIR.rglob("*.yaml")])

BuildStorage = Callable[[Mapping[str, Any]], None]


@pytest.fixture
def stub_platform() -> SimpleNamespace:
    """A platform holding only analog settings, so any other access by the parser fails."""
    return SimpleNamespace(analog_compilation_settings=SimpleNamespace(qubits=[1, 14], topology=[[1, 14]]))


@pytest.fixture
def build_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[BuildStorage]:
    """Builds a ``storage`` block with every folder under ``tmp_path`` and a mock database manager.

    The build fails if it warns, writes outside ``tmp_path`` or does not pass its ``db_manager`` settings on.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    get_db_manager = MagicMock(
        name="get_db_manager", side_effect=lambda **kwargs: MagicMock(database_schema=kwargs["database_schema"])
    )
    monkeypatch.setattr("seqtante_open.outputs.outputs.get_db_manager", get_db_manager)

    def build(storage: Mapping[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            output_controller.reset(storage_conf=dict(storage))
        assert Path(output_controller.storage_path).resolve().is_relative_to(tmp_path.resolve())
        get_db_manager.assert_called_once()
        assert get_db_manager.call_args.kwargs.items() >= dict(storage.get("db_manager") or {}).items()

    yield build
    output_controller._initialize()


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(EXAMPLES_DIR).as_posix())
class TestExample:
    # TODO: also build the graph and its nodes once the node classes and NODE_REGISTRY land.
    def test_tree_parses(self, path, stub_platform):
        CalibrationParser(calibration_config=load_config(path), platform=stub_platform).parse()

    def test_storage_builds(self, path, build_storage):
        build_storage(load_config(path).get("storage", {}))


class TestBuildStorageFails:
    """The checks ``build_storage`` runs on every example do catch a broken ``storage`` block."""

    def test_invalid_db_manager_key(self, build_storage):
        with pytest.raises(RuntimeWarning, match=r"Invalid key/s \['schema'\]"):
            build_storage({"db_manager": {"schema": "autocal"}})

    def test_qaas_schema(self, build_storage):
        with pytest.raises(ValueError, match="cannot run on the 'qaas' database schema"):
            build_storage({"db_manager": {"database_schema": "qaas"}})

    def test_root_dir_outside_tmp_path(self, build_storage, tmp_path):
        with pytest.raises(AssertionError):
            build_storage({"root_dir": str(tmp_path.parent / f"{tmp_path.name}_outside")})
