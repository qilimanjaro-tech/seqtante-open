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

"""Dry run of a calibration: the real ``CalibrationRun`` with no hardware, writing to a local database.

Run from the repository root, which puts ``tests`` (the fake-data builders) on the path::

    uv run python -m scripts.dry_run
    uv run python -m scripts.dry_run --config_path my_tree.yml --calibration my_calibration.yml

See ``README.md`` in this folder for what is simulated, the outputs and the fake-data parameters.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger
from qcodes import Instrument
from qililab import result as qililab_result
from qililab.data_management import build_platform
from qililab.result import StreamArray
from qililab.result.database import get_db_manager
from ruamel.yaml import YAML
from sqlalchemy.exc import OperationalError

from db_docker.create_tables import DATABASE_INI, SECTION, create_tables
from scripts.dry_run.fake_calibration import collect, fake_calibration
from scripts.dry_run.fake_data import LEGACY_EXPERIMENTS, Scope, environment_errors, fake_data
from scripts.dry_run.qdac_sim import attach_simulated_qdacs
from seqtante_open import calibration_run
from seqtante_open.controllers.calibration_node import CalibrationNode, write_per_point, write_whole

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

#: Modules that stream measurements, imported before their ``StreamArray`` is swapped.
STREAMING_MODULES = ("seqtante_open.experiments.nodes", "seqtante_open.experiments.experiment_functions")


class _AnyReadoutBus(dict):
    """``results`` of a fake execution: the acquisitions, whichever readout bus the node asks for."""

    def __init__(self, acquisitions: list) -> None:
        super().__init__()
        self.acquisitions = acquisitions

    def __missing__(self, bus: str) -> list:
        return self.acquisitions


class DryRun:
    """The state the patched platform and stream share: the measurement being streamed, and the log."""

    def __init__(self, folder: Path, seed: int) -> None:
        self.folder = folder
        self.rng = np.random.default_rng(seed)
        self.streaming: tuple[str, dict, str] | None = None
        self.node: tuple[str, str] = ("", "")
        self.per_point = False
        self.executions = 0
        self.qdacs: dict = {}
        self.scpi: dict[str, list[str]] = {}

    def build_platform(self, runcard: str):
        """The runcard's platform with simulated QDACs, connecting to nothing and executing with fake data."""
        platform = build_platform(runcard=runcard)
        self.qdacs = attach_simulated_qdacs(platform)
        for lifecycle in ("connect", "initial_setup", "turn_on_instruments", "disconnect"):
            setattr(platform, lifecycle, lambda *args, **kwargs: None)
        compile_qprogram = platform.compile_qprogram

        def execute_qprogram(qprogram, bus_mapping=None, calibration=None, crosstalk=True, debug=False):
            output = compile_qprogram(qprogram, bus_mapping=bus_mapping, calibration=calibration, crosstalk=crosstalk)
            if output.qdac is not None:
                for qdac in output.qdac.qdacs:
                    qdac.start()
            scpi = self._drain_scpi()
            if self.streaming is None:
                raise RuntimeError("execute_qprogram called outside a StreamArray: no sweep to fake data on")
            experiment_name, loops, target = self.streaming
            experiment, step = self.node
            scope = Scope(target, experiment or LEGACY_EXPERIMENTS.get(experiment_name, ""), step)
            data, values, fail = fake_data(experiment_name, loops, scope, self.rng)
            self._save_execution(experiment_name, scope, output, values, scpi)
            if fail == "execution":
                raise RuntimeError(f"dry run: {target} {experiment_name} forced to fail its execution (FAIL=execution)")
            if self.per_point:
                acquisitions = [SimpleNamespace(array=point[:, None]) for point in data]
            else:
                acquisitions = [SimpleNamespace(array=np.moveaxis(data, -1, 0))]
            return SimpleNamespace(results=_AnyReadoutBus(acquisitions))

        platform.execute_qprogram = execute_qprogram
        return platform

    def _drain_scpi(self) -> dict[str, list[str]]:
        """The SCPI each QDAC received since the last drain, also kept for the whole run's log."""
        drained = {alias: device.get_recorded_scpi_commands() for alias, device in self.qdacs.items()}
        for alias, commands in drained.items():
            self.scpi.setdefault(alias, []).extend(commands)
        return drained

    def _save_execution(self, experiment_name: str, scope: Scope, output, values: dict, scpi: dict) -> None:
        """Save one execution in its own folder: the Q1ASM of every bus, the SCPI each QDAC received since
        the previous execution, and the fake-data values."""
        self.executions += 1
        folder = self.folder / "executions" / f"{self.executions:03d}_{experiment_name}_{scope.target}"
        folder.mkdir(parents=True, exist_ok=True)
        record = {"target": scope.target, "experiment": scope.experiment, "step": scope.step, **values}
        (folder / "fake_data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        if output.qblox is not None:
            for bus, sequence in output.qblox.sequences.items():
                (folder / f"{bus}.q1asm").write_text(str(sequence._program), encoding="utf-8")
        for alias, commands in scpi.items():
            (folder / f"{alias}.scpi").write_text("\n".join(commands), encoding="utf-8")

    def save_scpi(self) -> None:
        """Save every SCPI command each QDAC received during the run, in order."""
        self._drain_scpi()
        for alias, commands in self.scpi.items():
            (self.folder / f"{alias}.scpi").write_text("\n".join(commands), encoding="utf-8")

    def install(self) -> None:
        """Route every StreamArray and ``CalibrationNode._stream`` through this dry run."""
        dry_run = self
        for name in STREAMING_MODULES:
            importlib.import_module(name)

        class DryRunStreamArray(StreamArray):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                target = kwargs.get("qubit_idx")
                dry_run.streaming = (kwargs["experiment_name"], kwargs["loops"], "" if target is None else str(target))

        for module in list(sys.modules.values()):
            if module is not None and module.__name__.startswith("seqtante_open"):
                if getattr(module, "StreamArray", None) is StreamArray:
                    module.StreamArray = DryRunStreamArray
        qililab_result.StreamArray = DryRunStreamArray

        stream = CalibrationNode._stream

        def _stream(node, target, qprogram, *, write=write_whole, **kwargs):
            dry_run.per_point = write is write_per_point
            dry_run.node = (node.experiment or "", node.name)
            try:
                return stream(node, target, qprogram, write=write, **kwargs)
            finally:
                dry_run.per_point = False
                dry_run.node = ("", "")

        CalibrationNode._stream = _stream
        calibration_run.build_platform = self.build_platform


def _run_database_ini(database_ini: str, section: str, folder: Path) -> tuple[Path, str]:
    """The run's own ``.ini``, holding only ``section``, with the data written inside the run's folder.

    Returns:
        The ``.ini`` and the section's schema: its ``database_schema``, or ``autocal`` when it sets none.
    """
    source = ConfigParser()
    if not source.read(database_ini):
        sys.exit(f"cannot read the database config {database_ini}")
    if not source.has_section(section):
        sys.exit(f"{database_ini} has no [{section}] section")
    schema = source[section].get("database_schema", "autocal")
    parser = ConfigParser()
    parser[section] = {
        **source[section],
        "database_schema": schema,
        "base_path_local": str(folder),
        "base_path_shared": str(folder),
        "data_write_folder": "measurements",
    }
    ini = folder / "database.ini"
    with ini.open("w", encoding="utf-8") as stream:
        parser.write(stream)
    return ini, schema


def _prepare_database(database_ini: Path, section: str, is_db_docker: bool) -> None:
    """Create db_docker's tables, or only check that another database can be reached; never create tables there."""
    try:
        if is_db_docker:
            create_tables(database_ini, section)
        else:
            with get_db_manager(path=str(database_ini), database_name=section).engine.connect():
                pass
    except OperationalError as error:
        sys.exit(
            f"cannot reach the database of [{section}] ({error.__class__.__name__}). "
            "For db_docker's: cd db_docker && docker compose up -d"
        )


def _steps_and_experiments(tree: dict) -> list[str]:
    """Every step name of the tree's pipelines and every experiment they run, legacy ones included."""
    steps = [
        (name, step.get("experiment", ""))
        for pipeline in (tree.get("pipelines") or {}).values()
        for name, step in (pipeline.get("steps") or {}).items()
    ]
    return list(dict.fromkeys([*(name for name, _ in steps), *(experiment for _, experiment in steps if experiment)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform_path", default=str(REPO_ROOT / "tests" / "runcards" / "test_AQPU_runcard.yml"))
    parser.add_argument("--config_path", default=str(HERE / "tree.yml"))
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--calibration", default=None, help="Calibration file to start from; copied, never written.")
    parser.add_argument("--database_ini", default=str(DATABASE_INI), help="Database config; db_docker's by default.")
    parser.add_argument("--database_section", default=SECTION)
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "dry_runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-report", dest="report", action="store_false")
    args = parser.parse_args()

    yaml = YAML()
    tree = yaml.load(Path(args.config_path).read_text(encoding="utf-8"))
    errors = environment_errors(collect(tree, "targets"), _steps_and_experiments(tree))
    if errors:
        sys.exit(f"fake-data parameters: {'; '.join(errors)}. See scripts/dry_run/README.md")

    folder = Path(args.data_root).resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder.mkdir(parents=True)
    database_ini, schema = _run_database_ini(args.database_ini, args.database_section, folder)
    is_db_docker = Path(args.database_ini).resolve() == DATABASE_INI.resolve()
    _prepare_database(database_ini, args.database_section, is_db_docker)

    tree["storage"] = {
        "root_dir": str(folder / "seqtante"),
        "db_manager": {"path": str(database_ini), "database_name": args.database_section, "database_schema": schema},
        "save_to_database": True,
    }
    tree_path = folder / "tree.yml"
    with tree_path.open("w", encoding="utf-8") as stream:
        yaml.dump(tree, stream)

    calibration_path = folder / "calibration.yml"
    if args.calibration is not None:
        shutil.copyfile(args.calibration, calibration_path)
    else:
        fake_calibration(tree, build_platform(runcard=args.platform_path), str(calibration_path))

    np.random.seed(args.seed)
    dry_run = DryRun(folder, args.seed)
    dry_run.install()
    logger.info(f"Dry run in {folder}")
    try:
        calibration_run.CalibrationRun(
            args.platform_path,
            str(tree_path),
            pipelines=args.pipelines,
            calibration=str(calibration_path),
            report=args.report,
        )
    finally:
        dry_run.save_scpi()
        Instrument.close_all()
        sys.stdout.write(f"Dry run in {folder}: {dry_run.executions} executions\n")


if __name__ == "__main__":
    main()
