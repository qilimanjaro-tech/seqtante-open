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
import json
import os
import sys
from datetime import datetime
from enum import StrEnum
from functools import wraps
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TypedDict, assert_never, cast
from uuid import uuid4
from warnings import warn

import yaml
from loguru import logger
from qililab.result.database import CalibrationRun, DatabaseManager, DatabaseSchema, SequenceRun, get_db_manager

DB_MANAGER_KEYS = ["path", "database_name", "database_schema"]
RunSchema = Literal["autocal", "measurements"]

# ---------------------Logger constants-----------------------
DEFAULT_LOG_CONFIG = Path(str(files("seqtante_open.outputs") / "default_logger.json"))
DEFAULT_FOLDER_ALIAS = "SEQTANTE_FOLDER"

SINKS = {
    "stderr": sys.stderr,
    "stdout": sys.stdout,
}

FILTERS = {
    "lt_warning": lambda r: r["level"].no < logger.level("WARNING").no,
    "seqtante_open": lambda r: r["name"].startswith("seqtante_open."),
    "lt_warning_seqtante_open": lambda r: (
        r["name"].startswith("seqtante_open.") and r["level"].no < logger.level("WARNING").no
    ),
}


class DbManagerConf(TypedDict, total=False):
    """`storage.db_manager` section of the calibration config."""

    path: str
    database_name: str
    database_schema: DatabaseSchema


class LogConf(TypedDict, total=False):
    """`storage.log_config` section, pointing at the loguru config file."""

    file_dir: str
    file_type: str


class StorageConf(TypedDict, total=False):
    """`storage` section of the calibration config."""

    root_dir: str
    db_manager: DbManagerConf
    log_config: LogConf
    save_to_database: bool
    parameters_to_save: list[str]


class FittedParameter(StrEnum):
    """Keys fits write to the ``fitting_parameters`` column of their measurement entry (QHC-1497)."""

    T1 = "t1"
    T2 = "t2"
    T2_ECHO = "t2_echo"
    READOUT_FIDELITY = "readout_fidelity"
    GATE_FIDELITY_1Q = "gate_fidelity_1q"
    GATE_FIDELITY_2Q = "gate_fidelity_2q"
    DRIVE_IF = "drive_if"
    DRIVE_LO = "drive_lo"
    READOUT_IF = "readout_if"
    READOUT_LO = "readout_lo"
    RABI_AMPLITUDE = "rabi_amplitude"
    DRAG_COEFFICIENT = "drag_coefficient"
    CZ_AMPLITUDE = "cz_amplitude"
    FLUX_OFFSET = "flux_offset"
    SWEET_SPOT_FLUX_OFFSET_Z = "sweet_spot_flux_offset_z"
    SWEET_SPOT_FLUX_OFFSET_X = "sweet_spot_flux_offset_x"
    THERMAL_POPULATION = "thermal_population"
    FIT_R2 = "fit_r2"
    FIT_REDCHI = "fit_redchi"


def _requires_initialized(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise AttributeError("Output has not been initialized. Use `Outputs.reset` to initialize.")
        return fn(self, *args, **kwargs)

    return wrapper


class Outputs:
    def __init__(self):
        self._initialize()

    def _initialize(self):
        self._initialized = False
        self._db_manager: DatabaseManager | None = None
        self.run: CalibrationRun | SequenceRun | None = None

        self._storage_path: str | None = None

        self.storage_conf: StorageConf = {}
        self.save_to_database: bool | None = None
        self.parameters_to_save: set[str] | None = None
        self._warned_save_to_database_unset = False

    def reset(self, storage_conf: StorageConf | None = None):
        self._initialize()
        self.storage_conf = storage_conf or {}
        self._initialized = True
        self._storage_path = self._create_storage_path()
        self._db_manager = self._create_db_manager()
        self.save_to_database = self.storage_conf.get("save_to_database", True)
        parameters_to_save = self.storage_conf.get("parameters_to_save")
        self.parameters_to_save = set(parameters_to_save) if parameters_to_save is not None else None

    @property
    @_requires_initialized
    def storage_path(self) -> str:
        if self._storage_path is None:
            self._storage_path = self._create_storage_path()
        return self._storage_path

    def _create_storage_path(self) -> str:
        # Allow ~ and $VARS in YAML paths
        root_dir = self.storage_conf.get("root_dir", "data")
        root_dir = os.path.expanduser(os.path.expandvars(root_dir))

        # Day folder (YYYY-MM-DD)
        day_str = datetime.now().strftime("%Y-%m-%d")
        day_folder = os.path.join(root_dir, day_str)

        # Per-run folder inside that day folder
        run_timestamp = datetime.now().strftime("%H-%M-%S")
        storage_path = os.path.join(day_folder, f"run_{run_timestamp}")
        os.makedirs(storage_path, exist_ok=True)
        return storage_path

    @property
    @_requires_initialized
    def db_manager(self) -> DatabaseManager:
        if self._db_manager is None:
            self._db_manager = self._create_db_manager()
        return self._db_manager

    def _create_db_manager(self) -> DatabaseManager:
        db_conf = self.storage_conf.get("db_manager") or {}

        key_warning = [key for key in db_conf if key not in DB_MANAGER_KEYS]
        if key_warning:
            warn(
                f"Invalid key/s {key_warning} found in db_manager config, only {DB_MANAGER_KEYS} are valid.",
                RuntimeWarning,
            )
        db_manager = get_db_manager(
            path=db_conf.get("path", "~/database.ini"),
            database_name=db_conf.get("database_name", "postgresql"),
            database_schema=db_conf.get("database_schema"),
        )
        if db_manager.database_schema == "qaas":
            raise ValueError("Seqtante cannot run on the 'qaas' database schema, use 'autocal' or 'measurements'.")
        return db_manager

    @property
    def fitting_saving_enabled(self) -> bool:
        """Whether fits write their results to the database, following ``storage.save_to_database``.

        ``save_to_database`` is unset until :meth:`reset`, so a fit used outside a calibration run writes
        nothing. A warning is raised the first time that happens.
        """
        if self.save_to_database is None:
            if not self._warned_save_to_database_unset:
                warn(
                    "save_to_database is unset because the output controller has not been reset; fitting results "
                    "are not saved to the database.",
                    RuntimeWarning,
                )
                self._warned_save_to_database_unset = True
            return False
        return self.save_to_database

    # Functions related to the run entry

    def start_run(self, run_tree: dict, sample_name: str, cooldown: str | None, sequence_name: str | None):
        """Start a sequence run ('measurements' schema) or a calibration run ('autocal' schema) in the database"""
        # _create_db_manager rejects 'qaas'
        schema = cast("RunSchema", self.db_manager.database_schema)
        match schema:
            case "measurements":
                if sequence_name is None:
                    raise ValueError(
                        "'sequence_name' is required in the config for the 'measurements' database schema."
                    )
                self.db_manager.set_sample_and_cooldown(sample_name, cooldown)
                self.run = self.db_manager.add_sequence_run(
                    sequence_name=sequence_name,
                    sequence_tree=run_tree,
                    sample_name=sample_name,
                    cooldown=cooldown,
                )
            case "autocal":
                if cooldown is None:
                    raise ValueError("'cooldown' is required in the config for the 'autocal' database schema.")
                self.run = self.db_manager.add_calibration_run(
                    calibration_tree=run_tree,
                    sample_name=sample_name,
                    cooldown=cooldown,
                )
            case _:
                assert_never(schema)

    def end_run(self):
        """End the run entry and save it to the database if one has been started"""
        if isinstance(self.run, SequenceRun):
            self.run.end_sequence(self.db_manager.session)
        elif isinstance(self.run, CalibrationRun):
            self.run.end_calibration(self.db_manager.session)

    @property
    def run_id(self) -> int:
        """Id of the current sequence or calibration run in the database, -1 if none has been started"""
        if isinstance(self.run, SequenceRun):
            return self.run.sequence_id
        if isinstance(self.run, CalibrationRun):
            return self.run.calibration_id
        return -1

    def setup_logger(self):
        """Sets up the loguru logger with the given or default configuration

        Args:
            data_folder (str): Folder where the standard calibration outputs (.h5 and plots) are saved. Used if the field "sink" starts with "SEQTANTE_FOLDER"
            run_id (int, optional): Id of the sequence or calibration run in the database. Used if RUN_ID in config["extras"].
            config (dict, optional): Log configuration dict, if one isn't provided or "file_dir" not in the dict, it uses the default file.

        Returns:
            list[int]
        """
        config = self.storage_conf.get("log_config", None) or {}
        try:
            file_dir = config.get("file_dir")
            path = Path(file_dir) if file_dir else DEFAULT_LOG_CONFIG
        except (TypeError, ValueError):
            path = DEFAULT_LOG_CONFIG

        if not path.is_absolute():
            path = path.expanduser().resolve()

        if config.get("file_type", "").lower() == "yaml" and path != DEFAULT_LOG_CONFIG:
            cfg = yaml.safe_load(path.read_text(encoding="utf8"))
        else:
            cfg = json.loads(path.read_text(encoding="utf8"))

        handlers: list[Any] = []
        for h in cfg.get("handlers", []):
            params: dict[str, Any] = dict(h)

            # Map sinks. If log directory starts with "SEQTANTE_FOLDER", "SEQTANTE_FOLDER" is substituted by the data folder
            sink = params.get("sink")
            if isinstance(sink, str):
                if sink in SINKS:
                    params["sink"] = SINKS[sink]
                elif sink.startswith(DEFAULT_FOLDER_ALIAS):
                    params["sink"] = sink.replace(DEFAULT_FOLDER_ALIAS, self.storage_path, 1)

            # Map filters to callables
            filt = params.get("filter")
            if isinstance(filt, str) and filt.lower() in FILTERS:
                params["filter"] = FILTERS[filt.lower()]

            params.pop("name", None)
            handlers.append(params)

        EXTRAS = {"RUN_ID": self.run_id, "GENERATED_UUID": uuid4()}

        extra_cfg: dict[str, Any] = {
            k.lower() if k in EXTRAS else k: EXTRAS[k] if k in EXTRAS else v
            for k, v in (cfg.get("extra") or {}).items()
        }

        return logger.configure(
            handlers=handlers,
            extra=extra_cfg,
        )


output_controller = Outputs()
