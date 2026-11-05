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
from enum import Enum
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


class CalibrationDataConf(TypedDict, total=False):
    """`storage.calibration_data` section. Unpacked into `CalibrationData`, minus `base_path`."""

    file_path: str
    file_type: Literal["yaml", "json"]


class LogConf(TypedDict, total=False):
    """`storage.log_config` section, pointing at the loguru config file."""

    file_dir: str
    file_type: str


class StorageConf(TypedDict, total=False):
    """`storage` section of the calibration config."""

    root_dir: str
    db_manager: DbManagerConf
    calibration_data: CalibrationDataConf
    log_config: LogConf


class CalibrationParameter(Enum):
    "Output parameters that aren't stored in the runcard"

    T1 = "t1"
    T2 = "t2"
    T2_ECHO = "t2_echo"
    SINGLE_QUBIT_GATE_FIDELITY = "1q_gate_fidelity"
    DOUBLE_QUBIT_GATE_FIDELITY = "2q_gate_fidelity"
    READOUT_FIDELITY = "readout_fidelity"


class CalibrationData:
    def __init__(self, base_path: str, file_path: str | None = None, file_type: Literal["yaml", "json"] | None = None):
        if file_path:
            if file_path.startswith(DEFAULT_FOLDER_ALIAS) and file_path:
                file_path = file_path.replace(DEFAULT_FOLDER_ALIAS, base_path, 1)
            self._path = Path(file_path).expanduser().resolve()
        else:
            self._path = None

        if file_type in ["yaml", "json"]:
            self._file_type = file_type
        elif file_path:
            _, _, file_extension = file_path.rpartition(".")
            self._file_type = file_extension if file_extension in ["yml", "json"] else None
        else:
            self._file_type = None

        self._data: dict[str, dict[str, float]] = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def save_file(self):
        if not self._path or not self._file_type:
            return
        with self._path.open("w", encoding="utf-8") as f:
            if self._file_type == "json":
                json.dump(self._data, f, indent=2)
            elif self._file_type in ["yaml", "yml"]:
                yaml.safe_dump(self._data, f, sort_keys=False)

    def load_file(self):
        """Load an existing calibration file into memory, so prior values can satisfy dependencies."""
        if not self._path or not self._file_type or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            loaded = json.load(f) if self._file_type == "json" else yaml.safe_load(f)
        if isinstance(loaded, dict):
            self._data = loaded

    def calibrated_targets(self) -> set[str]:
        """Names of the targets that already have a value in the calibration file."""
        return set(self._data)

    def __contains__(self, key) -> bool:
        return key in self._data

    def _store_parameter(self, parameter: CalibrationParameter, target: list[int] | int, value: float):
        target_str = str(target)
        if target_str not in self._data:
            self._data[target_str] = {}
        self._data[target_str][parameter.value] = float(value)


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
        self._calibration_data: CalibrationData | None = None

        self._db_manager: DatabaseManager | None = None
        self.run: CalibrationRun | SequenceRun | None = None

        self._storage_path: str | None = None

        self.storage_conf: StorageConf = {}

    def reset(self, storage_conf: StorageConf | None = None):
        self._initialize()
        self.storage_conf = storage_conf or {}
        self._initialized = True
        self._storage_path = self._create_storage_path()
        self._db_manager = self._create_db_manager()
        self._calibration_data = self._create_calibration_data()

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
    @_requires_initialized
    def calibration_data(self) -> CalibrationData:
        if self._calibration_data is None:
            self._calibration_data = self._create_calibration_data()
        return self._calibration_data

    def _create_calibration_data(self) -> CalibrationData:
        data_conf = self.storage_conf.get("calibration_data")
        if data_conf:
            data = CalibrationData(base_path=self.storage_path, **data_conf)
        else:
            data = CalibrationData(base_path=self.storage_path)
        # Load any existing calibration file so prior values can satisfy unselected dependencies.
        data.load_file()
        return data

    def store_parameter(self, parameter: CalibrationParameter, target: list[int] | int, value: float):
        """Stores a parameter for later serialization if configured"""
        self.calibration_data._store_parameter(parameter=parameter, target=target, value=value)

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
