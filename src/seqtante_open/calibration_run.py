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

"""Drives one calibration run end to end: compile the tree, run it, close it out."""

import time
import warnings
from typing import TYPE_CHECKING

from loguru import logger
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qililab.data_management import build_platform
from ruamel.yaml import YAML

from seqtante_open.controllers import CalibrationGraph, CalibrationParser, Pipeline
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    from qililab.platform.platform import Platform

warnings.filterwarnings("ignore", category=QCoDeSDeprecationWarning)
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0*")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout*")


# ------------------------- YAML IO -------------------------


def load_config(config_path):
    yaml = YAML()
    with open(config_path, "r") as f:
        return yaml.load(f)


# ------------------------- Main -------------------------


class CalibrationRun:
    """Compile a calibration tree into a graph and run it against a platform.

    Everything happens in ``__init__``: the run is the object's construction.

    Args:
        platform_path (str): Path to the runcard the platform is built from.
        config_path (str): Path to the YAML calibration tree.
        pipelines (list[str] | None): Names of the pipelines to run; None runs every enabled one.
        calibration (str | None): Path to the calibration file the run reads prior values from;
            overrides ``storage.calibration_data.file_path`` in the tree. None uses the tree's setting.
    """

    def __init__(
        self,
        platform_path: str,
        config_path: str,
        pipelines: list[str] | None = None,
        calibration: str | None = None,
    ):
        self.platform_path: str = platform_path
        self.config_path: str = config_path

        self.start = time.time()

        # Load YAML
        config: dict = load_config(self.config_path)

        # --- Parse the calibration tree, before anything is written to the database ---
        self.platform: Platform = build_platform(runcard=self.platform_path)
        self.pipelines: list[Pipeline] = CalibrationParser(calibration_config=config, platform=self.platform).parse()

        # --- Build base path for data from YAML ---
        storage_conf = config.get("storage", {})
        # A --calibration path overrides the calibration file the run reads prior values from.
        if calibration is not None:
            storage_conf.setdefault("calibration_data", {})["file_path"] = calibration

        # Set up the output controller and the logger. The run_id is only known once the run starts
        # (after the platform connects), so the logger is set up again then to carry it.
        output_controller.reset(storage_conf=storage_conf)
        output_controller.setup_logger()
        logger.opt(colors=True).info("Welcome to Seqtante-Open")

        # --- Build the graph from the selected pipelines, before touching the hardware ---
        # Unknown/disabled selections and `after` cycles are raised here, before the platform connects.
        self.graph = CalibrationGraph(self.platform, self.platform_path)
        self.graph.build(self.pipelines, selected_names=pipelines, data_root=output_controller.storage_path)
        calibrated_targets = output_controller.calibration_data.calibrated_targets()

        # --- Connect to platform ---
        self.platform.connect()
        self.platform.initial_setup()
        self.platform.turn_on_instruments()

        # Start the run entry once connected, then re-setup the logger so its records carry the run_id.
        output_controller.start_run(
            run_tree=config,
            sample_name=config["sample"],
            cooldown=config.get("cooldown"),
            sequence_name=config.get("sequence_name"),
        )
        output_controller.setup_logger()

        logger.opt(colors=True).info(
            "Executing Calibration. Outputs in folder {path}", path=output_controller.storage_path
        )

        self.graph.run_calibration(calibrated_targets)
        self._end_calibration()

    def _display_time(self):
        """Measure run time."""
        elapsed = time.time() - self.start

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60

        logger.opt(colors=True).info(
            "Execution time: <g>{hours}</g>h <g>{minutes}</g>m <g>{seconds}</g>s",
            hours=hours,
            minutes=minutes,
            seconds=f"{seconds:.2f}",
        )

    def _end_calibration(self):
        """End the calibration run and disconnect from platform."""
        self._display_time()
        output_controller.end_run()
        self.platform.disconnect()
        output_controller.calibration_data.save_file()
