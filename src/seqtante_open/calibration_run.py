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

from loguru import logger
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qililab.data_management import build_platform
from qililab.platform.platform import Platform
from ruamel.yaml import YAML

from seqtante_open.controllers import CalibrationGraph, CalibrationParser
from seqtante_open.outputs import output_controller

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
    """

    def __init__(self, platform_path: str, config_path: str):
        self.platform_path: str = platform_path
        self.config_path: str = config_path

        self.start = time.time()

        # Load YAML
        config: dict = load_config(self.config_path)

        # --- Build base path for data from YAML ---
        storage_conf = config.get("storage", {})

        # Setup the output_controller
        output_controller.reset(storage_conf=storage_conf)
        output_controller.add_calibration_run(
            calibration_tree=config, sample_name=config["sample"], cooldown=config["cooldown"]
        )

        output_controller.setup_logger()
        logger.opt(colors=True).info("Welcome to Seqtante-Open")

        self.platform: Platform = build_platform(runcard=self.platform_path)
        self.graph = CalibrationGraph(self.platform, self.platform_path)

        # --- Compile the calibration tree ---
        compiler = CalibrationParser(
            calibration_config=config,
            platform=self.platform,
            platform_path=platform_path,
            graph=self.graph,
        )
        compiler.compile_nodes()

        # --- Connect to platform ---
        self.platform.connect()
        self.platform.initial_setup()
        self.platform.turn_on_instruments()
        logger.opt(colors=True).info(
            "Executing Calibration. Outputs in folder {path}", path=output_controller.storage_path
        )

        self.graph.run_calibration()
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
        output_controller.end_calibration()
        self.platform.disconnect()
        output_controller.calibration_data.save_file()
