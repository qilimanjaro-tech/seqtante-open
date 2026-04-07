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

"""Automatic calibration script (YAML-driven 1Q/2Q classification via `kind`)."""
import argparse
import time
import warnings

import qililab as ql
from loguru import logger
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
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


def main(platform_path: str, config_path: str):
    start = time.time()

    # Optional ASCII logo

    # TODO: LOGO not used with logs. Decide what to do about it in the future.
    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # logo_path = os.path.join(base_dir, "img", "seqtante_logo.txt")
    # if os.path.exists(logo_path):
    #     with open(logo_path, "r") as f:
    #         print("\n" + f.read())

    # Load YAML
    config: dict = load_config(config_path)

    # --- Build base path for data from YAML ---
    storage_conf = config.get("storage", {})

    # Setup the output_controller
    output_controller.reset(storage_conf=storage_conf)

    output_controller.add_calibration_run(calibration_tree=config, sample_name=config["sample"], cooldown=config["cooldown"])

    output_controller.setup_logger()
    logger.opt(colors=True).info("Welcome to Seqtante")

    platform = ql.build_platform(runcard=platform_path)
    graph = CalibrationGraph(platform, platform_path)

    # --- Compile the calibration tree ---
    compiler = CalibrationParser(
        calibration_config=config,
        platform=platform,
        platform_path=platform_path,
        graph=graph
    )
    compiler.compile_nodes()

    # --- Run the calibration graph in topological order ---
    platform.connect()
    platform.initial_setup()
    platform.turn_on_instruments()
    logger.opt(colors=True).info("Executing Calibration. Outputs in folder {path}", path=output_controller.storage_path)
    graph.run_calibration()

    end = time.time()
    elapsed = end - start

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    logger.opt(colors=True).info("Execution time: <g>{hours}</g>h <g>{minutes}</g>m <g>{seconds}</g>s", hours=hours, minutes=minutes, seconds=f"{seconds:.2f}")

    output_controller.end_calibration()
    platform.disconnect()
    output_controller.calibration_data.save_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic calibration runner")
    parser.add_argument("--platform_path", required=True, help="Path to the runcard or platform configuration")
    parser.add_argument("--config_path", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    main(args.platform_path, args.config_path)