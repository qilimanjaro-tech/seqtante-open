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

import os
import time
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qililab.data_management import build_platform
from qililab.utils.serialization import deserialize_from
from ruamel.yaml import YAML

from seqtante_open.calibration.io import read_calibration_file
from seqtante_open.controllers import CalibrationGraph, CalibrationParser, Pipeline, RunContext
from seqtante_open.outputs import output_controller
from seqtante_open.run_results import RunResults, StepOutcome

if TYPE_CHECKING:
    from qililab.platform.platform import Platform
    from qililab.qprogram.calibration import Calibration

    from seqtante_open.calibration.model import CalibrationFile


def _package_version(package: str) -> str:
    """The installed version of ``package``, or ``"unknown"`` if it cannot be read."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # ruff: ignore[blind-except] -- the version is cosmetic; never fail a run over it
        return "unknown"


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
        calibration (str | None): Path to the run's calibration file (a serialized ``Calibration``,
            ADR-0001). The nodes read and write it, and an unselected dependency is satisfied from it.
            None runs without one, so no unselected dependency is satisfied.
        report (bool): Whether to write the PDF report at the end of the run (ADR-0009). ``--no-report``
            turns it off; with it off the report module is never imported.
    """

    def __init__(
        self,
        platform_path: str,
        config_path: str,
        pipelines: list[str] | None = None,
        calibration: str | None = None,
        report: bool = True,
    ):
        self.platform_path: str = platform_path
        self.config_path: str = config_path
        self.report: bool = report

        self.start = time.time()
        self.started_at = datetime.now()

        # Load YAML
        config: dict = load_config(self.config_path)

        # --- Parse the calibration tree, before anything is written to the database ---
        self.platform: Platform = build_platform(runcard=self.platform_path)
        self.pipelines: list[Pipeline] = CalibrationParser(calibration_config=config, platform=self.platform).parse()

        # Set up the output controller and the logger. The run_id is only known once the run starts
        # (after the platform connects), so the logger is set up again then to carry it.
        output_controller.reset(storage_conf=config.get("storage", {}))
        output_controller.setup_logger()
        logger.opt(colors=True).info("Welcome to Seqtante-Open")

        # --- Load the calibration file once (ADR-0001) and build the run context ---
        # Deserialised here, before the hardware, into the Calibration the nodes share. An unselected
        # dependency is satisfied from its content.
        context, calibration_file = self._load_calibration(calibration)

        # --- Build the graph from the selected pipelines, before touching the hardware ---
        # Unknown/disabled selections and `after` cycles are raised here, before the platform connects.
        self.graph = CalibrationGraph(self.platform, self.platform_path)
        self.graph.build(
            self.pipelines, selected_names=pipelines, data_root=output_controller.storage_path, context=context
        )

        # --- Connect to platform; from here the instruments are on, so close the run out in `finally` ---
        self.platform.connect()
        try:
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
            self.graph.run_calibration(calibration_file)
        finally:
            self._end_calibration()

    def _load_calibration(self, calibration: str | None) -> "tuple[RunContext | None, CalibrationFile | None]":
        """Deserialise ``--calibration`` into the run context and calibration file, or return ``(None, None)``."""
        if calibration is None:
            return None, None
        calibration_obj: Calibration = deserialize_from(calibration)
        context = RunContext(
            platform=self.platform,
            platform_path=self.platform_path,
            calibration=calibration_obj,
            calibration_path=calibration,
        )
        return context, read_calibration_file(calibration_obj, check_pulses=False)

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
        """Close the run out: zero the bias, end the run entry, save the data and disconnect.

        Called in a ``finally`` after ``connect``, so it runs even when the run raised (a DB error in
        ``start_run``, a flux conflict, anything escaping ``run_calibration``). Each step is guarded so
        one failure does not skip the rest, and so closing out never masks the error that caused it.
        """
        self._display_time()
        self._safely("zero the bias", self.platform.set_bias_to_zero)
        self._safely("end the run entry", output_controller.end_run)
        self._safely("save the calibration data", output_controller.calibration_data.save_file)
        self._safely("disconnect the platform", self.platform.disconnect)
        if self.report:
            self._write_report()

    @staticmethod
    def _safely(what: str, action):
        """Run ``action``, logging and swallowing any error so the rest of the close-out still runs."""
        try:
            action()
        except Exception:  # ruff: ignore[blind-except] -- closing out must finish and not mask the run's own error
            logger.opt(exception=True).warning("Could not {what} while ending the run", what=what)

    def _collect_results(self) -> RunResults:
        """Gather the run's in-memory results for the report (ADR-0009).

        The timestamp, versions and run id, plus every target that failed or was skipped with its
        reason (from the graph). Benchmark results, operating points and fit plots join this record as
        the experiment nodes are ported (tasks 6-12); the report renders whatever is present.
        """
        run_id = output_controller.run_id
        results = RunResults(
            timestamp=self.started_at,
            seqtante_open_version=_package_version("seqtante_open"),
            qililab_version=_package_version("qililab"),
            run_id=run_id if run_id != -1 else None,
        )
        for target, step, reason in self.graph.report:
            results.outcomes.append(StepOutcome(target=target, step=step, reason=reason))
        return results

    def _write_report(self) -> None:
        """Write the PDF report, if the ``seqtante_open[report]`` extra is installed (ADR-0009).

        Called after the database entry and the calibration file are saved, so a report failure never
        costs the run its results. A missing extra logs a warning naming the install and the run
        finishes normally; any other report error is logged and swallowed for the same reason.
        """
        try:
            from seqtante_open.report import write_report
        except ImportError:
            logger.opt(colors=True).warning(
                'The PDF report needs the report extra. Install it with: pip install "seqtante_open[report]"'
            )
            return
        try:
            path = os.path.join(output_controller.storage_path, f"report_{self.started_at:%Y-%m-%d_%H-%M-%S}.pdf")
            write_report(self._collect_results(), path)
            logger.opt(colors=True).info("Report written to <g>{path}</g>", path=path)
        except Exception:  # ruff: ignore[blind-except] -- a report failure must not cost the run its saved results
            logger.opt(exception=True).warning("Writing the PDF report failed; the run's results are already saved")
