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
import traceback
import warnings
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger
from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qililab.data_management import build_platform
from qililab.utils.serialization import deserialize_from
from ruamel.yaml import YAML

from seqtante_open.calibration.io import read_calibration_file
from seqtante_open.controllers import CalibrationGraph, CalibrationParser, Pipeline, RunContext, experiment_spec
from seqtante_open.outputs import output_controller
from seqtante_open.run_results import BenchmarkRow, RunResults, StepOutcome

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
        # Set when the run raises, so the close-out can tell the report the results are partial and
        # end the database entry as not completed (with the traceback) instead of as a clean run.
        self.aborted: bool = False
        self.abort_traceback: str | None = None

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
        except BaseException:
            # The run did not finish -- an error or a manual Ctrl-C (KeyboardInterrupt is a BaseException,
            # not an Exception, so it is caught here and re-raised). Record it so the report says so
            # rather than reporting a clean run.
            self.aborted = True
            self.abort_traceback = traceback.format_exc()
            raise
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
        # Pass the abort's traceback on, so qililab records the run as not completed (it marks a run
        # completed only when no traceback is given) and qompass does not show an aborted run as clean.
        self._safely(
            "end the run entry", lambda: output_controller.end_run(traceback=getattr(self, "abort_traceback", None))
        )
        self._safely("disconnect the platform", self.platform.disconnect)
        if self.report:
            self._safely("write the PDF report", self._write_report)

    @staticmethod
    def _safely(what: str, action):
        """Run ``action``, logging and swallowing any error so the rest of the close-out still runs."""
        try:
            action()
        except Exception:  # ruff: ignore[blind-except] -- closing out must finish and not mask the run's own error
            logger.opt(exception=True).warning("Could not {what} while ending the run", what=what)

    def _collect_results(self) -> RunResults:
        """Gather the run's in-memory results for the report (ADR-0009).

        The timestamp, versions and run id; whether the run was aborted; every benchmark result the
        nodes collected (value, uncertainty, verdict, thresholds and a fit plot), including benchmarks
        measured but not scored; each recalibration node's fit plots, per target; and every target that
        failed or was skipped with its reason (from the graph). Operating points and the reset-point
        move log join this record as the remaining experiment nodes are ported; the report renders
        whatever is present.
        """
        run_id = output_controller.run_id
        results = RunResults(
            timestamp=self.started_at,
            seqtante_open_version=_package_version("seqtante_open"),
            qililab_version=_package_version("qililab"),
            run_id=run_id if run_id != -1 else None,
            aborted=getattr(self, "aborted", False),
        )
        self._collect_benchmarks(results)
        self._collect_fit_plots(results)
        for target, step, reason in self.graph.report:
            results.outcomes.append(StepOutcome(target=target, step=step, reason=reason))
        return results

    def _collect_benchmarks(self, results: RunResults) -> None:
        """Turn each node's ``benchmark_results`` into the report's benchmark rows (ADR-0002, ADR-0009).

        Walks the graph's nodes in order; a benchmark node collects a :class:`BenchmarkResult` per
        target on :attr:`~seqtante_open.controllers.calibration_node.CalibrationNode.benchmark_results`
        as it runs. The value, uncertainty, verdict, thresholds and unit all come from the result (the
        run scored it against the thresholds, whose unit #38 checks against the experiment's spec); the
        fit plot, when one was produced, from the node's ``analysis_plots``. A benchmark target measured
        but not scored (no thresholds, no value) becomes a "Not evaluated" row rather than dropping out.
        Legacy function nodes carry no benchmark results and contribute nothing.
        """
        for node in self._graph_nodes():
            benchmark_results = getattr(node, "benchmark_results", ())
            unscored = getattr(node, "unscored_benchmarks", ())
            if not benchmark_results and not unscored:
                continue
            experiment = getattr(node, "experiment", None)
            plots_by_target = getattr(node, "analysis_plots", {})
            for benchmark in benchmark_results:
                plots = plots_by_target.get(benchmark.target, ())
                results.benchmarks.append(
                    BenchmarkRow(
                        target=benchmark.target,
                        experiment=experiment or node.name,
                        value=benchmark.value,
                        uncertainty=benchmark.uncertainty,
                        verdict=benchmark.verdict,
                        passed=benchmark.thresholds.passed,
                        warning=benchmark.thresholds.warning,
                        unit=benchmark.thresholds.unit,
                        direction=benchmark.thresholds.direction,
                        plot=plots[0] if plots else None,
                    )
                )
            for skipped in unscored:
                plots = plots_by_target.get(skipped.target, ())
                spec = experiment_spec(experiment) if experiment is not None else None
                results.benchmarks.append(
                    BenchmarkRow(
                        target=skipped.target,
                        experiment=experiment or node.name,
                        value=skipped.value,
                        uncertainty=skipped.uncertainty,
                        verdict=None,
                        passed=None,
                        warning=None,
                        unit=spec.benchmark.unit if spec is not None and spec.benchmark is not None else None,
                        direction=None,
                        plot=plots[0] if plots else None,
                        note=skipped.reason,
                    )
                )

    def _collect_fit_plots(self, results: RunResults) -> None:
        """Put each recalibration node's fit plots on its targets' per-target section (ADR-0009).

        A benchmark's plot is already shown with its row in the benchmark table, so only non-benchmark
        nodes contribute here, in graph order, so a target's plots read in the order its steps ran.
        """
        for node in self._graph_nodes():
            if getattr(node, "is_benchmark", False):
                continue
            for target, plots in getattr(node, "analysis_plots", {}).items():
                results.target(target).fit_plots.extend(plots)

    def _graph_nodes(self) -> list:
        """The graph's node objects in order, or none when the graph has not been built."""
        graph = getattr(self.graph, "graph", None)
        if graph is None:
            return []
        return [graph.nodes[node_idx]["data"] for node_idx in sorted(graph.nodes)]

    def _write_report(self) -> None:
        """Write the PDF report, if the ``seqtante_open[report]`` extra is installed (ADR-0009).

        Called after the database entry and the calibration file are saved, so a report failure never
        costs the run its results. A genuinely missing extra (``reportlab`` or ``PIL`` not importable)
        logs a warning naming the install and the run finishes normally; any other import failure -- a
        bug in ``pdf.py``, a platypus symbol that has moved, or a newer reportlab raising e.g. a
        ``TypeError`` while ``pdf.py`` builds its styles -- logs its traceback instead of wrongly telling
        the user to install an extra they already have. It is caught as ``Exception``, not only
        ``ImportError``: this runs in the run's ``finally``, where an escaping error would end a clean
        run with a traceback or replace an aborted run's real error. Any other report error is logged
        and swallowed for the same reason.
        """
        try:
            from seqtante_open.report import write_report
        except Exception as exc:  # ruff: ignore[blind-except] -- see the docstring: must not escape the close-out
            if isinstance(exc, ImportError) and exc.name in {"reportlab", "PIL"}:
                logger.opt(colors=True).warning(
                    'The PDF report needs the report extra. Install it with: pip install "seqtante_open[report]"'
                )
            else:
                logger.opt(exception=True).warning(
                    "The PDF report could not be imported (not a missing extra); the run's results are already saved"
                )
            return
        try:
            path = os.path.join(output_controller.storage_path, f"report_{self.started_at:%Y-%m-%d_%H-%M-%S}.pdf")
            write_report(self._collect_results(), path)
            logger.opt(colors=True).info("Report written to <g>{path}</g>", path=path)
        except Exception:  # ruff: ignore[blind-except] -- a report failure must not cost the run its saved results
            logger.opt(exception=True).warning("Writing the PDF report failed; the run's results are already saved")
