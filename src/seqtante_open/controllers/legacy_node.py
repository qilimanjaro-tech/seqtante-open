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

"""The legacy-function adapter the graph runs during the port (ADR-0007).

During the port not every experiment is a :class:`~seqtante_open.controllers.calibration_node.CalibrationNode`
subclass yet; the ones still implemented as functions in
:data:`seqtante_open.experiments.experiment_functions.experiment_functions_dict` are driven by this
adapter. :class:`LegacyNode` satisfies the same ``GraphNode`` contract (``idx``, ``name``, ``run``) as
a real node without subclassing the base, so :class:`~seqtante_open.controllers.calibration_graph.CalibrationGraph`
drives both the same way. As each experiment is ported to a node class it stops going through here.
"""

import os
from collections.abc import Sequence
from copy import copy
from typing import Any

from loguru import logger
from qililab import Platform

from seqtante_open.experiments.experiment_functions import ExperimentFunction, experiment_functions_dict
from seqtante_open.targets import ExperimentTarget

#: A target as its name ("q1", "c1_2"), either a plain ``str`` or an ``ExperimentTarget`` (ADR-0001).
TargetLike = str | ExperimentTarget


class LegacyNode:
    """Drives one legacy-function experiment for the graph during the port (ADR-0007).

    Stores everything the execution of a legacy experiment needs and checks the validity of its
    parameters; an invalid target or node is skipped rather than run. Satisfies the ``GraphNode``
    contract (``idx``, ``name``, ``run``) so the graph treats it like a real node.

    Args:
        platform (qililab.Platform): Platform used for the node execution.
        platform_path (str): Path were the runcard is stored, to update the parameters calibrated in the node.
        idx (int): Identifier of the node for the graph.
        experiment (str): Name of the type of experiment being run.
        name (str): Name the experiment has in the pipeline, for representation.
        targets (list[str]): Targets (qubits or couplings) were this experiment will run, e.g. ``"q1"``, ``"c1_2"``.
        simultaneous (bool | list | list[list], optional): Parameter that specifies if parallel calibration of qubits is done and in what way. Defaults to False.
    """

    experiment_func: ExperimentFunction

    def __init__(
        self,
        parameters: dict,
        platform: Platform,
        platform_path: str,
        experiment: str,
        name: str,
        idx: int,
        targets: Sequence[TargetLike],
        simultaneous: bool | list[list[TargetLike]] = False,
    ) -> None:
        self.platform = platform
        self.platform_path = platform_path
        self.name = name
        self.idx = idx
        self.targets = list(targets)
        self.experiment = experiment
        self.parameters: dict[Any, Any] = copy(parameters)

        if isinstance(simultaneous, list):
            if all(isinstance(sim, list) for sim in simultaneous):
                self.simultaneous: list[list[TargetLike]] = [list(sim) for sim in simultaneous]
                unexpected_sim_targets = {st for sim in self.simultaneous for st in sim if st not in self.targets}
                if unexpected_sim_targets:
                    logger.opt(colors=True).warning(
                        "Found the target/s <r>{target}</r> in simultaneous that weren't included in targets for node <i><fg #8838ff>{node}</></i>. (Targets unused)",
                        target=unexpected_sim_targets,
                        node=self.name,
                    )
            else:
                logger.opt(colors=True).warning(
                    "Invalid type for simultaneous on node <r>{node}</r>. Simultaneous has to be a list of lists of targets. "
                    "(Running node without parallelization)",
                    node=self.name,
                )
                self.simultaneous = []
        elif simultaneous:
            self.simultaneous = [self.targets]
        else:
            self.simultaneous = []

        self._valid_node: dict[TargetLike, bool] = dict.fromkeys(self.targets, True)
        self._validate_node()

        self.result: dict = {}

    # ------------------------- dict behavior -------------------------

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __getitem__(self, key):
        return self.parameters[key]

    def __delitem__(self, key):
        if key in self.parameters:
            del self.parameters[key]

    # ------------------------- Normalization helpers -------------------------

    def _validate_node(self):
        if self.experiment in experiment_functions_dict:
            self.experiment_func = experiment_functions_dict[self.experiment]
        else:
            logger.opt(colors=True).warning(
                "Experiment <i><fg #8838ff>{experiment}</></i> not implemented. (Node Skipped)",
                experiment=self.experiment,
            )
            self._valid_node = dict.fromkeys(self.targets, False)

        # A target is its name ("q1", "c1_2"), as a str or an ExperimentTarget (ADR-0001).
        for target in self.targets:
            if not isinstance(target, (str, ExperimentTarget)):
                logger.opt(colors=True).warning(
                    "Invalid target type for target <r>{target}</r> detected for experiment <i><fg #8838ff>{experiment}</></i>. "
                    "A target must be a qubit or coupler name, as a str or ExperimentTarget. (Target Skipped)",
                    target=target,
                    experiment=self.experiment,
                )
                self._valid_node[target] = False

    # ----------------------------- Execution -----------------------------
    def run(self, dead_targets: dict[str, str]) -> list[TargetLike]:
        """Using the context (targets that have stopped execution, invalid targets and the simultaneous parameter),
        determines what targets are executed, in wat order an how parallelized and passes the information to run targets for execution.

        Args:
            dead_targets (dict[str, str]): Targets that have been interrupted mid calibration, mapped to the node that interrupted them. This interruption, caused by some error in the execution.

        Returns:
            list[str]: Invalid targets and targets that have triggered a exception
        """
        targets = {target for target in self.targets if not LegacyNode._is_calibration_dead(target, dead_targets)}
        target_error: list = [
            target for target in targets if not self._valid_node[target] or not self.parameters[target]
        ]
        targets.difference_update(target_error)

        if not targets:
            logger.opt(colors=True).info("No valid targets for node: <i><fg #8838ff>{name}</></i>", name=self.name)
            return target_error

        logger.opt(colors=True).info("Running node: <i><fg #8838ff>{name}</></i>", name=self.name)
        if self.simultaneous:
            for sim_targets in self.simultaneous:
                sim_targets = [target for target in sim_targets if target in targets]
                targets.difference_update(sim_targets)
                if sim_targets:
                    target_error += self.run_targets(sim_targets)
        for target in targets:
            target_error += self.run_targets([target])

        return target_error

    def run_targets(self, targets: list[TargetLike]) -> list[TargetLike]:
        """Executes the experiment function on the targets and stops the calibration of the targets if an exception is triggered.

        Args:
            targets (list[str]): list of the targets to execute at the same time.

        Returns:
            list[str]: Returns the targets if an error has been triggered during execution.
        """
        parameters = self.parameters
        parameters["targets"] = targets
        target_error: list = []
        if not targets:
            return target_error
        if len(targets) > 1:
            logger.opt(colors=True).info(
                "Running experiment '<i><fg #8838ff>{name}</></i>' simultaneously on <r>{target}</>",
                name=self.name,
                target=targets,
            )
        else:
            logger.opt(colors=True).info(
                "Running experiment '<i><fg #8838ff>{name}</></i>' on <r>{target}</>", name=self.name, target=targets
            )
        for target in targets:
            os.makedirs(self.parameters[target]["data_folder"])
        try:
            result = self.experiment_func(
                platform=self.platform,
                platform_path=self.platform_path,
                parameters=parameters,
            )
            if isinstance(result, dict):
                self.result.update(result)
        except Exception as e:  # ruff: ignore[blind-except]
            logger.opt(exception=True).warning(str(e.__class__.__name__) + " " + str(e))
            logger.opt(colors=True).warning("Skipping calibration of target/s <r>{target}</>", target=targets)
            target_error = targets
        return target_error

    @staticmethod
    def _is_calibration_dead(target: TargetLike, dead_targets: dict[str, str]) -> bool:
        """Looks if the calibration of the target has been interrupted"""
        return target in dead_targets
