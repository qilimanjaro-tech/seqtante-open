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
"""Automatic-calibration Node module."""

import os
from copy import copy
from typing import Any, Dict

from loguru import logger
from qililab import Platform
from seqtante_open.experiments.experiment_functions import double_qubit_exp, experiment_functions_dict, single_qubit_exp


class CalibrationNode:
    """This class stores all the information needed for the execution of experiments though the CalibrationGraph.
        It also checks the validity of certain parameters. If they aren't, blocks the execution of that qubit.

        Args:
            platform (qililab.Platform): Platform used for the node execution.
            platform_path (str): Path were the runcard is stored, to update the parameters calibrated in the node.
            idx (int): Identifier of the node for the graph.
            experiment (str): Name of the type of experiment being run.
            name (str): Name the experiment has in the pipeline, for representation.
            targets (list): Targets (qubits or couplings) were this experiment will run.
            simultaneous (bool | list | list[list], optional): Parameter that specifies if parallel calibration of qubits is done and in what way. Defaults to False.
        """

    def __init__(
        self,
        parameters: dict,
        platform: Platform,
        platform_path: str,
        experiment: str,
        name: str,
        idx: int,
        targets: list,
        simultaneous: bool | list[list[list | int]] = False,
    ) -> None:
        self.platform = platform
        self.platform_path = platform_path
        self.name = name
        self.idx = idx
        self.targets = [self._to_valid_target(target) for target in targets]
        self.experiment = experiment
        self.experiment_func = experiment_functions_dict.get(experiment, None)  # Callable(platform, platform_path, parameters) -> Any
        self.parameters: Dict[tuple | str | int, Any] = copy(parameters)

        if isinstance(simultaneous, list) and False:  # TODO: support parallelization in fluxonium experiments in the future if possible
            if all(isinstance(sim, list) for sim in simultaneous):
                self.simultaneous = [[self._to_valid_target(target) for target in sim] for sim in simultaneous]
                unexpected_sim_targets = {st for sim in simultaneous for st in sim if st not in targets}
                if unexpected_sim_targets:
                    logger.opt(colors=True).warning("Found the target/s <r>{target}</r> in simultaneous that weren't included in targets for node <i><fg #8838ff>{node}</></i>. (Targets unused)",
                                        target=unexpected_sim_targets, node=self.name)
            else:
                logger.opt(colors=True).warning("Invalid type for simultaneous on node <r>{node}</r>. Simultaneous has to be a list of lists of targets. "
                "(Running node without parallelization)", node=self.name)
                self.simultaneous = False
        elif simultaneous and False:
            self.simultaneous = [self.targets]
        else:
            self.simultaneous = []

        self._valid_node: Dict[str, bool] = dict.fromkeys(self.targets, True)
        self._validate_node()

        self.result: dict = {}

    # ------------------------- dict behavior -------------------------

    def __setitem__(self, key, value):
        self.parameters[self._to_valid_target(key)] = value

    def __getitem__(self, key):
        return self.parameters[self._to_valid_target(key)]

    def __delitem__(self, key):
        vk = self._to_valid_target(key)
        if vk in self.parameters:
            del self.parameters[vk]

    def add_prev_results(self, results: dict[int | tuple, dict]):
        for target, res in results.items():
            if target in self.parameters:
                self.parameters[target].update(res)
                continue
            self.parameters[target] = res

    @staticmethod
    def _to_valid_target(key):
        return tuple(key) if isinstance(key, list) else key

    # ------------------------- Normalization helpers -------------------------

    def _validate_node(self):
        if not self.experiment_func:
            logger.opt(colors=True).warning("Experiment <i><fg #8838ff>{experiment}</></i> not implemented. (Node Skipped)", experiment=self.experiment)
            self._valid_node = dict.fromkeys(self.targets, False)

        if self.experiment in single_qubit_exp:
            target_clss = int
        elif self.experiment in double_qubit_exp:
            target_clss = tuple
        else:
            target_clss = None.__class__

        for target in self.targets:
            if not isinstance(target, target_clss):
                logger.opt(colors=True).warning("Invalid target type for target <r>{target}</r> detected for experiment <i><fg #8838ff>{experiment}</></i>. "
                                "All targets should be <blue>{clss}</>. (Target Skipped)",
                                target=target, experiment=self.experiment, clss=target_clss.__name__)
                self._valid_node[target] = False

    # ----------------------------- Execution -----------------------------
    def run(self, dead_targets: list) -> list:
        """Using the context (targets that have stopped execution, invalid targets and the simultaneous parameter),
        determines what targets are executed, in wat order an how parallelized and passes the information to run targets for execution.

        Args:
            dead_targets (list): Targets that have been interrupted mid calibration. This interruption, caused by some error in the execution.

        Returns:
            list: Invalid targets and targets that have triggered a exception
        """
        targets = {target for target in self.targets if not CalibrationNode._is_calibration_dead(target, dead_targets)}
        target_error: list = [target for target in targets if not self._valid_node[target] or not self.parameters[target]]
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

    def run_targets(self, targets: list[int | tuple]):
        """Executes the experiment function on the targets and stops the calibration of the targets if an exception is triggered.

        Args:
            targets (list[int  |  tuple]): list of the targets to execute at the same time.

        Returns:
            list[int  |  tuple]: Returns the targets if an error has been triggered during execution.
        """
        parameters = self.parameters
        parameters["targets"] = targets
        target_error: list = []
        if not targets:
            return target_error
        if len(targets) > 1:
            logger.opt(colors=True).info("Running experiment '<i><fg #8838ff>{name}</></i>' simultaneously on <r>{target}</>", name=self.name, target=targets)
        else:
            logger.opt(colors=True).info("Running experiment '<i><fg #8838ff>{name}</></i>' on <r>{target}</>", name=self.name, target=targets)
        for target in targets:
            os.makedirs(self.parameters[target]["data_folder"])
        try:
            result = self.experiment_func(
                platform=self.platform,
                platform_path=self.platform_path,
                parameters=parameters,
                )
            if isinstance(result, dict):
                for target in targets:
                    self.result[target] = result
        except Exception as e:
            logger.opt(exception=True).warning(str(e.__class__.__name__) + " " + str(e))
            logger.opt(colors=True).warning("Skipping calibration of target/s <r>{target}</>", target=targets)
            target_error = targets
        return target_error

    @staticmethod
    def _is_calibration_dead(target: int | tuple, dead_targets: dict) -> bool:
        """Looks if the calibration of the target has been interrupted"""
        if target in dead_targets or (isinstance(target, tuple) and any(qubit in dead_targets for qubit in target)):
            return True
        return False
