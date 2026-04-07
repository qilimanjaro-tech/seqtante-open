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

import os
from functools import reduce
from typing import Any

from loguru import logger
from qililab import Platform

from seqtante_open.controllers import CalibrationGraph, CalibrationNode
from seqtante_open.outputs import output_controller


class CalibrationParser:
    def __init__(self, calibration_config: Any, platform: Platform, platform_path: str, graph: CalibrationGraph,):
        """Class that parses the configuration and compiles it into nodes.

        Args:
            calibration_config (Any): The configuration from the yaml.
            base_folder (str): Folder were all the calibration info will be stored.
            platform (Platform): Platform were all the nodes will be executed.
            platform_path (str): Path of the runcard to be updated.
            graph (CalibrationGraph): Graph were the resulting nodes are stored.
        """
        self.calibration_config = calibration_config
        self.platform = platform
        self.platform_path = platform_path
        self.base_folder = output_controller.storage_path

        self.target_num_experiment: dict = {}
        self.graph = graph

    @staticmethod
    def _fuse_dicts(*dicts: dict) -> dict:
        """Takes multiple dicts and reduces them by updating an empty dict with all the input dicts in order"""
        return reduce(dict.__or__, dicts, {})

    @staticmethod
    def _get_overwrites(target: int | list[int], overwrites: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Returns the overwrite if the target is the target of any overwrite, if not it returns None.
        Fuses overwrites, if a parameter is overwritten multiple times only the last instance of that parameter is used"""
        overwrite_list = []
        for overwrite in overwrites:
            if target in overwrite["targets"]:
                overwrite_list.append(overwrite)
        return CalibrationParser._fuse_dicts(*overwrite_list)

    @staticmethod
    def _make_experiment_path(base_folder: str, target: int | tuple, name: str) -> str:
        """Generates the path to save the .h5 and plots. The directory is made in the node execution."""
        if isinstance(target, int):
            q_path = os.path.join(base_folder, f"q_{target}", f"{name}")
        elif isinstance(target, list) and len(target) == 2:
            q_path = os.path.join(base_folder, f"cq_{target[0]}_tq_{target[1]}", f"{name}")
        elif isinstance(target, list):
            q_path = os.path.join(base_folder, f"qubits_{target}", f"{name}")
        elif isinstance(target, str):
            q_path = os.path.join(base_folder, f"{target}", f"{name}")
        return q_path

    def _target_exp_num(self, target: int):
        num = self.target_num_experiment.get(str(target), 0) + 1
        self.target_num_experiment[str(target)] = num
        return num

    def compile_nodes(self):
        """This function takes the configuration and sequentially makes a CalibrationNode with the necessary information and complete parameters per target per node.
        """
        idx = 1
        for node_type in self.calibration_config["node_types"]:
            node_type_params: dict = self.calibration_config[node_type]
            for pipeline in node_type_params["pipelines"]:
                targets = pipeline["targets"]
                experiments: list[str] = pipeline["pipeline"]
                pipeline_simultaneous = pipeline.get("simultaneous", False)

                for experiment in experiments:
                    ex_base_name, _, is_dif = experiment.partition("-")
                    base_params: dict = node_type_params.get(ex_base_name, {}) if is_dif else {}
                    experiment_params: dict = CalibrationParser._fuse_dicts(base_params, node_type_params.get(experiment, {}))

                    if not any((base_params, experiment_params)):
                        logger.opt(colors=True).warning("No references found for <i><fg #8838ff>{experiment}</></i> or the base experiment in the calibration tree for targets <r>{targets}</r> (skipped).",
                            experiment=experiment, targets=targets)
                        continue

                    simultaneous = experiment_params.get("simultaneous", base_params.get("simultaneous", pipeline_simultaneous))

                    # Create the node. The idx is just the execution order of the nodes.
                    experiment_node = CalibrationNode(
                        parameters=experiment_params,
                        platform=self.platform,
                        platform_path=self.platform_path,
                        targets=targets,
                        experiment=ex_base_name,
                        idx=idx,
                        simultaneous=simultaneous,
                        name=experiment
                    )
                    del experiment_node["overwrites"]

                    # Create the parameters dict per target.
                    for target in targets:
                        target_params = CalibrationParser._get_overwrites(
                            target=target,
                            overwrites=[*base_params.get("overwrites", []), *experiment_params.get("overwrites", [])]
                        )
                        target_params.pop("targets", None)

                        target_params["data_folder"] = CalibrationParser._make_experiment_path(
                            base_folder=self.base_folder,
                            target=target,
                            name=f"{self._target_exp_num(target=target)}-{experiment}"
                        )

                        # Add the parameters to the calibration node
                        experiment_node[target] = target_params

                    self.graph.add_node(experiment_node)
                    idx += 1