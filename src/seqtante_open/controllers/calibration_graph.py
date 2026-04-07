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

from typing import TYPE_CHECKING

import networkx as nx
from loguru import logger

if TYPE_CHECKING:
    from seqtante_open.controllers.calibration_node import CalibrationNode


class CalibrationGraph:
    def __init__(self, platform, platform_path):
        """It's a graph that stores and orders all the calibration nodes and transmits information between them.

        Args:
            platform (Platform): Platform where all the experiments are executed on.
            platform_path (str): Path were the runcard is stored, to update the parameters calibrated in a node.
        """
        self.graph = nx.DiGraph()
        self.platform = platform
        self.platform_path = platform_path

        self.dead_targets = {}

    def add_node(self, node: "CalibrationNode"):
        """Adds a node to the graph"""
        self.graph.add_node(node.idx, data=node,
                            platform=self.platform,
                            platform_path=self.platform_path)

    def add_dependency(self, from_node: str, to_node: str):
        self.graph.add_edge(from_node, to_node)

    def run_calibration(self):
        """Runs all the nodes in order and interupts faulty calibrations"""
        prev_node = None
        for node_idx in nx.topological_sort(self.graph):
            node = self.graph.nodes[node_idx]["data"]
            if prev_node:
                node.add_prev_results(prev_node.results)

            dead_targets = node.run(self.dead_targets)

            if dead_targets:
                self.dead_targets.update(dict.fromkeys(dead_targets, node.name))

            prev_node = node

        for target, node_name in self.dead_targets.items():
            logger.opt(colors=True).info("Exception triggered for <r>{target}</r> in node <i><fg #8838ff>{node_name}</></i>", target=target, node_name=node_name)