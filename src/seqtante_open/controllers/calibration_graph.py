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

"""Automatic-calibration Graph module."""

from typing import TYPE_CHECKING

import networkx as nx
from loguru import logger

if TYPE_CHECKING:
    from seqtante_open.controllers.calibration_node import CalibrationNode


class CalibrationGraph:
    """Directed acyclic graph of calibration nodes, run in topological order."""

    def __init__(self, platform, platform_path: str):
        self.graph = nx.DiGraph()
        self.platform = platform
        self.platform_path = platform_path

        self.dead_targets: dict = {}

    def add_node(self, node: "CalibrationNode"):
        """Adds a node to the graph"""
        self.graph.add_node(node.idx, data=node, platform=self.platform, platform_path=self.platform_path)

    def add_dependency(self, from_node: str, to_node: str):
        """Adds a node topological dependency to the graph"""
        self.graph.add_edge(from_node, to_node)

    def run_calibration(self):
        """Runs all the nodes in order and interrupts faulty calibrations"""
        for node_idx in nx.topological_sort(self.graph):
            node = self.graph.nodes[node_idx]["data"]

            dead_targets = node.run(self.dead_targets)

            if dead_targets:
                self.dead_targets.update(dict.fromkeys(dead_targets, node.name))

        for target, node_name in self.dead_targets.items():
            logger.opt(colors=True).info(
                "Exception triggered for <r>{target}</r> in node <i><fg #8838ff>{node_name}</></i>",
                target=target,
                node_name=node_name,
            )
