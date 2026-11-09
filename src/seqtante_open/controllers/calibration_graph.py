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

import os
from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx
from loguru import logger

from seqtante_open.controllers.legacy_node import LegacyNode

if TYPE_CHECKING:
    from seqtante_open.controllers.calibration_tree_parser import Pipeline


class GraphBuildError(ValueError):
    """Raised when the selected pipelines cannot form a runnable acyclic graph."""


class CalibrationGraph:
    """Directed acyclic graph of calibration nodes, run in topological order."""

    def __init__(self, platform, platform_path: str):
        self.graph = nx.DiGraph()
        self.platform = platform
        self.platform_path = platform_path

        # First node of a pipeline -> targets an unselected `after` dependency needs in the
        # calibration file (the dependency step is not run this time; QHC-1800).
        self.unselected_deps: dict[int, set[str]] = defaultdict(set)
        # Why a target did not run: list of (target, node name, reason).
        self.report: list[tuple[str, str, str]] = []

    def add_node(self, node: LegacyNode):
        """Adds a node to the graph"""
        self.graph.add_node(node.idx, data=node, platform=self.platform, platform_path=self.platform_path)

    def add_dependency(self, from_node: int, to_node: int):
        """Adds a node topological dependency to the graph"""
        self.graph.add_edge(from_node, to_node)

    def build(
        self,
        pipelines: "list[Pipeline]",
        selected_names: list[str] | None = None,
        data_root: str | None = None,
    ) -> "CalibrationGraph":
        """Builds nodes and edges from the parsed pipelines, then checks the graph is acyclic.

        Args:
            pipelines (list[Pipeline]): Every parsed pipeline, enabled or not.
            selected_names (list[str] | None): Pipelines to run (``--pipelines``); None runs every enabled one.
            data_root (str | None): Base folder for per-target outputs; None skips the ``data_folder`` wiring.

        Raises:
            GraphBuildError: If a selected pipeline is unknown or disabled, or the pipelines form a cycle.
        """
        by_name = {pipeline.name: pipeline for pipeline in pipelines}
        self._validate_selection(by_name, selected_names)
        running = [
            pipeline
            for pipeline in pipelines
            if pipeline.enabled and (selected_names is None or pipeline.name in selected_names)
        ]

        step_idx, first_idx = self._build_nodes(running, data_root)
        self._build_after_edges(running, step_idx, first_idx)
        self._raise_if_cyclic()
        return self

    @staticmethod
    def _validate_selection(by_name: "dict[str, Pipeline]", selected_names: list[str] | None):
        """Rejects a ``--pipelines`` selection that names an unknown or disabled pipeline."""
        if selected_names is None:
            return
        for name in selected_names:
            if name not in by_name:
                raise GraphBuildError(f"--pipelines: unknown pipeline '{name}'. Known: {sorted(by_name)}.")
            if not by_name[name].enabled:
                raise GraphBuildError(f"--pipelines: pipeline '{name}' is disabled (enabled: false).")

    def _build_nodes(self, running: "list[Pipeline]", data_root: str | None) -> tuple[dict[str, int], dict[str, int]]:
        """Adds one node per step and the intra-pipeline edges (steps run in order).

        Returns:
            step_idx: step name -> node idx, for steps that run this time.
            first_idx: pipeline name -> its first step's node idx.
        """
        step_idx: dict[str, int] = {}
        first_idx: dict[str, int] = {}
        idx = 0
        for pipeline in running:
            previous: int | None = None
            for step in pipeline.steps:
                node = LegacyNode(
                    parameters=step.parameters,
                    platform=self.platform,
                    platform_path=self.platform_path,
                    experiment=step.experiment,
                    name=step.name,
                    idx=idx,
                    targets=step.targets,
                )
                if data_root is not None:
                    for target in step.targets:
                        node.parameters[str(target)]["data_folder"] = os.path.join(data_root, step.name, str(target))
                self.add_node(node)
                step_idx[step.name] = idx
                if previous is None:
                    first_idx[pipeline.name] = idx
                else:
                    self.add_dependency(previous, idx)
                previous = idx
                idx += 1
        return step_idx, first_idx

    def _build_after_edges(self, running: "list[Pipeline]", step_idx: dict[str, int], first_idx: dict[str, int]):
        """Adds inter-pipeline edges from each pipeline's ``after``."""
        for pipeline in running:
            if not pipeline.steps:
                continue
            head = first_idx[pipeline.name]
            head_targets = {str(target) for target in pipeline.steps[0].targets}
            for dependency in pipeline.after:
                if dependency in step_idx:
                    self.add_dependency(step_idx[dependency], head)
                else:
                    # The dependency step is not run this time: satisfy it from the calibration file.
                    self.unselected_deps[head] |= head_targets

    def _raise_if_cyclic(self):
        """Raises if the graph has an ``after`` cycle, naming the pipelines in it."""
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            names = [self.graph.nodes[origin]["data"].name for origin, _ in cycle]
            raise GraphBuildError(f"Pipelines form a cycle through `after`: {' -> '.join([*names, names[0]])}.")

    def run_calibration(self, calibrated_targets: set[str] | None = None):
        """Runs every node in topological order, skipping targets blocked within their pipeline.

        A target that fails (or is skipped) at a node blocks the rest of that target's pipeline and, through
        ``after`` edges, any pipeline that depends on the failed step. Other targets and other pipelines run on.

        Args:
            calibrated_targets (set[str] | None): Targets already present in the calibration file, used to
                satisfy a dependency on a step that is not run this time.
        """
        calibrated_targets = calibrated_targets or set()
        failed_at: dict[int, set[str]] = {}
        skipped_at: dict[int, set[str]] = {}

        for node_idx in nx.topological_sort(self.graph):
            node = self.graph.nodes[node_idx]["data"]
            node_targets = {str(target) for target in node.targets}

            # Targets blocked by a failed or skipped predecessor step.
            blocked: set[str] = set()
            for predecessor in self.graph.predecessors(node_idx):
                blocked |= failed_at.get(predecessor, set()) | skipped_at.get(predecessor, set())
            blocked &= node_targets

            # Targets whose unselected dependency is not satisfied by the calibration file.
            misses = {
                t
                for t in self.unselected_deps.get(node_idx, set())
                if t in node_targets and t not in calibrated_targets
            }
            for target in misses:
                self.report.append((target, node.name, "missing dependency (not run this time, no calibrated value)"))
            blocked |= misses

            dead_targets = dict.fromkeys(blocked, node.name)
            failed = {str(target) for target in node.run(dead_targets)}

            failed_at[node_idx] = failed
            skipped_at[node_idx] = blocked
            for target in failed:
                self.report.append((target, node.name, "failed"))
            for target in blocked - misses:
                self.report.append((target, node.name, "skipped (dependency failed upstream)"))

        self._log_report()

    def _log_report(self):
        """Logs, per target, every node it did not run in and why."""
        for target, node_name, reason in self.report:
            logger.opt(colors=True).info(
                "Target <r>{target}</r> skipped in node <i><fg #8838ff>{node}</></i>: {reason}",
                target=target,
                node=node_name,
                reason=reason,
            )
