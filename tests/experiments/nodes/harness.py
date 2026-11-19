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

"""Reusable building blocks for testing :class:`~seqtante_open.controllers.calibration_node.CalibrationNode`.

A node test drives the real three-phase ``run`` (phase order, operating-point placement, restore and
crosstalk wrapping, benchmark scoring) while stubbing the three boundaries a node touches: the
platform, the experiment driver and the fit. The platform fake and the output/save stubbing are the
same for every node, so they live here; the driver and fit stubs are the node's own and stay in its
test module.

The shape mirrors :mod:`tests.experiments.fitting.harness`: shared fakes and helpers in one place, so a
new node's test is the node-specific stubs plus a few calls here. As more nodes land this can grow a
declarative ``NodeTestCase`` (scripted driver + fit + EXPECTED verdict) in the same vein; for now it is
the toolkit the one real node (``T1Node``, QHC-1803) needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from seqtante_open.controllers import calibration_node as node_module
from seqtante_open.controllers.calibration_node import RunContext
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    import pytest
    from qililab.qprogram.calibration import Calibration


class FakePlatform:
    """A platform stand-in for node tests: records every set/get and the lifecycle calls a run makes.

    Covers the surface ``CalibrationNode.run`` and the operating-point placement reach for (crosstalk
    zeroing, flux/bias reset, ``get_parameter``/``set_parameter``), so no hardware or runcard is needed.
    Parameter writes are remembered in :attr:`values`, and every call name is appended to :attr:`calls`,
    so a test can assert what a node set and in what order.
    """

    def __init__(self, values: dict | None = None) -> None:
        self.values: dict = dict(values or {})
        self.calls: list = []

    def set_crosstalk(self, *, crosstalk: Any) -> None:
        self.calls.append("set_crosstalk")

    def set_flux_to_zero(self) -> None:
        self.calls.append("set_flux_to_zero")

    def set_bias_to_zero(self) -> None:
        self.calls.append("set_bias_to_zero")

    def get_parameter(self, *, alias: str, parameter: Any) -> Any:
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias: str, parameter: Any, value: Any) -> None:
        self.values[alias, parameter] = value
        self.calls.append(f"set:{alias}")


def build_context(
    calibration: Calibration,
    platform: Any | None = None,
    *,
    platform_path: str = "runcard.yml",
    calibration_path: str = "calibration.yml",
) -> RunContext:
    """A :class:`RunContext` wrapping ``calibration`` and a platform (a fresh :class:`FakePlatform` by default)."""
    return RunContext(
        platform=platform if platform is not None else FakePlatform(),
        platform_path=platform_path,
        calibration=calibration,
        calibration_path=calibration_path,
    )


def stub_output_and_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the boundaries every node run touches off the hardware path: the DB and the calibration save.

    Marks the ``output_controller`` initialised with a dummy db manager (so ``db_manager`` is available
    without a real database) and turns ``save_calibration`` into a no-op. A node's own driver and fit
    stubs are layered on top of this in the node's test module.
    """
    monkeypatch.setattr(node_module, "save_calibration", lambda calibration, path: None)
    monkeypatch.setattr(output_controller, "_initialized", True)
    monkeypatch.setattr(output_controller, "_db_manager", object())
