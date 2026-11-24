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
declarative ``NodeTestCase`` (scripted driver + fit + EXPECTED verdict) in the same vein. For now it is
the toolkit every node test shares (QHC-1932):

- :class:`FakePlatform`: records writes and lifecycle calls.
- :class:`ExecutingPlatform`: also answers ``get_element`` for the buses a test declares, and
  ``execute_qprogram`` with the readout shape the test's :meth:`~ExecutingPlatform.results` returns.
- :class:`FakeStreamArray`: stands in for ``StreamArray``, recording how it was built and what was written.
- :func:`identity_crosstalk`: the crosstalk matrix the operating-point placement needs.
- :func:`build_context` and :func:`stub_output_and_save`: the run context and the DB/save stubs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

from seqtante_open.controllers import calibration_node as node_module
from seqtante_open.controllers.calibration_node import RunContext
from seqtante_open.outputs import output_controller

if TYPE_CHECKING:
    import pytest
    from qililab.qprogram.calibration import Calibration


#: The flux buses of ``q1`` the identity crosstalk spans by default.
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")


def identity_crosstalk(buses: tuple[str, ...] = CROSSTALK_BUSES) -> CrosstalkMatrix:
    """An identity crosstalk matrix over ``buses``: each flux line drives only itself."""
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


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


class ExecutingPlatform(FakePlatform):
    """A :class:`FakePlatform` that also answers what a node running its own QProgram calls.

    ``get_element`` answers only for the buses declared in ``instruments`` (alias to instrument name,
    e.g. ``{"drive_q1": "ROHDE_SCHWARZ"}``) and raises for any other, so a node that asks for the wrong
    bus fails the test instead of getting a plausible instrument back. ``execute_qprogram`` counts the
    executions and snapshots the parameters
    set so far into :attr:`values_at_execute`, before the run restores them, then returns
    ``SimpleNamespace(results=self.results())``. A node test subclasses it and overrides :meth:`results`
    with the readout shape its QProgram produces.
    """

    def __init__(self, values: dict | None = None, *, instruments: dict[str, str]) -> None:
        super().__init__(values)
        self.instruments = dict(instruments)
        self.values_at_execute: dict = {}
        self.executions = 0

    def get_element(self, alias: str) -> Any:
        self.calls.append(f"get_element:{alias}")
        if alias not in self.instruments:
            raise KeyError(f"no instrument declared for bus {alias!r}; declared: {sorted(self.instruments)}")
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name=self.instruments[alias]))])

    def execute_qprogram(self, qprogram: Any, **kwargs: Any) -> Any:
        self.calls.append("execute_qprogram")
        self.executions += 1
        self.values_at_execute = dict(self.values)
        return SimpleNamespace(results=self.results())

    def results(self) -> dict:
        """The ``{readout bus: acquisitions}`` one execution returns; the node test sets the shape."""
        raise NotImplementedError("override results() with the readout shape the node's QProgram produces")


class FakeStreamArray:
    """Stand-in for ``StreamArray``: records how it was built and every write, and carries a measurement id.

    The node module's ``StreamArray`` is patched with this class. :attr:`last` is the most recent instance,
    so a test can read the loops and shape the node declared and the data it streamed. Reset
    :attr:`last` (and set :attr:`measurement_id`) per test with ``monkeypatch``.
    """

    measurement_id: ClassVar[int] = 100
    last: ClassVar[FakeStreamArray | None] = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.writes: list[tuple[Any, Any]] = []
        self.measurement = SimpleNamespace(measurement_id=type(self).measurement_id)
        type(self).last = self

    def __enter__(self) -> FakeStreamArray:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __setitem__(self, key: Any, value: Any) -> None:
        self.writes.append((key, value))


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
