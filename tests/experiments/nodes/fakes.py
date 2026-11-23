"""Hardware and stream fakes the fixed-flux node tests share (no instruments, no database)."""

from types import SimpleNamespace

import numpy as np
from qililab.qprogram.crosstalk_matrix import CrosstalkMatrix

#: The flux buses of ``q1`` the identity crosstalk spans.
CROSSTALK_BUSES = ("flux_z_q1", "flux_x_q1")


def identity_crosstalk(buses: tuple[str, ...] = CROSSTALK_BUSES) -> CrosstalkMatrix:
    """An identity crosstalk matrix over ``buses``."""
    return CrosstalkMatrix.from_buses({b: {bb: (1.0 if b == bb else 0.0) for bb in buses} for b in buses})


class FakePlatform:
    """A platform that records parameter writes and answers the calls a fixed-flux node makes.

    ``get_element`` reports a Rohde & Schwarz drive instrument (so the IQ-modulation/POWER/RF branch
    runs) and ``execute_qprogram`` returns a readout result for ``q1``. On execute it snapshots the
    parameters set so far into :attr:`values_at_execute`, before the node's run restores them.
    """

    def __init__(self):
        self.values: dict = {}
        self.calls: list = []
        self.values_at_execute: dict = {}

    def set_crosstalk(self, *, crosstalk):
        self.calls.append("set_crosstalk")

    def set_flux_to_zero(self):
        self.calls.append("set_flux_to_zero")

    def set_bias_to_zero(self):
        self.calls.append("set_bias_to_zero")

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value
        self.calls.append(f"set:{alias}")

    def get_element(self, alias):
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name="ROHDE_SCHWARZ"))])

    def execute_qprogram(self, qprogram, **kwargs):
        self.calls.append("execute_qprogram")
        self.values_at_execute = dict(self.values)
        return SimpleNamespace(results={"readout_q1": [SimpleNamespace(array=np.zeros((2, 2)))]})


class FakeStreamArray:
    """Stand-in for ``StreamArray``: a context manager that swallows the write and carries a measurement id."""

    measurement_id = 321

    def __init__(self, **kwargs):
        self.measurement = SimpleNamespace(measurement_id=self.measurement_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __setitem__(self, key, value):
        return None
