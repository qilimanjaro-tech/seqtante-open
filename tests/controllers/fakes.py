"""A platform fake the drive-source and operating-point tests share (no instruments)."""

from types import SimpleNamespace


class InstrumentPlatform:
    """A platform whose buses report ``instrument`` and that records parameter writes."""

    def __init__(self, instrument: str):
        self.instrument = instrument
        self.values: dict = {}

    def get_parameter(self, *, alias, parameter):
        return self.values.get((alias, parameter), 0.0)

    def set_parameter(self, *, alias, parameter, value):
        self.values[alias, parameter] = value

    def get_element(self, alias):
        return SimpleNamespace(instruments=[SimpleNamespace(name=SimpleNamespace(name=self.instrument))])
