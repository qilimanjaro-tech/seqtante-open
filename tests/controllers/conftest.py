"""Shared fixtures for the controller tests."""

import pytest

from seqtante_open.controllers import node_registry


@pytest.fixture(autouse=True)
def restore_registry():
    """Snapshot ``NODE_REGISTRY`` and restore it, so a test that defines a node class does not leak it.

    Defining a ``CalibrationNode`` subclass registers it in the module-global registry; without this
    the entry would survive into other tests.
    """
    saved = dict(node_registry.NODE_REGISTRY)
    try:
        yield
    finally:
        node_registry.NODE_REGISTRY.clear()
        node_registry.NODE_REGISTRY.update(saved)
