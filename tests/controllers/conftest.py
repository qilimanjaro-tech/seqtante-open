"""Shared fixtures for the controller tests."""

import importlib

import pytest

from seqtante_open.controllers import node_registry


@pytest.fixture(scope="session")
def _registry_baseline() -> dict[str, node_registry.ExperimentSpec]:
    """The registry as it stands once every real node module has registered, snapshot once per session.

    A :class:`CalibrationNode` subclass registers itself on import. Importing the node package here, up
    front, puts every real node in the baseline, so a node first imported inside a test is already in
    the snapshot the per-test restore reinstates — otherwise the restore would wipe that registration
    while the module stayed in ``sys.modules``, and it would never register again, failing later tests
    depending on order (``-n auto --dist loadfile``).
    """
    importlib.import_module("seqtante_open.experiments.nodes")  # import for its registration side effects

    return dict(node_registry.NODE_REGISTRY)


@pytest.fixture(autouse=True)
def restore_registry(_registry_baseline: dict[str, node_registry.ExperimentSpec]):
    """Restore ``NODE_REGISTRY`` to the session baseline after each test.

    A test that defines a ``CalibrationNode`` subclass registers it in the module-global registry;
    restoring to the baseline drops that leaked entry without wiping the real nodes it holds.
    """
    try:
        yield
    finally:
        node_registry.NODE_REGISTRY.clear()
        node_registry.NODE_REGISTRY.update(_registry_baseline)
