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

"""Shared fixtures for experiment tests.

Fixtures defined here are automatically available to every test under
``tests/experiments/`` — no ``pytest_plugins`` registration is needed (and it is
no longer supported in a non-top-level conftest).

- ``platform``: a *real* ``Platform`` built from a runcard file, so all
  parameters (analog settings, topology, buses, IFs) are genuine. Nothing is
  connected, so no hardware is touched; the execution functions that would talk
  to instruments are mocked in the individual tests instead. The runcard is
  taken from a ``RUNCARD_PATH`` defined in the test module::

      RUNCARD_PATH = Path(__file__).resolve().parents[2] / "runcards" / "test_AQPU_runcard.yml"

- ``mock_db_manager``: replaces the ``output_controller.db_manager`` singleton
  with a MagicMock so nothing hits a database.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qililab.data_management import build_platform
from qililab.platform.platform import Platform

from seqtante_open.outputs import output_controller


@pytest.fixture
def platform(request: pytest.FixtureRequest) -> Platform:
    """A real ``Platform`` built (offline, not connected) from the runcard the
    test module points to via a module-level ``RUNCARD_PATH``.

    Parameters read from the runcard are genuine; instruments are not connected,
    so reads/writes stay in memory. The experiment sets its own crosstalk matrix
    before any ``set_parameter``/``set_bias_to_zero`` call, so those work
    without hardware. The execution functions that talk to instruments are
    mocked in the individual tests.
    """
    runcard_path = getattr(request.module, "RUNCARD_PATH", None)
    if runcard_path is None:
        raise pytest.UsageError(
            f"{request.module.__name__} must define a module-level RUNCARD_PATH for the `platform` fixture."
        )
    runcard_path = Path(runcard_path)
    if not runcard_path.exists():
        pytest.skip(f"Runcard not found: {runcard_path}")
    return build_platform(runcard=str(runcard_path))


@pytest.fixture
def mock_db_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the singleton ``output_controller.db_manager`` with a MagicMock.

    ``db_manager`` is a property backed by ``_db_manager``; setting the backing
    field makes the property return our mock without hitting a real database.
    ``monkeypatch`` restores the original after the test.
    """
    db_manager = MagicMock(name="db_manager")
    monkeypatch.setattr(output_controller, "_db_manager", db_manager)
    return db_manager


class MockRecorder:
    """Patch callables with recording stubs, one record per patched function.

    Call :meth:`mock` to replace a function with a stub that returns ``output``
    and logs every invocation. Recorded calls live in :attr:`calls`, keyed by
    function name, each entry being ``{"args": (...), "kwargs": {...}}`` in call
    order. The fixture that hands out a recorder is function-scoped, so a fresh,
    empty recorder is created for every test method and all patches are undone
    on teardown — nothing leaks between tests.

    Stubs are autospecced by default, so a call that doesn't match the real
    callable's signature raises ``TypeError`` instead of being recorded — a
    renamed or dropped parameter fails the test that asserts on it rather than
    passing against a signature the production code no longer has. Pass
    ``validate_signature=False`` to opt out for a given target.
    """

    def __init__(self) -> None:
        self.calls: dict[str, list[dict]] = {}
        self._patchers: list = []

    def mock(
        self,
        target: str | Callable,
        output: object = None,
        *,
        name: str | None = None,
        validate_signature: bool = True,
    ) -> MagicMock | Callable:
        """Replace ``target`` with a stub returning ``output`` and recording calls.

        Args:
            target: Dotted path to patch (e.g.
                ``"seqtante_open.experiments.nodes.offset_calibration.single_tone_vs_flux"``),
                or the callable itself. Patch at the *import site* — where the
                function is looked up — not its definition module, or a
                ``from x import y`` reference won't be intercepted.
            output: Value the stub returns on every call.
            name: Key under which calls are recorded. Defaults to the last
                dotted segment of ``target`` (or ``target.__name__``).
            validate_signature: When true (the default) the stub is autospecced
                against the real callable, so a call whose arguments don't match
                the real signature raises ``TypeError`` instead of being silently
                recorded. Set to false to patch a target that cannot be
                autospecced (a name that doesn't exist yet, a dynamically built
                attribute, …).

        Returns:
            The stub now installed in place of ``target``. With
            ``validate_signature`` it is a signature-matching function wrapper
            rather than a bare ``MagicMock``; either way it carries the usual
            ``assert_called_*``/``call_args``/``call_count`` API.
        """
        if name is None:
            name = target.rsplit(".", 1)[-1] if isinstance(target, str) else target.__name__
        if not isinstance(target, str):
            target = f"{target.__module__}.{target.__name__}"

        calls = self.calls.setdefault(name, [])

        def _record(*args: object, **kwargs: object) -> object:
            calls.append({"args": args, "kwargs": kwargs})
            return output

        if validate_signature:
            patcher = patch(target, autospec=True, side_effect=_record)
        else:
            patcher = patch(target, side_effect=_record)
        mock = patcher.start()
        self._patchers.append(patcher)
        return mock

    def reset(self) -> None:
        """Undo every patch and drop all recorded calls."""
        for patcher in self._patchers:
            patcher.stop()
        self._patchers.clear()
        self.calls.clear()


@pytest.fixture
def mock_recorder() -> Iterator[MockRecorder]:
    """Function-scoped recorder for mocking functions and recording their calls.

    Yields a fresh :class:`MockRecorder`; on teardown all patches are undone and
    recorded calls are cleared, so state never persists between test methods.
    """
    recorder = MockRecorder()
    yield recorder
    recorder.reset()
