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

"""Shared fixtures for node tests. The reusable fakes/helpers live in :mod:`tests.experiments.nodes.harness`."""

import pytest

from tests.experiments.nodes.harness import FakePlatform


@pytest.fixture
def fake_platform() -> FakePlatform:
    """A fresh :class:`~tests.experiments.nodes.harness.FakePlatform` per test."""
    return FakePlatform()
