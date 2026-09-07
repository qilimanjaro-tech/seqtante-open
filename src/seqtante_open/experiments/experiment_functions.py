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

"""Registry mapping experiment names to their callables.

One flat dict. Every experiment takes ``str`` targets (``"q1"``, ``"c1_2"``), so
there is nothing to group by: ``CalibrationNode`` validates targets against
``str`` directly rather than looking the experiment up in a category.

The experiments seqtante-open ships:

- ``single_tone_vs_flux`` (offset calibration)
- ``single_tone``
- ``two_tone``
- ``two_tone_vs_flux``

Register each here as it lands.
"""
from typing import Any, Protocol

from qililab import Platform

from seqtante_open.experiments.nodes import (
    single_tone_node,
    single_tone_vs_flux,
    two_tone_frequency_vs_flux_node,
    two_tone_node,
)


class ExperimentFunction(Protocol):
    """Signature every experiment in the registry must implement.

    Experiments are always invoked with keyword arguments, so parameter order in
    the implementations is irrelevant, but the names are part of the contract.
    """

    def __call__(self, *, platform: Platform, platform_path: str, parameters: dict[Any, Any]) -> Any: ...


experiment_functions_dict: dict[str, ExperimentFunction] = {
    "offset_calibration": single_tone_vs_flux,
    "single_tone": single_tone_node,
    "two_tone": two_tone_node,
    "two_tone_vs_flux": two_tone_frequency_vs_flux_node,
}
"""``{experiment name: Callable(platform, platform_path, parameters) -> Any}``."""
