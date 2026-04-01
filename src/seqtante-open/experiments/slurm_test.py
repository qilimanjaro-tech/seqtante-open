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

from time import sleep
from typing import Any
from qililab.platform import Platform

def slurm_test(platform: Platform, platform_path: str, parameters: dict[str, Any]) -> bool:
    """Test experiment for SLURM submission infrastructure.

    Args:
        platform (Platform): The Qililab platform instance.
        platform_path (str): Path to the platform YAML/runcard.
        parameters (dict): Parameters for the test experiment.

    Returns:
        bool: True when experiment is done.
    """
    print("Running SLURM test experiment...")
    sleep(60)
    print("SLURM test experiment finished.")
    return True
