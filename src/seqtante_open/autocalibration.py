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

"""Automatic calibration script (YAML-driven 1Q/2Q classification via `kind`)."""

import argparse
import warnings

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

from seqtante_open.calibration_run import CalibrationRun

warnings.filterwarnings("ignore", category=QCoDeSDeprecationWarning)
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0*")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic calibration runner")
    parser.add_argument("--platform_path", required=True, help="Path to the runcard or platform configuration")
    parser.add_argument("--config_path", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    CalibrationRun(args.platform_path, args.config_path)
