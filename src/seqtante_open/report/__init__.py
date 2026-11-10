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

"""The PDF report, built in-process from a run's results (ADR-0009).

This package is imported only when the report runs, behind the optional install extra
``seqtante_open[report]``; importing it pulls in the PDF-assembly dependency. The ``outputs`` service
never imports it, so a calibration run that does not build a report needs neither this package nor the
extra. :func:`write_report` turns a :class:`~seqtante_open.run_results.RunResults` into a PDF.
"""

from seqtante_open.report.pdf import write_report

__all__ = ["write_report"]
