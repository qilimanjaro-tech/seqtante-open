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
"""Flux-bus and target helpers."""

import re
from typing import Any

from qililab.platform.platform import Platform


def apply_flux_filters(
    targets: list[str], qubit_loops: int, coupler_loops: int, filters: list[str] | None = None
) -> list[str]:
    """Create flux names based on number of loops and filters."""
    # Filters for couplers, qubits, and flux axis.
    filters = filters or []
    if "q" in filters:
        targets = [t for t in targets if t.startswith("q")]
    if "c" in filters:
        targets = [t for t in targets if t.startswith("c")]
    req = set()
    if "flux_x" in filters:
        req.add("x")
    if "flux_z" in filters:
        req.add("z")

    allowed_q = set(_axes_for_loops(qubit_loops))
    allowed_c = set(_axes_for_loops(coupler_loops))

    flux_buses = []
    for target in targets:
        if target.startswith("q"):
            q = int(target[1:])
            axes = (req & allowed_q) if req else allowed_q
            flux_buses += [_ch_qubit(q, ax) for ax in sorted(axes)]
        else:
            a, b = map(int, target[1:].split("_"))
            axes = (req & allowed_c) if req else allowed_c
            flux_buses += [_ch_coupler(a, b, ax) for ax in sorted(axes)]
    return flux_buses


def get_all_flux_buses(platform: Platform) -> list[str]:
    """Create the full list of flux buses based on the analogic topology inside the runcard"""
    if (
        platform.analog_compilation_settings is None
        or platform.analog_compilation_settings.qubits is None
        or platform.analog_compilation_settings.topology is None
    ):
        raise ValueError("Add qubits and topology inside runcard's analog.")
    qubit_targets = [f"q{qubit}" for qubit in platform.analog_compilation_settings.qubits]
    coupler_targets = [f"c{couplers[0]}_{couplers[1]}" for couplers in platform.analog_compilation_settings.topology]
    targets = qubit_targets + coupler_targets

    qubit_loops = platform.analog_compilation_settings.qubit_loops
    coupler_loops = platform.analog_compilation_settings.coupler_loops

    return apply_flux_filters(targets=targets, qubit_loops=qubit_loops, coupler_loops=coupler_loops)


def _axes_for_loops(num_loops: int):
    return ("z",) if num_loops == 1 else ("x", "z")


def _ch_qubit(q: int, axis: str) -> str:
    return f"flux_q{q}_{axis}"


def _ch_coupler(a: int, b: int, axis: str) -> str:
    a, b = sorted((a, b))
    return f"flux_c{a}_{b}_{axis}"


def coupler_readout_qubit(couplers: list[str], coupler_readout_overwrite: dict[str, str]):
    digits = re.compile(r"\d+")
    readout_x_couplers = {
        coupler: coupler_readout_overwrite.get(coupler, f"q{min(int(num) for num in digits.findall(coupler))}")
        for coupler in couplers
    }
    return readout_x_couplers


def x_loop_readout_flux(qubit: str, qubit_loops: int, parameters: dict[str, Any]) -> tuple[str, float] | None:
    """Bus and value of the x-loop bias holding the resonator readable, or None if there is no x loop."""
    if qubit_loops < 2:
        return None
    flux = parameters.get(qubit, {}).get("x_loop_readout_flux", parameters.get("x_loop_readout_flux"))
    return (f"flux_{qubit}_x", flux) if flux else None
