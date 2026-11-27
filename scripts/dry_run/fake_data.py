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

"""Fake measurements for a dry run, built by the fit tests' data builders on the sweep a node declares.

Each entry is keyed by the experiment name a node streams its measurement under and gives the builder's
arguments for that sweep. The data comes back in the stream's shape, ``(*sweep axes, IQ)``.

Every parameter is read from the environment as ``DRY_RUN_[<TARGET>_][<SCOPE>_]<NAME>``, where the scope
is a step of the tree or an experiment, from the most specific key to the least. Unset, a feature sits at
``FEATURE_AT`` of the sweep, so a fit finds it whatever window the tree sweeps. See ``README.md``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.experiments.fitting.test_lorentzian_peak_fit import make_lorentzian_data
from tests.experiments.fitting.test_rabi_fit import make_rabi_data
from tests.experiments.fitting.test_single_tone_fit import make_single_tone_data
from tests.experiments.fitting.test_single_tone_vs_flux_fit import make_single_tone_vs_flux_data
from tests.experiments.fitting.test_t1_fit import make_t1_data
from tests.experiments.fitting.test_t2_fit import make_t2_data
from tests.experiments.fitting.test_two_tone_vs_flux_fit import make_two_tone_vs_flux_data

ENV_PREFIX = "DRY_RUN_"

#: Where in a sweep an unset feature is placed, as a fraction of the sweep from its first value.
FEATURE_AT = 0.4

#: Where in the IF sweep the apex of a vs-flux arc is placed when unset; the arc bends away from it.
APEX_IF_AT = 0.6

#: Every parameter the environment can set, with what it means.
PARAMETERS = {
    "FEATURE_AT": "fraction of the sweep an unset feature sits at",
    "QUBIT_IF": "two-tone line (two_tone, reset_point_search), drive IF in Hz",
    "RESONANCE_IF": "single-tone resonator dip, readout IF in Hz",
    "SWEET_SPOT": "vs-flux sweet spot (two_tone_vs_flux, offset_calibration), in the flux sweep's unit",
    "SWEET_SPOT_IF": "IF of the vs-flux arc at the sweet spot, in Hz",
    "PI_AMPLITUDE": "rabi pi amplitude, in drive amplitude",
    "T1": "T1, in ns",
    "T2": "T2, in ns",
    "NOISE": "standard deviation of the noise on I and Q, in signal units",
    "FAIL": "'fit' for data with no feature, 'execution' for an execution that raises",
}

#: The values ``FAIL`` takes.
FAIL_MODES = ("fit", "execution")

#: What a parameter's value must satisfy beyond being a finite number, and how to say it.
LIMITS: dict[str, tuple[Callable[[float], bool], str]] = {
    "NOISE": (lambda value: value >= 0, "must be 0 or more"),
    "T1": (lambda value: value > 0, "must be more than 0"),
    "T2": (lambda value: value > 0, "must be more than 0"),
    "PI_AMPLITUDE": (lambda value: value != 0, "must not be 0"),
}

#: The experiment of the legacy nodes, which stream outside ``CalibrationNode`` and so name none.
LEGACY_EXPERIMENTS = {
    "single_tone__frequency_sweep": "single_tone",
    "single_tone__frequency_vs_flux": "offset_calibration",
}


def _env_key(prefix: str, name: str) -> str:
    return f"{ENV_PREFIX}{prefix}_{name}" if prefix else f"{ENV_PREFIX}{name}"


@dataclass(frozen=True)
class Scope:
    """Who a measurement is for: its target, its experiment and the tree step that runs it."""

    target: str = ""
    experiment: str = ""
    step: str = ""

    def prefixes(self) -> list[str]:
        """The key prefixes to look a parameter up under, most specific first, ending with the global one."""
        target, experiment, step = (part.upper() for part in (self.target, self.experiment, self.step))
        candidates = [(target, step), (step,), (target, experiment), (experiment,), (target,), ()]
        prefixes = ["_".join(parts) for parts in candidates if all(parts)]
        return list(dict.fromkeys(prefixes))

    def lookup(self, name: str) -> tuple[str, str] | None:
        """The first ``(key, value)`` set for ``name`` in this scope, or ``None``."""
        for prefix in self.prefixes():
            key = _env_key(prefix, name)
            if key in os.environ:
                return key, os.environ[key]
        return None

    def number(self, name: str, default: float) -> float:
        """The number set for ``name`` in this scope, or ``default``.

        Raises:
            ValueError: if the value set is not a number.
        """
        found = self.lookup(name)
        if found is None:
            return default
        key, raw = found
        try:
            return float(raw)
        except ValueError as error:
            raise ValueError(f"{key}={raw!r} is not a number") from error

    def fail(self) -> str | None:
        """How this measurement is forced to fail, ``fit`` or ``execution``, or ``None``."""
        found = self.lookup("FAIL")
        return None if found is None else found[1].lower()


def environment_errors(targets: Iterable[str], scopes: Iterable[str]) -> list[str]:
    """What is wrong with the ``DRY_RUN_*`` variables set, given the tree's targets and its steps and experiments.

    A key that is no ``DRY_RUN_[<TARGET>_][<SCOPE>_]<NAME>`` of the tree (a typo like ``DRY_RUN_Q1_TI``, or a
    step the tree does not have), a ``FAIL`` that is not one of :data:`FAIL_MODES`, another value that is
    not a finite number, or one outside its :data:`LIMITS`.
    """
    targets = [target.upper() for target in targets]
    scopes = [scope.upper() for scope in scopes]
    prefixes = ["", *targets, *scopes, *(f"{target}_{scope}" for target in targets for scope in scopes)]
    valid = {_env_key(prefix, name): name for prefix in prefixes for name in PARAMETERS}
    errors = []
    for key in sorted(key for key in os.environ if key.startswith(ENV_PREFIX)):
        name, raw = valid.get(key), os.environ[key]
        if name is None:
            errors.append(f"{key} names no parameter, target, step or experiment of the tree")
        elif name == "FAIL":
            if raw.lower() not in FAIL_MODES:
                errors.append(f"{key}={raw!r} is not one of {', '.join(FAIL_MODES)}")
        else:
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"{key}={raw!r} is not a number")
                continue
            if not math.isfinite(value):
                errors.append(f"{key}={raw!r} is not a finite number")
            elif name in LIMITS and not LIMITS[name][0](value):
                errors.append(f"{key}={raw!r} {LIMITS[name][1]}")
    return errors


def _at(axis: np.ndarray, fraction: float) -> float:
    return float(axis[0] + fraction * (axis[-1] - axis[0]))


Number = Callable[[str, float], float]
Arguments = Callable[[Sequence[np.ndarray], Number], dict[str, Any]]


def _one_axis(axis: str, feature: str, parameter: str) -> Arguments:
    """The arguments of a one-sweep builder: the sweep as ``axis``, and ``feature`` from ``parameter``,
    at ``FEATURE_AT`` of the sweep when unset."""

    def arguments(axes: Sequence[np.ndarray], number: Number) -> dict[str, Any]:
        (sweep,) = axes
        return {axis: sweep, feature: number(parameter, _at(sweep, number("FEATURE_AT", FEATURE_AT)))}

    return arguments


def _vs_flux(axes: Sequence[np.ndarray], number: Number) -> dict[str, Any]:
    fluxes, frequencies = axes
    feature_at = number("FEATURE_AT", FEATURE_AT)
    return {
        "fluxes": fluxes,
        "frequencies": frequencies,
        "sweet_spot": number("SWEET_SPOT", _at(fluxes, feature_at)),
        "sweet_spot_if": number("SWEET_SPOT_IF", _at(frequencies, APEX_IF_AT)),
    }


Builder = Callable[..., tuple[np.ndarray, dict]]

#: The builder and its arguments, by the experiment name a node streams its measurement under.
FAKE_DATA: dict[str, tuple[Builder, Arguments]] = {
    "two_tone_saturation": (make_lorentzian_data, _one_axis("frequencies", "qubit_if", "QUBIT_IF")),
    "two_tone_vs_flux_saturation": (make_two_tone_vs_flux_data, _vs_flux),
    "single_tone__frequency_sweep": (make_single_tone_data, _one_axis("frequencies", "resonance_if", "RESONANCE_IF")),
    "single_tone__frequency_vs_flux": (make_single_tone_vs_flux_data, _vs_flux),
    "rabi_amplitude_drag": (make_rabi_data, _one_axis("amplitudes", "pi_amplitude", "PI_AMPLITUDE")),
    "t1_pulsed_saturation": (make_t1_data, _one_axis("wait", "t1", "T1")),
    "t2_drag": (make_t2_data, _one_axis("wait", "t2", "T2")),
}


def _featureless(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noise of the same mean and spread as ``data`` on each of I and Q, with nothing left to fit."""
    sweep_axes = tuple(range(data.ndim - 1))
    mean = data.mean(axis=sweep_axes, keepdims=True)
    spread = data.std(axis=sweep_axes, keepdims=True)
    return mean + spread * rng.standard_normal(data.shape)


def fake_data(
    experiment_name: str, loops: dict, scope: Scope, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any], str | None]:
    """The fake measurement streamed as ``experiment_name`` for ``scope``, over the sweep axes of ``loops``.

    With ``FAIL=fit`` the data is noise of the measurement's own size, with no feature for the fit to find.

    Returns:
        The data in stream shape, the values it was built with, and how it is forced to fail (or ``None``).

    Raises:
        KeyError: if no fake data is declared for ``experiment_name``.
        ValueError: if a parameter set in the environment is not a number.
    """
    if experiment_name not in FAKE_DATA:
        raise KeyError(f"no fake data for experiment '{experiment_name}'; declared: {sorted(FAKE_DATA)}")
    builder, arguments = FAKE_DATA[experiment_name]
    axes = [np.asarray(loop["array"]) for loop in loops.values()]
    kwargs = arguments(axes, scope.number)
    noise = scope.lookup("NOISE")
    if noise is not None:
        kwargs["sigma"] = scope.number("NOISE", 0.0)
    fail = scope.fail()
    data, _ = builder(rng, **kwargs)
    if fail == "fit":
        data = _featureless(data, rng)
    values = {name: value for name, value in kwargs.items() if not isinstance(value, np.ndarray)}
    return data, {**values, "fail": fail}, fail
