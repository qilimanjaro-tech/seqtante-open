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

import os
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, cast

import numpy as np
import plotly.graph_objects as go
from lmfit import Model
from qililab.data_management import build_platform
from scipy.special import erf as _erf
from xarray import DataArray

from seqtante_open.outputs import FittedParameter, output_controller

if TYPE_CHECKING:
    from qililab.platform.platform import Platform
    from qililab.result.database import AutocalMeasurement, Measurement

RENDER_FORMAT = "png"

RENDER_SCALE = 2  # pixel multiplier on the figure's layout size


class LorentzianFit(TypedDict):
    """What :meth:`FittingClass.fit_lorentzian` reports; every value NaN for a flat trace or a failed fit."""

    amplitude: float
    center: float
    #: Full width at half maximum, in the units of the swept values.
    width: float
    offset: float
    r_squared: float


#: What a fit's :meth:`FittingClass.fit` produces, per subclass (usually a ``TypedDict``).
ResultsT = TypeVar("ResultsT")


class FittingClass(ABC, Generic[ResultsT]):
    """Base class for all fittings.

    A subclass's :meth:`fit` assigns :attr:`results`, and every fitted value is read through it. Reading
    :attr:`results` before a successful fit raises, so no method has to remember to check first.
    """

    def __init__(
        self,
        measurement_id: int,
        target: str | None = None,
        path: str | None = None,
    ):
        """Base class for all fittings. It includes many functions for the fittings.

        Args:
            measurement_id (int): ID of the experiment to fit in the database of the `output_controller` schema.
            target (str | None, optional): Qubit or coupler token the fit belongs to, e.g. ``"q1"`` or ``"c1_2"``. Defaults to None.
            path (str | None, optional): Directory of the folder where the plot/s are saved, if None it shows the plot. Defaults to None.
        """
        self.id = measurement_id
        # The output_controller rejects the 'qaas' schema, so the entry is never a QaaS_Experiment
        self.measurement = cast(
            "Measurement | AutocalMeasurement", output_controller.db_manager.load_entry_by_id(measurement_id)
        )
        self.array, self.loops = self.measurement.load_h5()

        self.fitting: dict[str, float] = {}
        self.path = path
        self.target = target
        self._results: ResultsT | None = None

    @cached_property
    def platform(self) -> "Platform":
        """Platform the measurement was taken with, built from the runcard stored in the database."""
        if self.measurement.platform is None:
            raise ValueError(f"Measurement {self.id} has no platform stored in the database.")
        return build_platform(self.measurement.platform)

    def fit(self):
        """Fits the experimental data to the corresponding function."""

    def plot(self):
        """Plots the fitted data. if a folder is given, it saves ther plot, else it shows it."""

    @property
    def results(self) -> ResultsT:
        """What :meth:`fit` produced.

        Raises:
            RuntimeError: before :meth:`fit` has assigned a result, or after a fit that found none.
        """
        if self._results is None:
            raise RuntimeError(f"{type(self).__name__} has no fit result: call fit() first.")
        return self._results

    @results.setter
    def results(self, value: ResultsT) -> None:
        self._results = value

    def fit_and_plot(self):
        """Sequentially does the fit and the plot functions."""
        self.fit()
        self.plot()

    def add_fitted_parameter(self, parameter: FittedParameter, value: float):
        """Collect a fitted parameter in memory, for :meth:`save_fitting_to_database` to write.

        Args:
            parameter (FittedParameter): Name the value is stored under.
            value (float): Fitted value, rounded to 6 decimals.
        """
        self.fitting[parameter.value] = round(float(value), 6)

    def save_fitting_to_database(self):
        """Write the collected parameters to the ``fitting_parameters`` column of the measurement entry.

        Only the parameters listed in ``storage.parameters_to_save`` are written, or all of them when it
        is not set. Nothing is written when ``output_controller.fitting_saving_enabled`` is false.
        """
        if not output_controller.fitting_saving_enabled:
            return
        parameters = self._filtered_fitting()
        if parameters:
            output_controller.db_manager.add_fitting(id=self.id, parameters=parameters)

    def _filtered_fitting(self) -> dict[str, float]:
        if output_controller.parameters_to_save is None:
            return dict(self.fitting)
        return {k: v for k, v in self.fitting.items() if k in output_controller.parameters_to_save}

    def save_fitting_path_to_database(self, filepath: str):
        """Write ``filepath`` to the ``fitting_path`` column of the measurement entry.

        Nothing is written when ``output_controller.fitting_saving_enabled`` is false.

        Args:
            filepath (str): Path of the fit's representative plot.
        """
        if not output_controller.fitting_saving_enabled:
            return
        output_controller.db_manager.add_fitting(id=self.id, path=filepath)

    def save_plot(self, fig: go.Figure, title: str, save_to_db: bool = True) -> str | None:
        """Write ``fig`` under :attr:`path`, or show it when there is no path.

        Args:
            fig (go.Figure): Figure to render.
            title (str): Basename for the file. Newlines are stripped, so a plot
                title broken across lines can be passed straight in.
            save_to_db (bool, optional): Store the written path as the fit's representative plot with
                :meth:`save_fitting_path_to_database`, so the measurement entry points at it. Defaults to True
                (QHC-1932): every fit stores its plot's path.

        Returns:
            str | None: Path written, or None when the figure was shown instead.
        """
        if not self.path:
            fig.show()
            return None

        os.makedirs(self.path, exist_ok=True)
        filename = title.replace("\n", "").strip()
        filepath = os.path.join(self.path, f"{filename}.{RENDER_FORMAT}")
        fig.write_image(filepath, format=RENDER_FORMAT, scale=RENDER_SCALE)
        if save_to_db:
            self.save_fitting_path_to_database(filepath)
        return filepath

    # ----------------- Data-processing------------------
    def get_xarray(self) -> DataArray:
        """Build the measurement's ``S21`` map as a labelled :class:`~xarray.DataArray`.

        Reloads the measurement with ``load_h5`` and assembles one dimension per
        sweep loop. Each dimension is named ``"{parameter} {bus} ({units})"``
        (just ``"{bus} ({units})"`` for flux loops, whose parameter is implied)
        and carries the loop metadata in its coord ``attrs``, so downstream
        helpers such as :meth:`convert_plot_units` can rescale the
        axes. Loops without a ``parameter`` entry keep their raw h5 name.

        VNA results are already complex and are used as-is; Qblox and QM results
        arrive with a trailing I/Q axis which is combined into ``I + 1j * Q``.

        Returns:
            DataArray: Complex ``S21`` named ``"S21"``, with one labelled
                dimension per sweep loop.
        """

        # For the VNA
        if len(self.array.shape) == len(self.loops.keys()):
            s21 = self.array
        else:  # For Qblox and QM
            s21 = self.array[..., 0] + 1j * self.array[..., 1]

        coords = {}
        dims = []
        for dim_name, metadata in self.loops.items():
            if metadata["parameter"]:
                if metadata["parameter"] == "Flux":
                    new_dim_name = f"{metadata['bus']} ({metadata['units']})"
                else:
                    new_dim_name = f"{metadata['parameter']} {metadata['bus']} ({metadata['units']})"
            else:
                new_dim_name = dim_name
            dims.append(new_dim_name)
            attrs = {k: v for k, v in metadata.items() if k != "array"}
            coords[new_dim_name] = ((new_dim_name,), metadata["array"], attrs)

        return DataArray(data=s21, dims=dims, coords=coords, name="S21")

    @staticmethod
    def convert_plot_units(xobj: DataArray) -> DataArray:
        """Rescale coords and rename dims so plots read ``"{parameter} {bus} ({unit})"``.

        Reads the loop metadata :meth:`get_xarray` puts in each coord's ``attrs``,
        so the two are meant to be used together: build the map, then convert it
        just before plotting.

        Conversions:
          - ``IF_frequency``: Hz -> MHz
          - ``LO_frequency``, ``frequency``: Hz -> GHz
          - ``current``: A -> mA
          - ``voltage``: V -> V

        A coord whose ``parameter`` is not listed is left untouched, dimension
        name included.

        Args:
            xobj (DataArray): Labelled map, normally straight from :meth:`get_xarray`.

        Returns:
            DataArray: The same data with rescaled coords and relabelled dims.
        """
        conversions = {
            "IF_frequency": (1e-6, "MHz"),
            "LO_frequency": (1e-9, "GHz"),
            "frequency": (1e-9, "GHz"),
            "current": (1e3, "mA"),
            "voltage": (1.0, "V"),
        }

        new_coords = {}
        rename_dims = {}
        for dim, coord in xobj.coords.items():
            parameter = coord.attrs.get("parameter", dim)
            bus = coord.attrs.get("bus", "")
            if parameter not in conversions:
                continue
            scale, new_unit = conversions[parameter]
            new_coords[dim] = (coord.dims, coord.values * scale, {**coord.attrs, "unit": new_unit})
            rename_dims[dim] = f"{parameter} {bus} ({new_unit})".strip()

        converted = xobj.assign_coords(new_coords)
        return converted.rename(rename_dims) if rename_dims else converted

    @staticmethod
    def decibels(s21: np.typing.ArrayLike) -> np.ndarray:
        """Convert result values from s21 into dB

        Args:
            s21 (np.ndarray): combination of I + iQ

        Returns:
            np.ndarray: dB conversion
        """
        return 20 * np.log10(np.abs(s21))

    # ------------------- Utils Block -------------------
    @staticmethod
    def rotate_iq(arr: np.ndarray):
        """
        Function to rotate the I vs Q matrix to synthesized the data obtained
        into a single dimension (Real component)
        leaving the other only with noise (Imaginary component)

        Args:
            arr (np.array(float)): array of complex numbers corresponding to IQ signal

        Returns:
            np.array(float): rotated values
        """
        cov = np.cov(arr.real, arr.imag)
        # eigh, not eig: cov is symmetric, and eig returns complex eigenpairs on numpy >= 2.5.
        w, v = np.linalg.eigh(cov)
        principal = v[:, np.argmax(w)]
        if principal[0] < 0:  # the eigenvector sign is arbitrary, pin it
            principal = -principal
        angle = np.arctan2(principal[1], principal[0])
        return arr * np.exp(-1j * angle)

    @staticmethod
    def wrap_pi(angle: float) -> float:
        """Wrap angle to (-π, π]."""
        angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
        if angle <= -np.pi:
            angle = np.pi
        return angle

    @staticmethod
    def find_peaks_poly(poly_fit):
        sorted_peaks_indices = []
        aux = poly_fit[:-1] - poly_fit[1:]
        aux = aux > 0
        bool_comulative = aux[0]
        for i, bool_curr in enumerate(aux[1:]):
            if bool_curr != bool_comulative:
                bool_comulative = bool_curr
                sorted_peaks_indices.append(i + 1)

        return sorted_peaks_indices

    @staticmethod
    def fit_drag(loop_values, signal_0, signal_1):
        difference = signal_0 - signal_1

        z = np.polyfit(loop_values, difference, 2)
        p = np.poly1d(z)
        fit = p(loop_values)
        roots = p.roots
        fitted_drag_coeff = roots[0] if loop_values[0] < roots[0] < loop_values[1] else roots[1]

        return fit, fitted_drag_coeff

    @staticmethod
    def exponential(x: float | np.ndarray, a: float, b: float, c: float):
        """Returns exponential a * np.exp(b * x) + c"""
        return a * np.exp(b * x) + c

    @staticmethod
    def exponential_initial_guess(x_array: list | np.ndarray, y_array: list | np.ndarray) -> tuple[float, float, float]:
        """Generates an rough initial guess for an exponential fitting."""
        _n = len(y_array)
        _end_idx = 1 if int(_n * 2 / 100) < 1 else int(_n * 2 / 100)
        c: float = np.average(y_array[-_end_idx])
        a: float = y_array[0] - c
        b: float = -1 / x_array[int(_n / 2)]
        return a, b, c

    @staticmethod
    def joint_model(x, mmt_relax, thermal_pop, std0, v0, std1, v1, N):
        """Model used to fit the data."""
        prep0_m0 = 1 - thermal_pop
        prep1_m0 = mmt_relax
        cdf_prep0 = N * FittingClass.two_gaussians(x, prep0_m0, std0, v0, std1, v1)
        cdf_prep1 = N * FittingClass.two_gaussians(x, prep1_m0, std0, v0, std1, v1)

        return np.concatenate((cdf_prep0, cdf_prep1))

    @staticmethod
    def lorentzian(x, amplitude: float, center: float, width: float, offset: float):
        """A Lorentzian line of peak height ``amplitude`` (negative for a dip) and full width ``width`` at half maximum."""
        return amplitude / (1 + ((x - center) / (0.5 * width)) ** 2) + offset

    @staticmethod
    def lorentzian_initial_guess(
        x_values: np.ndarray, y_values: np.ndarray, dip: bool | None = None
    ) -> dict[str, float]:
        """A starting point for :meth:`lorentzian`, read off the trace, keyed by its parameter names.

        The baseline is the median of the window's edges; the line is a dip when ``dip`` says so, or,
        when it is ``None``, when the trace's mean sits below the baseline; the centre is the extremum, the
        height its distance from the baseline, and the width the span of the points past half that height
        (a tenth of the window when fewer than two are). Nothing depends on where the window sits in
        frequency or on the signal's scale.
        """
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        edge = max(3, len(y) // 10)
        offset = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
        if dip is None:
            dip = bool(np.mean(y) < offset)
        extreme = int(np.argmin(y) if dip else np.argmax(y))
        height = float(y[extreme] - offset)
        half = offset + height / 2
        past_half = y < half if dip else y > half
        span = float(np.ptp(x))
        width = float(np.ptp(x[past_half])) if past_half.sum() > 1 else span / 10
        return {"amplitude": height, "center": float(x[extreme]), "width": max(width, span / 100), "offset": offset}

    @staticmethod
    def fit_lorentzian(x_values: np.ndarray, y_values: np.ndarray) -> LorentzianFit:
        """Fit :meth:`lorentzian` to a trace and report its parameters and R²; the one Lorentzian fit (QHC-1932).

        A bounded local least-squares fit from :meth:`lorentzian_initial_guess`: the centre inside the
        window, the width between a hundredth of the window and the whole of it, the height within the
        trace's range and the baseline within its values. It fits once from a peak and once from a dip
        and keeps the higher R²: a line near the window's edge pulls the edge baseline, so the trace's
        mean alone can pick the wrong sign and start the local fit on a noise extremum. A fit that does
        not converge is dropped, since its last iterate could still pass a caller's checks. So is a fit
        whose centre lies within one sample spacing of either end of the window: the centre is bounded
        to the window, so a line just outside it is fit with its centre pinned on the edge, and the tail
        alone can reach a good R². A flat trace, or no fit kept, has every parameter and R² NaN, so a
        caller's R² check rejects it.
        """
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        nan = float("nan")
        failed = LorentzianFit(amplitude=nan, center=nan, width=nan, offset=nan, r_squared=nan)
        y_range, span = float(np.ptp(y)), float(np.ptp(x))
        if y_range == 0 or span == 0:
            return failed
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        # Centres closer than one sample spacing to an end of the window are not resolved inside it.
        margin = span / (x.size - 1)
        lowest, highest = float(x.min()) + margin, float(x.max()) - margin
        best = failed
        for dip in (False, True):
            guess = FittingClass.lorentzian_initial_guess(x, y, dip=dip)
            model = Model(FittingClass.lorentzian)
            model.set_param_hint("amplitude", value=guess["amplitude"], min=-y_range, max=y_range)
            model.set_param_hint("center", value=guess["center"], min=float(x.min()), max=float(x.max()))
            model.set_param_hint("width", value=guess["width"], min=span / 100, max=span)
            model.set_param_hint("offset", value=guess["offset"], min=float(y.min()), max=float(y.max()))
            try:
                fit = model.fit(data=y, params=model.make_params(), x=x, method="least_squares")
            except (ValueError, RuntimeError):
                continue
            if not fit.success or not lowest <= fit.params["center"].value <= highest:
                continue
            r_squared = 1.0 - float(np.sum((y - fit.best_fit) ** 2)) / ss_tot
            if np.isnan(best["r_squared"]) or r_squared > best["r_squared"]:
                best = LorentzianFit(
                    amplitude=float(fit.params["amplitude"].value),
                    center=float(fit.params["center"].value),
                    width=float(fit.params["width"].value),
                    offset=float(fit.params["offset"].value),
                    r_squared=r_squared,
                )
        return best

    @staticmethod
    def lorentzian_fit(y_values, x_values):
        """``(centre, fitted curve, R²)`` of :meth:`fit_lorentzian`, for the callers that take the tuple."""
        x = np.asarray(x_values, dtype=float)
        fit = FittingClass.fit_lorentzian(x, y_values)
        params = (fit["amplitude"], fit["center"], fit["width"], fit["offset"])
        return fit["center"], FittingClass.lorentzian(x, *params), fit["r_squared"]

    @staticmethod
    def sinus(x, a, b, c, d):
        return a * np.sin(b * np.array(x) - c) + d

    @staticmethod
    def sinus_abs(x, a, b, c, d):
        return abs(a) * np.sin(b * np.array(x) - c) + d

    @staticmethod
    def two_gaussians(x, a0, std0, v0, std1, v1):
        a1 = 1 - a0
        cdf_0 = a0 * (1 + _erf((x - v0) / (std0 * 2**0.5))) / 2
        cdf_1 = a1 * (1 + _erf((x - v1) / (std1 * 2**0.5))) / 2

        return cdf_0 + cdf_1

    @staticmethod
    def sinus_exp(x, a, b, c, d, f):
        # Define the decaying exponential function
        return a * np.cos(2 * np.pi * b * x + c) * np.exp(-1 * x / d) + f

    @staticmethod
    def line(x, a, b):
        return a * x + b

    @staticmethod
    def cosfunc(phi, a, omega, offset, phase_offset):
        return offset + a * np.cos(omega * phi + phase_offset)

    @staticmethod
    def oscillation_initial_guess(
        x_array: np.typing.ArrayLike, y_array: np.typing.ArrayLike, *, free_phase: bool = True
    ) -> dict[str, float]:
        """A starting point for fitting :meth:`cosfunc` to an oscillating trace, keyed by its parameter names.

        Scans angular frequencies from an eighth of a period over the window up to the sampling's Nyquist
        limit and, at each, fits the offset and the cosine (and the sine, with ``free_phase``) by linear
        least squares, keeping the frequency with the smallest residual. An FFT peak cannot do this for a
        trace holding one or two periods: its bins are a whole period per window apart, and a local fit
        started from the wrong bin settles in a neighbouring minimum.

        Args:
            x_array: The swept values.
            y_array: The trace.
            free_phase: Fit the phase as well. Without it ``phase_offset`` is 0 and ``a`` is signed.

        Returns:
            ``a``, ``omega``, ``offset`` and ``phase_offset``. ``omega`` is 0 when ``x_array`` spans no range.
        """
        x = np.asarray(x_array, dtype=float)
        y = np.asarray(y_array, dtype=float)
        span = float(np.ptp(x))
        if span == 0:
            return {"a": 0.0, "omega": 0.0, "offset": float(np.mean(y)), "phase_offset": 0.0}
        step = span / (len(x) - 1)
        best_residual, best_omega, best_coefficients = np.inf, 0.0, np.zeros(3)
        # 10 frequencies per FFT bin over the scanned range.
        for omega in np.linspace(np.pi / (4 * span), np.pi / step, 10 * len(x)):
            columns = [np.ones_like(x), np.cos(omega * x)] + ([np.sin(omega * x)] if free_phase else [])
            basis = np.column_stack(columns)
            coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
            residual = float(np.sum((basis @ coefficients - y) ** 2))
            if residual < best_residual:
                best_residual, best_omega, best_coefficients = residual, float(omega), coefficients
        offset, cosine = float(best_coefficients[0]), float(best_coefficients[1])
        if not free_phase:
            return {"a": cosine, "omega": best_omega, "offset": offset, "phase_offset": 0.0}
        # The cosine and sine coefficients are a·cos(phase) and -a·sin(phase).
        sine = float(best_coefficients[2])
        return {
            "a": float(np.hypot(cosine, sine)),
            "omega": best_omega,
            "offset": offset,
            "phase_offset": float(np.arctan2(-sine, cosine)),
        }
