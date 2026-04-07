# Copyright 2023 Qilimanjaro Quantum Tech
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

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qililab.result import Measurement
from qililab.utils.serialization import deserialize

from seqtante_open.experiments.plotting import downconvert_IQ_filter


def plot_two_tone_readout_optimization(measurement: Measurement, title: str):

    # Load data and compute S21
    data, _ = measurement.load_old_h5()
    start = 0
    stop = data.shape[1]

    qp = deserialize(measurement.qprogram)  # type:ignore [call-overload]
    readout_if_freq = qp.body.elements[0].frequency
    # TODO: modify this hardcoded readout_if_freq

    I_drive, Q_drive, t = downconvert_IQ_filter(data[0, :, 0].T, data[0, :, 1].T, readout_if_freq, start, stop)
    I_no_drive, Q_no_drive, _ = downconvert_IQ_filter(data[1, :, 0].T, data[1, :, 1].T, readout_if_freq, start, stop)

    S21_drive = I_drive + 1j * Q_drive
    S21_no_drive = I_no_drive + 1j * Q_no_drive

    # Create subplot structure
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.35, 0.35, 0.35],
        subplot_titles=("Pulse Sequence", "S21 Real + |Abs|", "S21 Imag + Phase"),
    )

    # --- Subplot 1: Pulse sequence ---
    pulse_fig = qp.draw()

    offset = 0
    for trace in pulse_fig.data:
        if trace["x"]:
            offset = trace["x"][0] - 4
            break

    def shift_xaxis_all_traces(fig, offset):
        for trace in fig.data:
            if not hasattr(trace, "x") or trace.x is None:
                trace.x = list(range(len(trace.y)))
            trace.x = [x - offset for x in trace.x]

    shift_xaxis_all_traces(pulse_fig, offset=offset)  # NOTE HARDCODED OFFSET

    for trace in pulse_fig["data"]:
        trace.update(legendgroup="pulse", showlegend=True)
        fig.add_trace(trace, row=1, col=1)

    # --- Subplot 2: Real ---
    fig.add_trace(
        go.Scatter(x=t, y=np.real(S21_drive), mode="lines", name="S21_drive_real", legendgroup="real", showlegend=True),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=np.real(S21_no_drive), mode="lines", name="S21_no_drive_real", legendgroup="real", showlegend=True
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.abs(S21_drive),
            mode="lines",
            name="S21_drive_abs",
            visible="legendonly",
            legendgroup="real",
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.abs(S21_no_drive),
            mode="lines",
            name="S21_no_drive_abs",
            visible="legendonly",
            legendgroup="real",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    # --- Subplot 3: Imag ---
    fig.add_trace(
        go.Scatter(x=t, y=np.imag(S21_drive), mode="lines", name="S21_drive_imag", legendgroup="imag", showlegend=True),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=np.imag(S21_no_drive), mode="lines", name="S21_no_drive_imag", legendgroup="imag", showlegend=True
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.angle(S21_drive),
            mode="lines",
            name="S21_drive_phase",
            visible="legendonly",
            legendgroup="imag",
            showlegend=True,
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.angle(S21_no_drive),
            mode="lines",
            name="S21_no_drive_phase",
            visible="legendonly",
            legendgroup="imag",
            showlegend=True,
        ),
        row=3,
        col=1,
    )

    # Compute difference
    S21_diff = S21_drive - S21_no_drive

    # Real + Imag shown by default
    fig.add_trace(
        go.Scatter(x=t, y=np.real(S21_diff), mode="lines", name="S21_diff_real", legendgroup="diff", showlegend=True),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=np.imag(S21_diff), mode="lines", name="S21_diff_imag", legendgroup="diff", showlegend=True),
        row=4,
        col=1,
    )

    # Hidden toggles for magnitude and phase
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.abs(S21_diff),
            mode="lines",
            name="S21_diff_abs",
            visible="legendonly",
            legendgroup="diff",
            showlegend=True,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=np.unwrap(np.angle(S21_diff)),
            mode="lines",
            name="S21_diff_phase",
            visible="legendonly",
            legendgroup="diff",
            showlegend=True,
        ),
        row=4,
        col=1,
    )

    # --- Layout: adjust legend position ---
    fig.update_layout(
        height=1200,
        width=1100,
        title_text=title,
        xaxis4_title="Time (ns)",
        legend={"x": 1.02, "y": 0.5, "tracegroupgap": 200, "groupclick": "toggleitem"},  # Key for independent toggling!
    )
    fig.update_yaxes(title_text="Relative Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Relative Amplitude", row=3, col=1)

    return fig
