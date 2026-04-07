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

from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xarray import DataArray, apply_ufunc

from seqtante_open.experiments.analysis import decibels


def plot_measurement_1d_line_updated(
    xarr: DataArray,
    title: str,
    dataprocessing: Callable | None = None,
    fixed_LO_freq: float | None = None,
):

    x_var = xarr[xarr.dims[0]]  # in Hz'

    if dataprocessing:
        xarr = apply_ufunc(dataprocessing, xarr)
        label = f"S21 {dataprocessing.__name__}"
    else:
        label = "S21"

    # Add IF + LO for x-axis
    if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:  # only if we are sweeping an IF frequency
        start = xarr[xarr.dims[0]][0] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        stop = xarr[xarr.dims[0]][-1] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        y_vals = xarr.values
        if isinstance(xarr.values[0], complex):
            y_min_real, y_max_real = np.nanmin(np.real(y_vals)), np.nanmax(np.real(y_vals))
            y_min_imag, y_max_imag = np.nanmin(np.imag(y_vals)), np.nanmax(np.imag(y_vals))
        else:
            y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)

    if isinstance(xarr.values[0], complex):
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=x_var, y=np.real(xarr.values), mode="lines+markers", name="I"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_var, y=np.imag(xarr.values), mode="lines+markers", name="Q"), row=1, col=2)

        fig.update_xaxes(title_text=f"I {xarr.dims[0]}", row=1, col=1)
        fig.update_yaxes(title_text="Integrated Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text=f"Q {xarr.dims[0]}", row=1, col=2)
        fig.update_yaxes(title_text="Integrated Voltage (V)", row=1, col=2)

        if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:

            fig.update_layout(
                xaxis11={
                    "showgrid": False,
                    "title": "LO + IF (GHz)",
                    "anchor": "free",
                    "overlaying": "x1",
                    "side": "top",
                    "position": 1.0,
                },
                xaxis12={
                    "showgrid": False,
                    "title": "LO + IF (GHz)",
                    "overlaying": "x2",
                    "side": "top",
                },
            )
            fig.add_trace(
                go.Scatter(
                    x=[start, stop],
                    y=[y_min_real, y_max_real],
                    name="xaxis3 data",
                    xaxis="x11",
                    yaxis="y",
                    line={"width": 0},
                    marker={"opacity": 0},
                    showlegend=False,
                    hoverinfo="none",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[start, stop],
                    y=[y_min_imag, y_max_imag],
                    name="xaxis4 data",
                    xaxis="x12",
                    yaxis="y2",
                    line={"width": 0},
                    marker={"opacity": 0},
                    showlegend=False,
                    hoverinfo="none",
                ),
            )

        shapes = []
        # For Subplot 1
        domain1 = fig.layout.xaxis.domain
        y_domain1 = fig.layout.yaxis.domain
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": domain1[0],
                "x1": domain1[1],
                "y0": y_domain1[0],
                "y1": y_domain1[1],
                "line": {"color": "black", "width": 1},
                "fillcolor": "rgba(0,0,0,0)",
            }
        )
        # For Subplot 2
        domain2 = fig.layout.xaxis2.domain
        y_domain2 = fig.layout.yaxis2.domain
        shapes.append(
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": domain2[0],
                "x1": domain2[1],
                "y0": y_domain2[0],
                "y1": y_domain2[1],
                "line": {"color": "black", "width": 1},
                "fillcolor": "rgba(0,0,0,0)",
            }
        )
        fig.update_layout(shapes=shapes)

    else:
        fig = px.line(y=xarr.values, x=x_var, markers=True)
        fig.update_layout(xaxis_title=xarr.dims[0], yaxis_title=label)  # or a custom string

        fig.update_layout(
            shapes=[
                {
                    "type": "rect",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0,
                    "y0": 0,
                    "x1": 1,
                    "y1": 1,
                    "line": {"color": "black", "width": 1},
                    "fillcolor": "rgba(0,0,0,0)",
                }
            ]
        )

        if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:
            fig.add_trace(
                go.Scatter(
                    x=[start, stop],
                    y=[y_min, y_max],
                    name="xaxis3 data",
                    xaxis="x11",
                    line={"width": 0},
                    marker={"opacity": 0},
                    showlegend=False,
                    hoverinfo="none",
                )
            )
            fig.update_layout(
                xaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "x1", "side": "top", "position": 1.0}
            )

    fig.update_layout(
        font={"size": 15, "family": "Helvetica"},
        template="plotly_white",
        width=1000,
        height=700,
        margin={"t": 120},
        title={
            "text": title,
            "x": 0.5,  # centers the title (0 is left, 1 is right)
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "top",  # ensures the x position is the center of the text
        },
        showlegend=False,
    )
    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig


def plot_measurement_1d_freq_updated(
    xarr: DataArray, title: str, fixed_LO_freq: float | None = None, mag_phase_IQ_unwrap=True
):

    freq = xarr[xarr.dims[0]]  # in Hz'

    s21_corr = xarr

    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.08,
    )

    freq_vals = xarr[xarr.dims[0]]

    # trace 1 - magnitude
    fig.add_trace(
        go.Scatter(x=freq_vals, y=decibels(s21_corr.values), mode="lines+markers", name="Magnitude"), row=1, col=1
    )
    fig.update_xaxes(title_text=xarr.dims[0], row=1, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)

    # Trace 2 - phase
    if mag_phase_IQ_unwrap:
        y_phase = np.unwrap(np.angle(s21_corr.values))
    else:
        y_phase = np.angle(s21_corr.values)
    fig.add_trace(
        go.Scatter(x=freq_vals, y=y_phase, mode="lines+markers", name="Phase"),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=xarr.dims[0], row=1, col=2, scaleanchor="x", scaleratio=1)
    fig.update_yaxes(title_text="Phase (rad)", row=1, col=2)

    # Trace 3 and 4 - IQ-plane
    fig.add_trace(
        go.Scatter(x=np.real(s21_corr.values), y=np.imag(s21_corr.values), mode="lines+markers", name="IQ Circle"),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker={"color": "red", "size": 8}, name="Origin"), row=1, col=3
    )
    fig.update_xaxes(title_text="Real", row=1, col=3)
    fig.update_yaxes(title_text="Imaginary", row=1, col=3)

    # Styling and spacing
    fig.update_layout(
        autosize=True,
        template="plotly_white",
        width=1200,
        height=400,
        margin={"l": 80, "r": 90, "t": 100, "b": 60},
        title={"text": title, "x": 0.5, "xanchor": "center", "y": 0.95, "yanchor": "top"},
        font={"size": 15, "family": "Helvetica"},
    )

    # Adding dummy-trace to have axis for combined frequency
    # For trace 1
    if (freq.parameter == "IF_frequency") and fixed_LO_freq:  # only for IF freqs
        y_magnitude = decibels(s21_corr.values)

        start = xarr[xarr.dims[0]][0] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        stop = xarr[xarr.dims[0]][-1] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        fig.add_trace(
            go.Scatter(
                x=[start, stop],
                y=[y_magnitude.min(), y_magnitude.max()],
                name="xaxis3 data",
                xaxis="x11",
                line={"width": 0},
                marker={"opacity": 0},
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_layout(
            xaxis11={
                "showgrid": False,
                "title": "LO + IF (GHz)",
                "anchor": "free",
                "overlaying": "x1",
                "side": "top",
                "position": 1.0,
            }
        )

        # For trace 2
        fig.add_trace(
            go.Scatter(
                x=[start, stop],
                y=[y_phase.min(), y_phase.max()],
                name="xaxis4 data",
                xaxis="x12",
                yaxis="y2",
                line={"width": 0},
                marker={"opacity": 0},
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig.update_layout(
            xaxis12={
                "showgrid": False,
                "title": "LO + IF (GHz)",
                "anchor": "free",
                "overlaying": "x2",
                "side": "top",
                "position": 1.0,
            }
        )

        fig.update_xaxes(title={"standoff": 5})

    shapes = []
    # For Subplot 1 (primary axes: xaxis, yaxis)
    domain1 = fig.layout.xaxis.domain  # x-axis domain for subplot 1
    y_domain1 = fig.layout.yaxis.domain  # y-axis domain for subplot 1
    shapes.append(
        {
            "type": "rect",
            "xref": "paper",
            "yref": "paper",
            "x0": domain1[0],
            "x1": domain1[1],
            "y0": y_domain1[0],
            "y1": y_domain1[1],
            "line": {"color": "black", "width": 1},
            "fillcolor": "rgba(0,0,0,0)",
        }
    )
    # For Subplot 2 (primary axes: xaxis2, yaxis2)
    domain2 = fig.layout.xaxis2.domain
    y_domain2 = fig.layout.yaxis2.domain
    shapes.append(
        {
            "type": "rect",
            "xref": "paper",
            "yref": "paper",
            "x0": domain2[0],
            "x1": domain2[1],
            "y0": y_domain2[0],
            "y1": y_domain2[1],
            "line": {"color": "black", "width": 1},
            "fillcolor": "rgba(0,0,0,0)",
        }
    )
    # For Subplot 3 (primary axes: xaxis3, yaxis3)
    domain3 = fig.layout.xaxis3.domain
    y_domain3 = fig.layout.yaxis3.domain
    shapes.append(
        {
            "type": "rect",
            "xref": "paper",
            "yref": "paper",
            "x0": domain3[0],
            "x1": domain3[1],
            "y0": y_domain3[0],
            "y1": y_domain3[1],
            "line": {"color": "black", "width": 1},
            "fillcolor": "rgba(0,0,0,0)",
        }
    )
    fig.update_layout(shapes=shapes)

    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig


def plot_measurement_2d_heatmap_updated(
    xarr: DataArray, title: str, fixed_LO_freq: float | None = None, dataprocessing: Callable | None = None
):

    if dataprocessing:
        xarr = apply_ufunc(dataprocessing, xarr)
        coloraxis_colorbar = f"S21 {dataprocessing.__name__}"
    else:
        coloraxis_colorbar = "S21"

    fig = px.imshow(xarr, origin="lower", title=title)

    fig.update_layout(margin={"l": 60, "r": 60, "t": 60, "b": 60}, coloraxis_colorbar={"title": coloraxis_colorbar})

    # Add IF + LO for y-axis
    if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:  # y-axis
        fig.update_coloraxes(colorbar_x=1.15)

        fig.add_trace(
            go.Heatmap(
                z=np.array([np.zeros(len(xarr[xarr.dims[0]]))]).T,
                x=[None],
                y=np.array(xarr[xarr.dims[0]] * 1e-3 + fixed_LO_freq * 1e-9),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                yaxis="y11",
            )
        )

        fig.update_layout(
            yaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "y", "side": "right", "position": 1.0}
        )

    # Add IF + LO for x-axis
    elif (xarr[xarr.dims[1]].parameter == "IF_frequency") and fixed_LO_freq:  # x-axis
        fig.add_trace(
            go.Heatmap(
                z=[np.zeros(len(xarr[xarr.dims[1]]))],
                x=np.array(xarr[xarr.dims[1]] * 1e-3 + fixed_LO_freq * 1e-9),
                y=[None],
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                xaxis="x11",
            )
        )

        fig.update_layout(
            xaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "x1", "side": "top", "position": 1.0}
        )
        fig.update_layout(margin={"t": 120})

        fig.update_layout(title={"y": 0.95})

    fig.update_coloraxes(colorbar={"outlinewidth": 1}, colorbar_tickmode="auto", colorbar_ticks="outside")

    fig.update_layout(
        font={"size": 15, "family": "Helvetica"},
        template="plotly_white",
        width=1000,
        height=700,
        title={"x": 0.5, "xanchor": "center"},
        autosize=True,
    )

    fig.update_layout(
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": "black", "width": 1},
                "fillcolor": "rgba(0,0,0,0)",
            }
        ]
    )
    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig


def plot_measurement_2d_line_updated(
    xarr: DataArray, title: str, fixed_LO_freq: float | None = None, dataprocessing: Callable | None = None
):

    # xarr = convert_plot_units(xarr)
    if dataprocessing:
        xarr = apply_ufunc(dataprocessing, xarr)
        label = f"S21 {dataprocessing.__name__}"
    else:
        label = "S21"
    df = xarr.to_dataframe(name=label).reset_index()
    dim0, dim1 = xarr.dims[0], xarr.dims[1]

    df["trace_label"] = df[dim1].astype(str)

    fig = go.Figure()
    for trace_val in df["trace_label"].unique():
        trace_df = df[df["trace_label"] == trace_val]
        fig.add_trace(go.Scatter(x=trace_df[dim0], y=trace_df[label], mode="lines+markers", name=str(trace_val)))
    legend_title_text = xarr.coords[dim1].attrs.get("long_name", dim1)

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "y": 0.95, "yanchor": "top"},
        margin={"t": 90},
    )
    fig.update_layout(
        font={"size": 15, "family": "Helvetica"},
        template="plotly_white",
        width=1000,
        height=700,
        xaxis_title=dim0,
        yaxis_title=label,
        legend_title_text=legend_title_text,
    )

    # Add IF + LO for x-axis
    if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:  # only if we are sweeping an IF frequency
        start = xarr[xarr.dims[0]][0] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        stop = xarr[xarr.dims[0]][-1] * 1e-3 + fixed_LO_freq * 1e-9  # In GHz
        y_vals = xarr.values
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)

        fig.add_trace(
            go.Scatter(
                x=[start, stop],
                y=[y_min, y_max],
                name="xaxis3 data",
                xaxis="x11",
                line={"width": 0},
                marker={"opacity": 0},
                showlegend=False,
                hoverinfo="none",
            )
        )
        fig.update_layout(
            xaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "x1", "side": "top", "position": 1.0}
        )
        fig.update_layout(margin={"t": 120})

        fig.update_layout(title={"y": 0.95})

    fig.update_layout(
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": "black", "width": 1},
                "fillcolor": "rgba(0,0,0,0)",
            }
        ]
    )
    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig


def plot_measurement_3d_heatmap_slider_updated(
    xarr: DataArray, title: str, fixed_LO_freq: float | None = None, dataprocessing: Callable | None = None
):

    # xarr = convert_plot_units(xarr)
    if dataprocessing:
        xarr = apply_ufunc(dataprocessing, xarr)
        coloraxis_colorbar = f"S21 {dataprocessing.__name__}"
    else:
        coloraxis_colorbar = "S21"
    fig = px.imshow(
        xarr,
        animation_frame=xarr.dims[2],
        origin="lower",
        title=title,
    )

    fig.update_layout(margin={"l": 60, "r": 60, "t": 80, "b": 60}, coloraxis_colorbar={"title": coloraxis_colorbar})

    # Add IF + LO for y-axis
    if (xarr[xarr.dims[0]].parameter == "IF_frequency") and fixed_LO_freq:  # y-axis
        fig.update_coloraxes(colorbar_x=1.15)

        fig.add_trace(
            go.Heatmap(
                z=np.array([np.zeros(len(xarr[xarr.dims[0]]))]).T,
                x=[None],
                y=np.array(xarr[xarr.dims[0]] * 1e-3 + fixed_LO_freq * 1e-9),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                yaxis="y11",
            )
        )

        fig.update_layout(
            yaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "y", "side": "right", "position": 1.0}
        )

    # Add IF + LO for x-axis
    if (xarr[xarr.dims[1]].parameter == "IF_frequency") and fixed_LO_freq:  # x-axis
        fig.add_trace(
            go.Heatmap(
                z=[np.zeros(len(xarr[xarr.dims[1]]))],
                x=np.array(xarr[xarr.dims[1]] * 1e-3 + fixed_LO_freq * 1e-9),
                y=[None],
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                xaxis="x11",
            )
        )

        fig.update_layout(
            xaxis11={"title": "LO + IF (GHz)", "anchor": "free", "overlaying": "x1", "side": "top", "position": 1.0}
        )
        fig.update_layout(margin={"t": 110})

        fig.update_layout(title={"y": 0.95})

    fig.update_coloraxes(colorbar={"outlinewidth": 1}, colorbar_tickmode="auto", colorbar_ticks="outside")

    fig.update_layout(
        font={"size": 15, "family": "Helvetica"},
        template="plotly_white",
        width=1000,
        height=850,
        title={"x": 0.5, "xanchor": "center"},
        autosize=True,
    )

    fig.update_layout(
        shapes=[
            {
                "type": "rect",
                "xref": "paper",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "line": {"color": "black", "width": 1},
                "fillcolor": "rgba(0,0,0,0)",
            }
        ]
    )
    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig


def plot_measurement_3d_heatmap_grid_updated(xarr: DataArray,
                                             title: str,
                                             dataprocessing: Callable | None = None):

    if len(xarr[xarr.dims[2]]) > 10:  # we have this here for now to not make it crash
        xarr = xarr[:, :, :30]

    if dataprocessing:
        xarr = apply_ufunc(dataprocessing, xarr)
        coloraxis_colorbar = f"S21 {dataprocessing.__name__}"
    else:
        coloraxis_colorbar = "S21"
    if len(xarr[xarr.dims[2]]) <= 3:
        facet_col_wrap = len(xarr[xarr.dims[2]])
    else:
        facet_col_wrap = 4
    fig = px.imshow(
        xarr,
        facet_col=xarr.dims[2],
        facet_col_wrap=facet_col_wrap,
        origin="lower",
        title=title,
    )

    fig.update_layout(
        margin={"l": 60, "r": 60, "t": 100, "b": 60},
        coloraxis_colorbar={"title": coloraxis_colorbar},
    )

    fig.update_coloraxes(colorbar={"outlinewidth": 1}, colorbar_tickmode="auto", colorbar_ticks="outside")

    fig.update_layout(
        font={"size": 15, "family": "Helvetica"},
        template="plotly_white",
        width=1000,
        height=800,
        title={"x": 0.5, "xanchor": "center"},
        autosize=True,
    )

    fig.update_xaxes(minor_ticks="outside")
    fig.update_yaxes(minor_ticks="outside")

    return fig
