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

import io
from typing import Callable, List, Optional, Sequence, Union, cast

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image  # pip install -U plotly kaleido pillow
from qililab.result import Measurement
from scipy.signal import firwin, lfilter
from tqdm.auto import tqdm
from xarray import DataArray

# kaleido_get_chrome #NOTE look into if this code can be run without running this command. Potentially by using an earlier version.

def center_phase_around_median(S21):
    centered = np.apply_along_axis(subtract_median, axis=0, arr=phase(S21))
    return ((centered + np.pi) % (2 * np.pi)) - np.pi

def get_xarray_from_meas(measurement: Measurement):
    results, loops = measurement.load_old_h5()

    if len(results.shape) == len(loops.keys()):  # For the VNA
        S21 = results
    else:  # For Qblox and QM
        S21 = results[..., 0] + 1j * results[..., 1]

    coords = {}
    dims = []
    for dim_name, metadata in loops.items():
        if metadata["parameter"]:  #
            new_dim_name = f"{metadata['parameter']} {metadata['bus']} ({metadata['units']})"
        else:
            new_dim_name = dim_name
        dims.append(new_dim_name)
        vals = metadata["array"]
        attrs = {k: v for k, v in metadata.items() if k != "array"}
        coords[new_dim_name] = ((new_dim_name,), vals, attrs)

    xarr = DataArray(data=S21, dims=dims, coords=coords, name="S21")
    return xarr


def convert_plot_units(xobj: DataArray):
    """
    Rescale coords and rename dims so that plots will show
      "{bus} {parameter} ({unit})" on each axis.

    Conversions:
      - IF_frequency: Hz → MHz
      - LO_frequency: Hz → GHz
      - current:       A → mA
      - voltage:       V → V
    """
    conv = {
        "IF_frequency": (1e-6, "MHz"),
        "LO_frequency": (1e-9, "GHz"),
        "frequency": (1e-9, "GHz"),
        "current": (1e3, "mA"),
        "voltage": (1.0, "V"),
    }

    new_coords = {}
    rename_dims = {}

    for dim, coord in xobj.coords.items():
        param = coord.attrs.get("parameter", dim)
        bus = coord.attrs.get("bus", "")

        if param in conv:
            scale, new_unit = conv[param]

            new_coords[dim] = (coord.dims, coord.values * scale, {**coord.attrs, "unit": new_unit})

            label = f"{param} {bus} ({new_unit})".strip()
            rename_dims[dim] = label

    x2 = xobj.assign_coords(new_coords)

    if rename_dims:
        x2 = x2.rename(rename_dims)

    return x2


def downconvert_IQ_filter(I: np.ndarray, Q: np.ndarray, readout_if_freq: int, start: int = 0, stop: int = 10000):
    """digital downconversion of I and Q"""

    time_vec = np.arange(0, len(I), 1)

    roi = [start, stop]
    time_roi = time_vec[(time_vec > roi[0]) & (time_vec < roi[1])]
    I_roi = I[(time_vec > roi[0]) & (time_vec < roi[1])]
    Q_roi = Q[(time_vec > roi[0]) & (time_vec < roi[1])]

    # fft
    Fs = 1 / 1e-9  # Hz
    T = 1 / Fs
    L = len(time_roi)
    t = np.linspace(0.0, L * T, L, endpoint=False)

    # mixing with an LO at nco frequency
    mixed_I = np.exp(-1j * readout_if_freq * 2 * np.pi * t) * I_roi
    mixed_Q = np.exp(-1j * readout_if_freq * 2 * np.pi * t) * Q_roi

    # define a low-pass filter
    fs = 1 / (t[1] - t[0])
    nyq = fs / 2
    cutoff = 20e6  # Keep only baseband (adjust as needed)
    numtaps = 101
    lpf = firwin(numtaps, cutoff / nyq)  # construct the filter time response

    # Apply to I and Q components
    I_I = lfilter(lpf, 1.0, mixed_I.real)  # apply filters to data
    I_Q = lfilter(lpf, 1.0, mixed_I.imag)  # apply filters to data

    Q_I = lfilter(lpf, 1.0, mixed_Q.real)  # apply filters to data
    Q_Q = lfilter(lpf, 1.0, mixed_Q.imag)  # apply filters to data

    downverted_I = I_I + 1j * I_Q
    downverted_Q = Q_I + 1j * Q_Q

    return downverted_I, downverted_Q, time_vec


def _slider_steps_labels(slider: "go.layout.Slider") -> List[str]:
    """Extract labels from a Slider's steps safely."""
    steps = getattr(slider, "steps", [])
    labels = []
    for i, st in enumerate(steps):
        # st is a go.layout.slider.Step
        lbl = getattr(st, "label", None)
        labels.append(str(lbl) if lbl is not None else str(i))
    return labels


def _infer_frame_labels(fig: go.Figure) -> List[str]:
    """Infer per-frame labels from the first slider's steps or frame names."""
    frames = cast("Sequence[go.Frame]", fig.frames)

    sliders_obj = getattr(fig.layout, "sliders", None)
    sliders: Optional[Sequence[go.layout.Slider]] = (
        cast("Sequence[go.layout.Slider]", sliders_obj) if sliders_obj is not None else None
    )

    if sliders and len(sliders) > 0:
        labels = _slider_steps_labels(sliders[0])
        if labels and len(labels) == len(frames):
            return labels

    # fallback to frame names
    if frames and all(getattr(fr, "name", None) is not None for fr in frames):
        return [str(fr.name) for fr in frames]

    # fallback to indices
    return [str(i) for i in range(len(frames))]


def plotly_slider_to_gif(
    fig: go.Figure,
    output_path: str,
    ms_per_frame: int = 100,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: int = 2,
    include_base_frame: bool = False,
    loop: int = 0,
    include_base_traces_each_frame: bool = False,
    frame_subset: Optional[Sequence[int]] = None,
    labels: Optional[Sequence[Union[str, float, int]]] = None,
    label_formatter: Optional[Union[str, Callable[[Union[str, float, int], int], str]]] = "t = {label}",
    show_label_annotation: bool = True,
    label_xy: tuple = (0.01, 0.98),
    label_bg: str = "rgba(255,255,255,0.65)",
    label_borderpad: int = 3,
) -> str:
    """
    Export a Plotly slider/animation figure to GIF, advancing the slider's active index
    and drawing a per-frame label (annotation and/or title).
    """
    # Make mypy aware that fig.frames is a sequence of go.Frame
    frames_obj = fig.frames
    frames = cast("Sequence[go.Frame]", frames_obj)
    if not frames:
        raise ValueError("Figure has no frames.")

    export_width = width if width is not None else getattr(fig.layout, "width", None)
    export_height = height if height is not None else getattr(fig.layout, "height", None)

    all_frames = list(frames)
    idxs = list(frame_subset) if frame_subset is not None else list(range(len(all_frames)))

    inferred = _infer_frame_labels(fig)
    if labels is not None:
        base_labels: Sequence[Union[str, float, int]] = list(labels)
    else:
        # _infer_frame_labels returns List[str], which is compatible with Sequence[Union[str,float,int]]
        base_labels = cast("Sequence[Union[str, float, int]]", inferred)

    if len(base_labels) != len(all_frames):
        raise ValueError(
            f"labels length ({len(base_labels)}) must match number of frames ({len(all_frames)})."
        )

    if len(base_labels) != len(all_frames):
        raise ValueError(
            f"labels length ({len(base_labels)}) must match number of frames ({len(all_frames)})."
        )

    def _fmt_label(val: Union[str, float, int], i: int) -> str:
        if callable(label_formatter):
            return str(label_formatter(val, i))
        if isinstance(label_formatter, str):
            return label_formatter.format(label=val, i=i)
        return str(val)

    def _fig_to_pil(_fig: go.Figure) -> Image.Image:
        png_bytes = pio.to_image(
            _fig,
            format="png",
            width=export_width,
            height=export_height,
            scale=scale,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    images: list[Image.Image] = []
    if include_base_frame:
        images.append(_fig_to_pil(fig))

    base_layout = fig.layout

    for i in tqdm(idxs):
        fr = all_frames[i]
        label_text = _fmt_label(base_labels[i], i)

        # Compose traces
        data = (list(fig.data) + list(fr.data)) if include_base_traces_each_frame else list(fr.data)

        # Build figure for this frame
        frfig = go.Figure(data=data, layout=base_layout)
        if getattr(fr, "layout", None):
            frfig.update_layout(fr.layout)

        # Move slider thumb: mutate first slider's 'active' property
        sliders_obj = getattr(frfig.layout, "sliders", None)
        sliders: Optional[Sequence[go.layout.Slider]] = (
            cast("Sequence[go.layout.Slider]", sliders_obj) if sliders_obj is not None else None
        )
        if sliders and len(sliders) > 0:
            sliders_list = list(sliders)
            s0 = sliders_list[0]
            s0_dict = s0.to_plotly_json() if hasattr(s0, "to_plotly_json") else dict(s0)
            s0_dict["active"] = i
            sliders_list[0] = go.layout.Slider(**s0_dict)
            frfig.update_layout(sliders=sliders_list)

        # Overlay label
        if show_label_annotation:
            frfig.add_annotation(
                text=label_text,
                x=label_xy[0],
                y=label_xy[1],
                xref="paper",
                yref="paper",
                showarrow=False,
                bgcolor=label_bg,
                borderpad=label_borderpad,
                font={"size": 14},
            )

        images.append(_fig_to_pil(frfig))

    # Save GIF
    if len(images) == 1:
        images[0].save(output_path, format="GIF")
    else:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=ms_per_frame,
            loop=loop,
            disposal=2,
        )
    return output_path
