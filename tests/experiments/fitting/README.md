# Testing a fit class

Subclass `FittingTestCase` and you inherit tests for `__init__`, `fit` and `plot`.
There is nothing to register and no conftest to know about.

## Adding a case

**1. Put the test where the fit class lives.** The test tree mirrors `src/`.
`FluxoniumSingleToneFluxModel` is in `/src/seqtante_open/experiments/fitting/single_tone_vs_flux_fit.py`,
so its test goes in `tests/experiments/fitting/test_single_tone_vs_flux_fit.py`.
`test_registry.py` checks this, and tells you where the file should have gone.

**2. Write a builder for the data,** next to the case that uses it. It takes a
seeded generator and returns the `(results, loops)` pair the acquisition would
have written. Build the shape of the signal out of the model functions already on
`FittingClass`; the harness supplies the noise and the IQ packing.

**3. Declare the case.** In full:

```python
import numpy as np
import pytest

from seqtante_open.experiments.fitting.single_tone_vs_flux_fit import FluxoniumSingleToneFluxModel
from tests.experiments.fitting.harness import FittingTestCase, as_iq, loop

SWEET_SPOT = 0.15
"""Flux bias the resonator response is symmetric about, in V."""

FLUX_HALF_SPAN = 0.5
FLUX_POINTS = 41

FLUX_STEP = 2 * FLUX_HALF_SPAN / (2 * FLUX_POINTS - 2)
"""Step of the auto-convolution axis the fit reports its answer on."""


def make_single_tone_vs_flux_data(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """A resonator dip tracing a parabola in flux, centred on the sweet spot.

    Every cut of the image at fixed frequency is symmetric about ``SWEET_SPOT``,
    because the resonance depends on ``(flux - sweet_spot) ** 2``. That symmetry
    is what the auto-convolution inside the fit locates.
    """
    fluxes = np.linspace(SWEET_SPOT - FLUX_HALF_SPAN, SWEET_SPOT + FLUX_HALF_SPAN, FLUX_POINTS)
    frequencies = np.arange(-50_000_000, 10_000_001, 1_000_000)

    resonance = -10.0e6 - 30.0e6 * ((fluxes - SWEET_SPOT) / FLUX_HALF_SPAN) ** 2
    width = 4.0e6
    detuning = (frequencies[None, :] - resonance[:, None]) / (0.5 * width)
    magnitude = 1.0 - 0.8 / (1.0 + detuning**2)

    results = as_iq(magnitude, rng, sigma=0.002)

    loops = {
        "flux": loop(fluxes, units="V", bus="flux_q1_z", parameter="Flux"),
        "frequency": loop(frequencies, units="Hz", bus="readout_q1", parameter="IF_frequency"),
    }
    return results, loops


class TestFluxoniumSingleToneFluxModel(FittingTestCase):
    FIT_CLASS = FluxoniumSingleToneFluxModel
    DATA = "single_tone_vs_flux_fit.h5"
    BUILDER = make_single_tone_vs_flux_data
    INIT = {"measurement_id": 1, "target": "q1"}
    EXPECTED = {
        "center": pytest.approx(SWEET_SPOT, abs=FLUX_STEP),
        "offset": pytest.approx(-SWEET_SPOT, abs=FLUX_STEP),
    }
```

**4. Write the data file and run.**

```
python -m tests.experiments.fitting.data.make_data
pytest tests/experiments/transmons/single_qubit_gates/fit/test_t1_fit.py
```

**5. There is no step 5.** `test_registry.py` discovers both sides by subclassing,
so your case is picked up the moment the module is imported. Nothing to register.

## What you get for free

| test | what it catches |
| --- | --- |
| `test_init` | constructor blows up on the data; an output already populated before `fit()`, which would make `EXPECTED` meaningless |
| `test_fit` | every `EXPECTED` path, after `fit()`. Skipped if the class does not override `fit` |
| `test_plot_saves_image_to_path` | `plot()` writing nothing, writing an empty file, or leaking an open matplotlib figure |
| `test_plot_shows_when_no_path` | `plot()` silently doing nothing when `path` is `None`. Counts both matplotlib and plotly |
| `test_added_methods_are_covered` | a public method beyond `fit`/`plot` with no test of its own |
| `test_init_signature_matches_declared_kwargs` | `INIT` drifting from a renamed constructor argument |

## Declaring the case

Required:

- **`FIT_CLASS`** the class under test.
- **`INIT`** the keyword arguments it is constructed with.
- **`DATA`** either the name of an `.h5` in `data/`, or a builder called at test
  time. A committed file and a builder reach the fit class through an identical
  path, so switching between them changes nothing else.
- **`EXPECTED`** dotted attribute path to expected value. `"optimized_params.rot.1"`
  resolves through attributes, dict keys and indices. Values compare with `==`, so
  `pytest.approx` works; a callable is used as a predicate instead. Required
  whenever the class overrides `fit`, because a fit test that asserts nothing is
  worse than no test at all.

Optional:

- **`BUILDER`** the builder that produced a committed `DATA` file, so
  `make_data` can rewrite it.
- **`MEASUREMENT`** overrides on the fake measurement. `platform_before` defaults
  to the test runcard and `data_shape` to the results shape, so most cases need
  nothing here. `T2Fit` needs a `calibration` blob.
- **`PLOTS`** exact filenames `plot()` must produce, if you want them pinned.
- **`WAIVED`** `{"method": "why it needs no test"}`. The reason is the point; a
  waiver naming a method the class does not define is rejected at import.
- **`SEED`** for a `DATA` builder, and for the global numpy RNG, which is seeded
  with it just before `fit()`. The Lorentzian fits run `differential_evolution`
  and draw their population from that RNG, so an unseeded fit answers differently
  depending on what ran before it. Defaults to 0.

## How the harness fakes a measurement

`__init__` on most fit classes calls
`output_controller.db_manager.load_calibration_by_id(id).load_h5()`. The harness
mocks `db_manager` and hands back a stand-in measurement carrying your data,
along with `platform_before`, `data_shape` and `calibration`.

`store_parameter` is replaced by a no-op. It is on its way out, and left alone it
raises `AttributeError`, because `Outputs.storage_conf` is only ever set by
`reset()`. Seven fit classes could not run at all otherwise. Nothing asserts on
what a fit stores.

`fit()` runs once per test class and each test gets a deep copy, so a test may
mutate its object freely without affecting the others.

## Data files

Committed `.h5` files live in `data/`, whatever the depth of the test using them.
`python -m tests.experiments.fitting.data.make_data` rewrites every one of them
from its `BUILDER`, and reports any file no case regenerates.

A case whose `DATA` is a builder writes nothing to `data/` at all. Prefer a
committed file when the data is expensive to build or you want it frozen against
a numpy upgrade; `numpy` does not guarantee `default_rng` stream stability across
versions.
