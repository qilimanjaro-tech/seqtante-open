# Testing a fit class

Subclass `FittingTestCase` and you inherit tests for `__init__`, `fit` and `plot`.
There is nothing to register and no conftest to know about.

## Adding a case

> The `T1Fit` walkthrough below came across from seqtante and is kept because it
> is the simplest possible case. seqtante-open has no transmon fits, so treat its
> paths as illustrative of the mirroring rule, not as files you will find.

**1. Put the test where the fit class lives.** The test tree mirrors `src/`.
`T1Fit` is in `src/seqtante_open/experiments/transmons/single_qubit_gates/fit/t1_fit.py`,
so its test goes in `tests/experiments/transmons/single_qubit_gates/fit/test_t1_fit.py`.
`test_registry.py` checks this, and tells you where the file should have gone.

**2. Write a builder for the data,** next to the case that uses it. It takes a
seeded generator and returns the `(results, loops)` pair the acquisition would
have written. Build the shape of the signal out of the model functions already on
`FittingClass`; the harness supplies the noise and the IQ packing.

**3. Declare the case.** In full:

```python
import numpy as np
import pytest

from seqtante_open.experiments.fitting.fit_base import FittingClass
from seqtante_open.experiments.transmons.single_qubit_gates.fit.t1_fit import T1Fit
from tests.experiments.fitting.harness import FittingTestCase, add_noise, as_iq, loop

T1_NS = 12_400
DECAY_RATE = -1.0 / T1_NS


def make_t1_data(rng):
    wait = np.arange(0, 40_001, 500)
    decay = FittingClass.exponential(wait, 1.0, DECAY_RATE, 0.0)

    iq = as_iq(decay, rng, sigma=0.01)
    threshold = np.clip(add_noise(decay, rng, 0.01), 0.0, 1.0)
    results = np.column_stack([iq[:, 0], iq[:, 1], threshold])

    return results, {"wait": loop(wait, units="ns", bus="drive_q1", parameter="duration")}


class TestT1Fit(FittingTestCase):
    FIT_CLASS = T1Fit
    DATA = "t1_fit.h5"
    BUILDER = make_t1_data
    INIT = {"qubit_idx": 0, "measurement_id": 1}
    EXPECTED = {"optimized_params.thresh.1": pytest.approx(DECAY_RATE, rel=0.02)}
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
- **`SEED`** for a `DATA` builder. Defaults to 0.

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
