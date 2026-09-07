# seqtante_open 0.1.0 (2026-08-26)

## Features

- Ported the calibration engine from [seqtante](https://github.com/qilimanjaro-tech/seqtante) at `v0.2.4`, and narrowed it to the experiments this package ships.

  ### Running a calibration

  `CalibrationRun` replaces the module-level `main()` in `autocalibration.py`, which is now an argparse shim over it. Constructing the object runs the calibration: build the platform, compile the tree, connect, walk the graph, close out.

  ```python
  from seqtante_open import CalibrationRun

  CalibrationRun(platform_path="runcard.yml", config_path="calibration_tree.yml")
  ```

  ### Calibration trees

  - **`all_q` and `all_c` targets.** A pipeline can now say `targets: [all_q, all_c]` and the parser expands them from the runcard's `analog.qubits` and `analog.topology`. Raises if the runcard has no `analog` section.
  - **`calibration_path`** declared at the top of the tree is propagated into every experiment's parameters, so experiments that read or update a `Calibration` file no longer need it repeated per node.
  - **`simultaneous`** is honoured. It was present but disabled upstream, so pipelines that declared parallel targets ran serially.
  - **Targets are strings.** Every experiment takes `"q1"` / `"c1_2"` tokens, and `CalibrationNode` rejects anything else with a warning rather than silently running it.

  ### Fitting

  `FittingClass` gains the pieces a fit needs to go from a measurement id to a labelled plot:

  - **`get_xarray()`** builds the measurement's `S21` map as an `xarray.DataArray`, one named dimension per sweep loop, with the loop metadata carried in each coord's `attrs`. Handles both the complex VNA layout and the trailing I/Q axis Qblox and QM return.
  - **`convert_plot_units()`** rescales and relabels those axes for plotting (Hz->MHz for IF, Hz->GHz for LO, A->mA). Reads the `attrs` `get_xarray()` writes, so the two are meant to be used together.
  - **`decibels()`** for `20*log10(|S21|)`.
  - **`save_plot(fig, title)`** now takes a plotly figure and renders it through kaleido, which needs no display. It writes a PNG under the fit's `path` and returns that path, or shows the figure when there is no path. Format, scale and engine are module constants (`RENDER_FORMAT`, `RENDER_SCALE`, `RENDER_ENGINE`) rather than call-site arguments. The previous implementation was matplotlib-only and returned nothing, so a fit could not register the file it had just written.

  `target` on a fit is a string token, and the base class no longer carries the `qubit_idx` / `control_qubit_idx` / `target_qubit_idx` triple.
  ([PR #4](https://github.com/qilimanjaro-tech/seqtante-open/pull/4))

- Added Offset Calibration (registered as `offset_calibration` in the calibration tree) single-tone-vs-flux experiment along with its fitter as an available fluxonium node.
  It sweeps the readout IF against the bias of one flux loop at a time, finds the flux-symmetry point of the resonator trace and accumulates the fitted offset into the `CrosstalkMatrix` flux offsets of the `Calibration`, which is written back to `calibration_path` when the node ends.

  Both qubit and coupler targets are accepted. Each target is measured once per flux loop the runcard declares (`analog.qubit_loops` / `analog.coupler_loops`), x loop first, then z with the x loop parked at `x_loop_readout_flux` so the resonator stays readable. A coupler has no readout line of its own, so it is read through a qubit: by default the lowest-index qubit of its token (`c1_2` -> `q1`).

  The `Calibration` at `calibration_path` must already carry a `CrosstalkMatrix`; the node raises otherwise.
  The parameters used are:
  ### **Required**

  Written under the node's parameters; any of these can also be overridden per target via `overwrites:`.

  - **if_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the readout IF sweep. Relative: the
    bus's current `Parameter.IF` is added to it.

  - **flux_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the flux sweep, in `phi_0`. Absolute:
    applied to the swept flux bus as `Parameter.FLUX`.

  - **readout_amp** : `float`.
    Readout pulse amplitude.

  - **duration** : `int`.
    Readout pulse duration in ns.

  - **averages** : `int`.
    Number of hardware averages.

  ### **Optional**

  - **x_loop_readout_flux** : `float`, default `None`.
    Bias held on the qubit's x loop while another loop is swept, in `phi_0`. Ignored on
    runcards with a single qubit loop. Read per qubit first, then from the node's
    parameters.

  - **coupler_readout_qubit** : `dict[str, str]`, default `{}`.
    Which qubit reads out a given coupler, e.g. `{c1_2: q2}`. Any coupler not listed is
    read through the lowest-index qubit of its token.

  - **minimum_wait_after_step** : `float`, default `None`.
    Settling wait after each QDAC flux step, in ns. Replaces the value computed from the
    platform's QDAC low-pass filters. Must be set together with `qdac_stop_ro_before_step`.

  - **qdac_stop_ro_before_step** : `float`, default `None`.
    How long readout stops before each QDAC step, in ns. Must be set together with
    `minimum_wait_after_step`.

  ## **Example**
  ```yml
  offset_calibration:
    if_sweep: [-1.5e6, 1.5e6, 201]
    flux_sweep: [-1, 1, 51]
    readout_amp: 0.075
    duration: 2000
    averages: 1000
    x_loop_readout_flux: 0.3
    coupler_readout_qubit:
      c1_2: q2
  ```
  ([PR #6](https://github.com/qilimanjaro-tech/seqtante-open/pull/6))

- Added Two Tone vs Flux (registered as `two_tone_vs_flux` in the calibration tree) qubit-spectroscopy-vs-flux experiment along with its fitter as an available fluxonium node.
  It sweeps the drive IF against the bias of one flux loop at a time, fits a Lorentzian to the rotated signal quadrature of every flux row and then a parabola through those fitted IFs: the vertex is the flux sweet spot, and its negative is accumulated into the `CrosstalkMatrix` flux offsets of the `Calibration`, which is written back to `calibration_path` when the node ends.

  Rows whose Lorentzian never reaches `r² >= 0.9` are dropped, and the parabola is only accepted when at least three rows survive and its vertex falls inside the swept flux range. A target whose fit is rejected leaves the flux offsets untouched instead of writing a meaningless one.

  Both qubit and coupler targets are accepted, measured once per flux loop the runcard declares (`analog.qubit_loops` / `analog.coupler_loops`), x loop first, then z with the x loop parked at `x_loop_readout_flux`. A coupler has no drive or readout line of its own, so it is driven and read through a qubit: by default the lowest-index qubit of its token (`c1_2` -> `q1`).

  The drive LO is taken from the `Calibration`'s `LO` entry for the target when there is one, and from the drive bus otherwise.
  The `Calibration` at `calibration_path` must already carry a `CrosstalkMatrix`; the node raises otherwise.
  The parameters used are:
  ### **Required**

  Written under the node's parameters; any of these can also be overridden per target via `overwrites:`.

  - **freq_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the drive IF sweep. Relative: the
    bus's current `Parameter.IF` is added to it.

  - **flux_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the flux sweep, in `phi_0`. Absolute:
    applied to the swept flux bus as `Parameter.FLUX`.

  - **drive_amplitude** : `float`.
    Drive pulse amplitude.

  - **drive_duration** : `int`.
    Drive pulse duration in ns.

  - **readout_amplitude** : `float`.
    Readout pulse amplitude.

  - **readout_duration** : `int`.
    Readout pulse duration in ns.

  - **averages** : `int`.
    Number of hardware averages.

  - **relax_duration** : `int`.
    Relaxation wait between shots in ns.

  ### **Optional**

  - **drive_gain** : `float`, default `1`.
    Output level of the drive source: power in dBm on a Rohde & Schwarz LO, gain on a
    QCM-RF.

  - **ringup_time** : `int`, default `0`.
    Time of the pulse needed to excite the resonator for readout (ns).

  - **overlap_time** : `int`, default `0`.
    Overlap between the end of the drive pulse and the start of the readout pulse (ns).

  - **x_loop_readout_flux** : `float`, default `None`.
    Bias held on the qubit's x loop while another loop is swept, in `phi_0`. Ignored on
    runcards with a single qubit loop. Read per qubit first, then from the node's
    parameters.

  - **coupler_readout_qubit** : `dict[str, str]`, default `{}`.
    Which qubit drives and reads out a given coupler, e.g. `{c1_2: q2}`. Any coupler not
    listed uses the lowest-index qubit of its token.

  ## **Example**
  ```yml
  two_tone_vs_flux:
    freq_sweep: [-300e6, 300e6, 301]
    flux_sweep: [-1, 1, 51]
    drive_amplitude: 0.5
    drive_duration: 4000
    readout_amplitude: 0.075
    readout_duration: 2000
    averages: 1000
    relax_duration: 200000
    drive_gain: 0.8
    overlap_time: 4000
    x_loop_readout_flux: 0.3
  ```
  ([PR #7](https://github.com/qilimanjaro-tech/seqtante-open/pull/7))

- Added Two Tone (registered as `two_tone` in the calibration tree) qubit spectroscopy experiment along with its fitter as an available fluxonium node.
  It sweeps the drive IF at zero flux, rotates the IQ plane so the response collapses onto one quadrature, fits a Lorentzian to it and writes the fitted IF back into the drive bus. The runcard and the `Calibration` are saved when the node ends.

  Only qubit targets are measured; coupler tokens in `targets` are skipped, since a coupler has no drive line of its own. The drive LO is taken from the `Calibration`'s `LO` entry for the target when there is one, and from the drive bus otherwise.
  The `Calibration` at `calibration_path` must already carry a `CrosstalkMatrix`; the node raises otherwise.
  The parameters used are:
  ### **Required**

  Written under the node's parameters; any of these can also be overridden per target via `overwrites:`.

  - **freq_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the drive IF sweep. Relative: the
    bus's current `Parameter.IF` is added to it.

  - **drive_amplitude** : `float`.
    Drive pulse amplitude.

  - **drive_duration** : `int`.
    Drive pulse duration in ns.

  - **readout_amplitude** : `float`.
    Readout pulse amplitude.

  - **readout_duration** : `int`.
    Readout pulse duration in ns.

  - **averages** : `int`.
    Number of hardware averages.

  - **relax_duration** : `int`.
    Relaxation wait between shots in ns.

  ### **Optional**

  - **drive_gain** : `float`, default `1`.
    Output level of the drive source: power in dBm on a Rohde & Schwarz LO, gain on a
    QCM-RF.

  - **ringup_time** : `int`, default `0`.
    Time of the pulse needed to excite the resonator for readout (ns).

  - **overlap_time** : `int`, default `0`.
    Overlap between the end of the drive pulse and the start of the readout pulse (ns).

  ## **Example**
  ```yml
  two_tone:
    freq_sweep: [-300e6, 300e6, 301]
    drive_amplitude: 0.5
    drive_duration: 4000
    readout_amplitude: 0.075
    readout_duration: 2000
    averages: 1000
    relax_duration: 200000
    drive_gain: 0.8
    overlap_time: 4000
  ```
  ([PR #8](https://github.com/qilimanjaro-tech/seqtante-open/pull/8))

- Added Single Tone (registered as `single_tone` in the calibration tree) resonator spectroscopy experiment along with its fitter as an available fluxonium node.
  It sweeps the readout IF at zero flux, fits a Lorentzian to the rotated signal quadrature and writes the fitted IF back into the readout bus.
  The parameters used are:
  ### **Required**

  Written under the node's parameters; any of these can also be overridden per target via `overwrites:`.

  - **if_sweep** : `tuple[float, float, int]`.
    `np.linspace` args (start, stop, num) for the readout IF sweep. Relative: the
    bus's current `Parameter.IF` is added to it.

  - **readout_amplitude** : `float`.
    Readout pulse amplitude.

  - **readout_duration** : `int`.
    Readout pulse duration in ns.

  - **averages** : `int`.
    Number of hardware averages.

  - **relax_duration** : `int`.
    Relaxation wait between shots in ns.

  ### **Optional**

  - **ringup_time** : `int`, default `0`.
    Time of the pulse needed to excite the resonator for readout (ns).

  ## **Example**
  ```yml
  single_tone:
    if_sweep: [-1.5e6, 1.5e6, 201]
    readout_amplitude: 0.075
    readout_duration: 2000
    averages: 4000
    relax_duration: 200000
  ```
  ([PR #9](https://github.com/qilimanjaro-tech/seqtante-open/pull/9))

## Improved Documentation

- Added the licence to the pyproject and filled place-holder on `LICENCE`.

  Added **`towncrier`** as our changelog system. Each PR now adds a news fragment under **changes/** instead of editing a shared file, and `towncrier` build compiles them into `CHANGELOG.md` at release time.
  ([PR #5](https://github.com/qilimanjaro-tech/seqtante-open/pull/5))

## Misc

- Added a test suite. There was none before.

  ### Fixtures

  `tests/experiments/conftest.py` provides three, available to every test under `tests/experiments/`:

  - **`platform`** -- a real `Platform` built offline from a runcard the test module names via a module-level `RUNCARD_PATH`. Parameters are genuine; nothing connects, so no hardware is touched.
  - **`mock_db_manager`** -- replaces the `output_controller.db_manager` singleton with a `MagicMock`.
  - **`mock_recorder`** -- patches callables with recording stubs. Stubs are autospecced by default, so a call that would not match the real signature raises `TypeError` instead of being recorded. A test cannot keep asserting on a parameter the production code has renamed or dropped.

  ### Testing a fit class

  `FittingTestCase` in `tests/experiments/fitting/harness.py` gives a fit class inherited tests for `__init__`, `fit` and `plot` from four declarations:

  ```python
  class TestT1Fit(FittingTestCase):
      FIT_CLASS = T1Fit
      DATA = "t1_fit.h5"  # a committed .h5, or a builder called at test time
      INIT = {"measurement_id": 1, "target": "q1"}
      EXPECTED = {"optimized_params.thresh.1": pytest.approx(DECAY_RATE, rel=0.02)}
  ```

  The suite covers a constructor that blows up on the data, an output already populated before `fit()` runs (which would make `EXPECTED` meaningless), every `EXPECTED` path after `fit()`, a `plot()` that writes nothing or leaks an open matplotlib figure, a `plot()` that silently does nothing when `path` is `None`, a public method beyond `fit`/`plot` with no test of its own, and `INIT` drifting from a renamed constructor argument. Both matplotlib and plotly display paths are counted. `python -m tests.experiments.fitting.data.make_data` rewrites every committed `.h5` from its builder. See `tests/experiments/fitting/README.md`.

  ### The registry discovers both sides

  `test_registry.py` checks that every fit class has a case and that each case sits at the mirror of its fit class's module. Both sides come from subclass discovery rather than a hand-maintained list, so a fit class landing without a test fails on its own. It also catches a case pointed at something that has `fit`/`plot` but never inherited `FittingClass`.

  ### The harness reads runcards with ruamel

  `default_platform_before()` loads the test runcard with `ruamel.yaml`, the loader `qililab.data_management.build_platform` uses. PyYAML reads YAML 1.1, where an exponent with no decimal point and no sign is a string, so `ramp_rate: 2e7` arrived as `"2e7"` in the harness and as `20000000.0` everywhere else. `build_platform(default_platform_before()).to_dict()` now equals `build_platform(runcard=path).to_dict()`.
  ([PR #4](https://github.com/qilimanjaro-tech/seqtante-open/pull/4))

- Updated dependencies (all the added deps where already installed second-hand):
  - qililab {==0.32.0 (pypl)} -> {>=0.35.0(AWS code artefact)}
  - scikit-optimize (removed)
  - kaleido {==0.2.1} -> {>=1.2.0}
  - networkx (added) {>=3.1}
  - numpy (added) {>=1.26}
  - plotly (added) {>=6.0.0}
  - qcodes (added) {}
  - ruamel-yaml (added) {>=0.18.10}
  - scipy (added) {>=1.15}
  - xarray (added) {>=1.15}

  dev group:
  - h5py (added) {>=3.13.0}
  - towncrier (added) {>=25.8.0}

  Updated **python** version to 3.11-3.13.

  Added the **publish** workflow to build the wheels and publish them in AWS code artefact when making a release.
  ([PR #5](https://github.com/qilimanjaro-tech/seqtante-open/pull/5))


## 0.0.1

### Improvements

- First release.

## 0.0.2

### Bug fixes

- Improve readme.