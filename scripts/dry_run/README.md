# Dry run

Runs a calibration tree through the real `CalibrationRun` with no hardware: every node builds its
QProgram, compiles it, writes its measurement and fit to a database and updates the calibration file,
on fake data.

```
cd db_docker && docker compose up -d && cd ..   # once: the local database, see db_docker/README.md
uv run python -m scripts.dry_run
```

Run it from the repository root, so `tests` (the fake-data builders) is importable.

## What is simulated

- **Platform**: built from the runcard (`tests/runcards/test_AQPU_runcard.yml` by default) and never
  connected.
- **QDAC-II**: the real qcodes driver on pyvisa-sim (`qdac_sim.py`), recording every SCPI command it
  receives.
- **Execution**: `execute_qprogram` compiles the QProgram as on hardware and starts the QDACs, then returns
  fake data on the sweep the node streams, instead of running the Qblox sequencers.
- **Fake data**: the fit tests' builders in `tests/experiments/fitting`, evaluated on the node's own sweep
  (`fake_data.py`).
- **Calibration file**: without `--calibration`, one is made up from the tree (`fake_calibration.py`): the
  operating points it names, for every target, with a square readout and a DRAG π pulse each. A file
  given is copied first, since the run writes to it.

## Options

| Option | Default |
|---|---|
| `--config_path` | `scripts/dry_run/tree.yml` (`q1` and `q2` of the test runcard) |
| `--platform_path` | `tests/runcards/test_AQPU_runcard.yml` |
| `--calibration` | made up from the tree |
| `--pipelines` | every enabled pipeline |
| `--database_ini`, `--database_section` | `db_docker/database.ini`, `[seqtante]` |
| `--data_root` | `data/dry_runs` |
| `--seed` | `0`, for the noise and for the fits that draw on numpy's global generator |
| `--no-report` | write the PDF report |

The tree's `storage` section is replaced: the run writes to the database section given, and its files to
its own folder. Only that section is used, with the schema it declares, or `autocal` when it declares none.
The run connects through a copy in a temporary folder, deleted when the run ends; the copy kept in the run
folder has no `passwd`, so a run folder can be shared.

Any database works, but the run writes a real calibration run with fake measurements and fits to it, which
qompass shows like any other. Point it at a production database only to mean that. The tables are created
only in db_docker's database; any other database must already have them.

## Outputs

Each run gets a folder `data/dry_runs/<date>_<time>/` with:

- `tree.yml`, `calibration.yml`, `database.ini`: what the run used, the database section without its password.
  The calibration holds what the run wrote.
- `seqtante/`: the run's outputs (plots, logs), as on hardware.
- `measurements/`: the `.h5` of every measurement.
- `executions/<n>_<experiment>_<target>/`: one folder per execution, with the Q1ASM of every bus, the SCPI
  each QDAC received since the previous execution (`<qdac>.scpi`, so the flux set for the operating point
  is there too), and `fake_data.json`, the step, experiment and values the fake data was built with.
- `<qdac>.scpi`: every SCPI command each QDAC received during the whole run, in order.

## Fake-data parameters

Every parameter is an environment variable `DRY_RUN_[<TARGET>_][<SCOPE>_]<NAME>`. The scope is a step of
the tree (`T1_SWEET_SPOT`) or an experiment (`RABI`), so a value can be set for one target, one step, one
experiment, or everything. The most specific key set wins, in this order:

1. `DRY_RUN_Q1_RABI_RESET_PI_AMPLITUDE`: target and step
2. `DRY_RUN_RABI_RESET_PI_AMPLITUDE`: step
3. `DRY_RUN_Q1_RABI_PI_AMPLITUDE`: target and experiment
4. `DRY_RUN_RABI_PI_AMPLITUDE`: experiment
5. `DRY_RUN_Q1_PI_AMPLITUDE`: target
6. `DRY_RUN_PI_AMPLITUDE`: everything

Names are upper case: target `q1` is `Q1`, step `rabi_reset` is `RABI_RESET`. Unset, a feature sits at
`FEATURE_AT` of the node's sweep, so the fit finds it whatever window the tree sweeps.

| Name | Sets | Unit | Unset |
|---|---|---|---|
| `FEATURE_AT` | where an unset feature sits in its sweep | fraction | 0.4 |
| `QUBIT_IF` | the two-tone line (`two_tone`, `reset_point_search`) | drive IF, Hz | `FEATURE_AT` of the IF sweep |
| `RESONANCE_IF` | the single-tone dip (`single_tone`) | readout IF, Hz | `FEATURE_AT` of the IF sweep |
| `SWEET_SPOT` | the vs-flux sweet spot (`two_tone_vs_flux`, `offset_calibration`) | the flux sweep's unit | `FEATURE_AT` of the flux sweep |
| `SWEET_SPOT_IF` | the IF of the vs-flux arc at the sweet spot | Hz | 0.6 of the IF sweep |
| `PI_AMPLITUDE` | the Rabi π amplitude (`rabi`) | drive amplitude | `FEATURE_AT` of the amplitude sweep |
| `T1` | the T1 decay (`t1`) | ns | `FEATURE_AT` of the wait sweep |
| `T2` | the Ramsey decay (`t2`) | ns | `FEATURE_AT` of the wait sweep |
| `NOISE` | the noise on I and Q | standard deviation, signal units | the builder's own, below |
| `FAIL` | force the measurement to fail: `fit` or `execution` | | no failure |

The values are absolute, in the node's own sweep: a value outside the sweep gives data the fit should
reject. These stop the run before it starts, so a typo is not silently ignored:

- a `DRY_RUN_*` variable that names no parameter, target, step or experiment of the tree;
- a `FAIL` that is neither `fit` nor `execution`;
- another value that is not a finite number;
- a negative `NOISE`, a `T1` or `T2` that is not positive, or a `PI_AMPLITUDE` of 0.

### Noise

`NOISE` is the standard deviation of the white noise added to I and to Q, in the units of the signal:

| Experiment | Signal | Unset `NOISE` |
|---|---|---|
| `two_tone`, `reset_point_search` | a peak 0.4 high | 0.002 |
| `single_tone` | a dip 0.8 deep | 0.002 |
| `offset_calibration` | a dip 0.8 deep | 0.002 |
| `two_tone_vs_flux` | a peak 1 high | 0.01 |
| `rabi` | a cosine of amplitude 1 | 0.02 |
| `t1`, `t2` | a decay from 1 | 0.01 |

Enough noise makes the fit's R² or uncertainty checks reject the step, as a bad measurement would. How much
depends on the fit: `rabi` still passes its 0.9 minimum R² with a noise of 0.3.

### Forcing a failure

- `FAIL=fit`: the measurement is noise of its own mean and spread, with no feature, so the analysis rejects
  it and the target fails the step.
- `FAIL=execution`: the QProgram is compiled and the QDAC programmed, then the execution raises, as a
  hardware error would; the target fails the step with that error.

```
# PowerShell
$env:DRY_RUN_Q1_T1 = "30000"; $env:DRY_RUN_RABI_NOISE = "0.3"; $env:DRY_RUN_Q2_T2_SWEET_SPOT_FAIL = "execution"
uv run python -m scripts.dry_run

# bash
DRY_RUN_Q1_T1=30000 DRY_RUN_RABI_NOISE=0.3 DRY_RUN_Q2_T2_SWEET_SPOT_FAIL=execution uv run python -m scripts.dry_run
```

Unset them afterwards in PowerShell with `Remove-Item Env:DRY_RUN_*`.

## Limitations

- **The data does not move with flux.** `reset_point_search` converges only if its first measurement is
  already inside the target window.
- **The vs-flux arcs have a fixed curvature.** The `two_tone_vs_flux` and `offset_calibration` builders take
  it from their tests' constants, not from the sweep they are given, so on a wide flux sweep the arc can leave
  the IF window and the fit fails. A narrow sweep sees an almost flat arc.
- **`landau_zener`** has no node yet, so a tree with that experiment does not run.
- **Qblox** programs are compiled, never run: nothing is checked on the acquisition side.
