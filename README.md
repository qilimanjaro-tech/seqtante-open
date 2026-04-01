# Seqtante

Calibration tools for superconducting devices. 

![alt text](https://github.com/qilimanjaro-tech/seqtante/blob/main/img/seqtante_logo.png)

Seqtante is a library/CLI that provides tools for characterising and calibrating superconducting devices.

Its main feauture is an **automatic calibration workflows** for quantum processors using a **YAML file** to declare the calibration steps (nodes) and their **dependencies** (edges).
Under the hood it builds a directed acyclic graph (DAG), injects parameters between steps when needed, and executes everything in a **topological order** on a `qililab` platform.

This repository’s main entrypoint is:

```
seqtante/autocalibration.py
```

It implements:

* YAML loading and validation
* Graph construction with **NetworkX**
* A barrier that ensures **all 1-qubit (“1q”) steps finish before any 2-qubit (“2q”) step starts**
* Execution within a `qililab` platform session
* An optional graph image artifact (`img/calibration_graph.png`)
* Per-run data folder (`data/run_<timestamp>/`)

---

## Contents

* [Requirements](#requirements)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [YAML Schema](#yaml-schema)
* [CLI Usage](#cli-usage)
* [How It Works](#how-it-works)
* [Extending with New Experiments](#extending-with-new-experiments)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Requirements

* Python **3.10+**
* [`qililab`](https://github.com/qilimanjaro-tech/qililab)
* [`ruamel.yaml`](https://pypi.org/project/ruamel.yaml/)
* [`networkx`](https://networkx.org/)
* [`qcodes`](https://github.com/QCoDeS/Qcodes)

Installable with `pip` as shown below.

---

## Installation

```bash
# clone your project
git clone https://github.com/qilimanjaro-tech/seqtante.git
cd seqtante

# (recommended) create a virtualenv
python -m venv .venv
source .venv/bin/activate

# install
pip install -U pip
pip install -r requirements.txt   # if present
# or, editable
pip install -e .
```

---

## Quick Start

1. Prepare a **runcard** (your `qililab` platform description), e.g. `platform.yaml`.
2. Write a **calibration YAML** that declares the steps and dependencies (see below).
3. Run:

```bash
python -m seqtante.autocalibration \
  --platform_path path/to/platform.yaml \
  --config_path   path/to/calibration.yaml
```

Artifacts:

* Databases entry per calibration experiment
* A timestamped data folder: `data/run_YYYY-MM-DD HH:MM:SS/`

---

## YAML Schema

Minimal structure:

```yaml
# calibration.yaml
nodes:
  - name: "Ramsey_1"
    experiment: "ramsey"
    kind: "1q"
    parameters:
      qubit_idx: 0
      hw_avg: 2000
      repetition_duration: 200000
      wait_sweep_values: [0, 2000, 31]
      if_sweep_values: [-2e6, 2e6, 11]
  - name: "SSRO_1"
    experiment: "ssro"
    kind: "1q"
    parameters:
      qubit_idx: 0
      hw_avg: 1
      repetition_duration: 200000
      num_bins: 2000
  - name: "Drag_1"
    experiment: "drag"
    kind: "1q"
    parameters:
      qubit_idx: 0
      hw_avg: 2000
      repetition_duration: 200000
      sweep_values: [-3, 3, 31]
  - name: "Flipping_1"
    experiment: "flipping"
    kind: "1q"
    parameters:
      qubit_idx: 0
      hw_avg: 2000
      repetition_duration: 200000
      iterations: 20

dependencies:
  - ["Ramsey_1", "SSRO_1"]
  - ["SSRO_1", "Drag_1"]
  - ["Drag_1", "Flipping_1"]
```

### Notes

* `kind` is **strongly recommended**:

  * `"1q"` → single-qubit step
  * `"2q"` → two-qubit step

* If `kind` is omitted, we use a **fallback**:

  * If `parameters` has `qubit_idx` → assume `"1q"`
  * If it has `control_qubit`+`target_qubit` or `pairs` → assume `"2q"`
  * Else default to `"1q"`

* Dependencies are **honored as written**, and the runner also adds a **barrier**: *all 1q nodes must finish before any 2q node starts.* (Helpful to avoid 2q calibration before 1q is stable.)

---

## CLI Usage

```bash
python -m seqtante.autocalibration \
  --platform_path ./platforms/my_device.yaml \
  --config_path   ./calibrations/calibration.yaml
```

Options:

* `--platform_path` : path to `qililab` runcard / platform configuration
* `--config_path`   : path to the YAML calibration DAG

---

## How It Works

* Loads your YAML with `ruamel.yaml`.
* Creates a `CalibrationGraph` and one `CalibrationNode` **per YAML node** (no per-qubit node explosion; parallelism can be handled inside each experiment).
* Wires the edges from `dependencies`.
* Splits nodes into 1q / 2q groups (via `kind`), then connects **all 1q ends → all 2q starts** to enforce the global barrier.
* Opens a `qililab` `platform.session()` and calls `run_calibration()` (topological order).

Data flow: if an experiment node produces a `result` dict, it can be injected as parameters into downstream nodes (this logic typically resides in `CalibrationGraph.run_calibration()` and each `CalibrationNode.run()`).

---

## Extending with New Experiments

Map experiment names to callables in `experiment_functions_dict`:

```python
# seqtante/experiments/experiment_functions.py
experiment_functions_dict = {
    "two_tone": two_tone_runner,
    "rabi": rabi_runner,
    "cz_phase": cz_phase_runner,
    # add yours here
}
```

Each runner should accept `(platform, parameters)` (or the signature used in your project), perform the measurement/fit, and return a dict with results to pass downstream (optional).

---

## Troubleshooting

* **Unknown experiment**:
  Ensure the `experiment` string exists in `experiment_functions_dict`.

* **Cycle in dependencies**:
  The DAG must be acyclic. Remove circular edges in your YAML.

* **2q starts too early**:
  Verify each 2q node has `kind: "2q"`; otherwise it may be classified as 1q.

---

## License

Apache 2.0.
See `LICENSE` for details.

---

## Acknowledgements

Built on top of:

* [`qililab`](https://github.com/qilimanjaro-tech/qililab) for platform and hardware control
* [`NetworkX`](https://networkx.org) for DAG plumbing
* [`ruamel.yaml`](https://yaml.readthedocs.io) for robust YAML I/O
