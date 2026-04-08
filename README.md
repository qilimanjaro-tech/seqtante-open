# Seqtante-open

Open-source version of Qilimanjaro calibration tools for analog superconducting devices. 


Seqtante-open is a library/CLI that provides tools for characterising and calibrating analog superconducting devices.

Its main feauture is an **automatic calibration workflows** for quantum processors using a **YAML file** to declare the calibration steps (nodes) and their **dependencies** (edges).
Under the hood it builds a directed acyclic graph (DAG), injects parameters between steps when needed, and executes everything in a **topological order** on a `qililab` platform.

This repository’s main entrypoint is:

```
seqtante-open/autocalibration.py
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
git clone https://github.com/qilimanjaro-tech/seqtante-open.git
cd seqtante-open

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

## CLI Usage

```bash
python -m seqtante.autocalibration \
  --platform_path ./platforms/my_device.yaml \
  --config_path   ./calibrations/calibration.yaml
```

Options:

* `--platform_path` : path to `qililab` runcard / platform configuration
* `--config_path`   : path to the YAML calibration DAG

## License

Apache 2.0.
See `LICENSE` for details.

---

## Acknowledgements

Built on top of:

* [`qililab`](https://github.com/qilimanjaro-tech/qililab) for platform and hardware control
* [`NetworkX`](https://networkx.org) for DAG plumbing
* [`ruamel.yaml`](https://yaml.readthedocs.io) for robust YAML I/O
