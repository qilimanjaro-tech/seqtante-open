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

- [Seqtante-open](#seqtante-open)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [CLI Usage](#cli-usage)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

---

## Requirements

* Python **3.11-3.13**
* [`qililab`](https://github.com/qilimanjaro-tech/qililab)
* [`ruamel.yaml`](https://pypi.org/project/ruamel.yaml/)
* [`networkx`](https://networkx.org/)
* [`qcodes`](https://github.com/QCoDeS/Qcodes)

Installable with `pip` as shown below.

---

## Installation

Releases are published to Qilimanjaro's internal CodeArtifact index, together with `qililab` and `qpysequence`. With that index configured, install an exact version:

```bash
uv pip install "seqtante-open==0.1.2"
```

Every merge to `main` that passes Tests and Code Quality also publishes a development version, for example `0.1.3.dev2`, where `2` is the number of merges since the last release. Installers skip development versions unless you ask for one, so pin the exact version you need in the same way. `import seqtante_open; seqtante_open.__version__` shows the version installed. Quote it in bug reports and calibration records.

For development, clone the repository and sync the environment with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/qilimanjaro-tech/seqtante-open.git
cd seqtante-open
uv sync
```

### Releases

The version comes from git tags, so there is no version to edit in `pyproject.toml`. To cut a release, run `towncrier build --version X.Y.Z`, merge that pull request, then publish a GitHub release whose tag is the bare version (for example `0.1.2`, no `v` prefix) on the merge commit. The Publish workflow builds the tagged commit and fails if the built version differs from the tag.

---

## Quick Start

1. Prepare a **runcard** (your `qililab` platform description), e.g. `platform.yaml`.
2. Write a **calibration YAML** that declares the steps and dependencies (see below).
3. Run:

```bash
python -m seqtante_open.autocalibration \
  --platform_path path/to/platform.yaml \
  --config_path   path/to/calibration.yaml
```

Artifacts:

* Databases entry per calibration experiment
* A timestamped data folder: `data/run_YYYY-MM-DD HH:MM:SS/`

## CLI Usage

```bash
python -m seqtante_open.autocalibration \
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
