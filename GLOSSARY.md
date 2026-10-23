# Seqtante-Open

Automatic calibration and benchmarking of qubits and couplers, driven by a calibration tree of experiments run against a qililab platform.

## Language

### Calibration runs

**Calibration tree**:
The configuration of one Seqtante-Open run: which experiments run on which targets, in what dependency order, with what sweep settings and benchmark thresholds.
_Avoid_: Calibration graph, pipeline config, calibration file

**Step**:
One named entry of a calibration tree: an experiment run with a given analysis at a given operating point, after another step. The same experiment can appear in many steps.
_Avoid_: Node, node type, pipeline entry

**Analysis**:
The fit that turns a step's measurement into results, and so decides what the step extracts and writes (for example readout levels or pi amplitude from the same Rabi measurement).
_Avoid_: Fitting, model, post-processing

**Recalibration**:
An experiment whose result is written back to the calibration file, updating an operating point or a pulse.
_Avoid_: Calibration experiment, tuning

**Benchmark**:
An experiment that measures a figure of merit of a target (T1, Landau-Zener agreement) and writes nothing back to the calibration file; it ends in a verdict.
_Avoid_: Characterisation, test

**Verdict**:
The qualitative outcome of a benchmark against its thresholds: Passed, Warning or Failed.
_Avoid_: Status, grade, result

**Report**:
The PDF produced at the end of a run, gathering calibrated values, fitted plots and benchmark verdicts per target.
_Avoid_: Summary, dashboard

### Operating points

**Calibration file**:
The per-device file holding what calibration has learnt about the chip: flux crosstalk and offsets, operating points, and experiment sections. Together with the runcard it is all a consumer needs to operate the device.
_Avoid_: Calibration, cal file, calibration tree

**Runcard**:
The per-device file describing the instruments and buses and their baseline settings.
_Avoid_: Platform file, config

**Operating point**:
A named configuration of one qubit or coupler (its fluxes, its drive and readout frequencies, and the default pulses used there) that experiments place the target in before measuring. It belongs to the target and is shared by every experiment that runs there.
_Avoid_: Operation point, working point, bias point

**Sweet spot**:
The operating point at the bottom of the qubit's frequency-versus-flux parabola, where the frequency is first-order insensitive to flux noise.
_Avoid_: SS, flux sweet spot, symmetry point

**Experiment section**:
The part of the calibration file holding what is specific to one experiment on one qubit: which operating point it runs at, its own parameters, and any pulse settings it does not share with other experiments.
_Avoid_: Experiment profile, experiment settings, experiment config

**Readout point**:
The operating point at which the qubit frequency equals a fixed target frequency chosen for reading it out.
_Avoid_: RO point, RO flux point, idle point

**Reset point**:
The operating point a target is returned to between and after experiments.
_Avoid_: Park point, home
