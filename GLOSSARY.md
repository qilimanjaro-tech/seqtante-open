# Seqtante-Open

Automatic calibration and benchmarking of qubits and couplers, driven by a calibration tree of experiments run against a qililab platform.

## Language

### Calibration runs

**Calibration tree**:
The configuration of one Seqtante-Open run: which pipelines and steps run on which targets, in what dependency order, with what sweep settings and benchmark thresholds.
_Avoid_: Calibration graph, pipeline config, calibration file

**Pipeline**:
A named, ordered group of steps in a calibration tree that calibrates or benchmarks one thing, such as the sweet spot, T1 or the reset point for Landau-Zener. It is the unit an operator turns on or off for a run. A failure for a target stops, for that target only, the rest of the pipeline and every pipeline whose `after` names a failed or skipped step.
_Avoid_: Branch, node type

**Step**:
One named entry of a pipeline: an experiment run with a given analysis at a given operating point. The same experiment can appear in many steps.
_Avoid_: Pipeline entry, node type

**Experiment**:
A measurement procedure on a target, such as two-tone, Rabi or T1, independent of where and why it runs.
_Avoid_: Experiment function, measurement

**Node**:
The implementation of an experiment in Seqtante-Open's code; the run builds one for each step. A step is what the tree says, a node is what runs it.
_Avoid_: Experiment class

**Analysis**:
The fit that turns a step's measurement into results, and so decides what the step extracts and writes (for example readout levels or pi amplitude from the same Rabi measurement). It is read-only and runs once, after the measurement; an experiment whose next measurement depends on a fit, such as the reset-point search, fits inside its own loop instead.
_Avoid_: Fitting, model, post-processing

**Recalibration**:
A step whose result is written back to the calibration file, updating an operating point, a pulse or the crosstalk offset. It is a kind of step, not of experiment: the same experiment can run in several steps that write different things (Rabi for readout levels or a pi amplitude).
_Avoid_: Calibration experiment, tuning

**Benchmark**:
A step that measures a figure of merit of a target (T1, T2, Landau-Zener agreement) and writes nothing back to the calibration file; it ends in a verdict.
_Avoid_: Characterisation, test

**Verdict**:
The qualitative outcome of a benchmark against its thresholds: Passed, Warning or Failed.
_Avoid_: Status, grade, result

**Report**:
The PDF produced at the end of a run, gathering calibrated values, fitted plots and benchmark verdicts per target, and the steps that failed or were skipped.
_Avoid_: Summary, dashboard

### Operating points

**Calibration file**:
The per-device file holding what calibration has learnt about the chip: flux crosstalk and offsets, operating points, pulses and experiment sections. Together with the runcard it is all a consumer needs to operate the device.
_Avoid_: Calibration, cal file, calibration tree

**Runcard**:
The per-device file describing the instruments and buses and their baseline settings.
_Avoid_: Platform file, config

**Operating point**:
A named configuration of one qubit or coupler (its fluxes and its drive and readout frequencies) that experiments place the target in before measuring. It belongs to the target and is shared by every experiment that runs there.
_Avoid_: Operation point, working point, bias point

**Sweet spot**:
The operating point at the bottom of the qubit's frequency-versus-flux parabola, where the frequency is first-order insensitive to flux noise.
_Avoid_: SS, flux sweet spot, symmetry point

**Reset point**:
The operating point at which the qubit frequency equals a fixed target frequency, where the qubit is read out and to which it returns between and after experiments.
_Avoid_: Idle point, park point, home, RO point

**Pulse**:
A named waveform played on one bus of a target, such as a readout pulse or a pi pulse, stored once in the calibration file and chosen by name wherever it is played.
_Avoid_: Default pulse, pulse settings

**Readout levels**:
The measured signal of |0⟩ and |1⟩ at an operating point, used to turn a signal into a population.
_Avoid_: Rabi levels, amplitudes

**Experiment section**:
The part of the calibration file holding what is specific to one experiment on one qubit: which operating point it runs at, which pulses it plays, and its own parameters.
_Avoid_: Experiment profile, experiment settings, experiment config
