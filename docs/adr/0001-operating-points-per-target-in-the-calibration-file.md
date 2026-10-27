---
status: accepted
---

# Operating points live per target in the calibration file, and experiment sections reference them

Seqtante-Open calibrates the device; qili-experiments, run through qili-runtime and QiliSDK, later operates it from the runcard and the calibration file alone. Today qili-experiments copies each qubit's fluxes and frequencies into every experiment section, so one recalibration would have to find and update every copy, and any copy it missed would silently run at a stale point. We decided that operating points are stored **once per target** in the calibration file, that each experiment section **names** the operating point it runs at and the pulses it plays, and holds only what is specific to it, and that Seqtante-Open writes operating points while qili-experiments changes its contract to read them. The runcard keeps only the baseline; operating-point values never reach it.

## The shape

```yaml
operating_points_version: 1
operating_points:
  q14:
    sweet_spot:
      flux: {z: 0.0, x: 0.0}
      drive:   {lo: 3.2e9,  if: -95e6}
      readout: {lo: 6.38e9, if: 63.4e6, attenuation: 48}
    reset:
      flux: {z: 0.615, x: 0.27}
      drive:   {lo: 2.9e9,  if: -100e6}
      readout: {lo: 6.38e9, if: 63.4e6, attenuation: 48, level_0: ..., level_1: ...}
experiments:
  rabi:
    q14: {operating_point: reset, pulses: {readout: readout_long, drive: drag}}
predistortion:
  fast_flux_q14_z: {b: [...], a: [...]}
# Pulses live in qililab's existing Calibration.waveforms, keyed by bus and pulse name:
#   readout_q14: {readout_long: Square(...)}, drive_q14: {drag: Drag(...)}
# AC crosstalk lives in qililab's existing Calibration.crosstalk_matrix_ac
```

- The EuroQCS tree uses two operating points per qubit, `sweet_spot` and `reset`. The qubit is read out at the reset point, so experiments take less time and the fridge does not heat up from moving to the sweet spot first. Names are free per target, so a later tree can add a separate readout point without changing the format. Targets are named `q14`, `c2_3`.
- Drive and readout store `lo` and `if` separately, because the instruments have a dedicated setting for each. When a frequency sweep needs the LO outside the sweep range, the helper that sets the operating point on the platform places it there; the stored values do not change. Changing an operating point's LO means recalibrating its IF.
- **Pulses are not part of an operating point.** They are named waveforms in `Calibration.waveforms`, per bus. A step in the calibration tree names the pulses it plays (ADR-0003), and an experiment section names the pulses its consumer plays. A recalibration whose analysis produces a pulse value (Rabi for a pi amplitude) writes into the named waveform.
- A calibration step writes its result to the operating point by default, and to the experiment section only when the calibration tree says so. Rabi at the reset point writes the |0⟩ and |1⟩ readout levels, which LZ uses to normalise. Levels taken from Rabi fold thermal population and SNR into the value; accepted until a single-shot readout calibration exists, and the operators' notebook does the same.
- Sweet-spot recalibration corrects the crosstalk flux offset, so the sweet spot sits at its nominal flux and every operating point of that qubit moves with it. The operators currently store the absolute flux instead; the experimentalists still have to confirm this.
- AC crosstalk and IIR predistortion are properties of the flux lines, so they sit at device level rather than inside the LZ section.
- Experiments set operating-point values on the in-memory platform, but the runcard is never saved from a run. A setting one experiment changes temporarily is restored by the node (ADR-0004), so it cannot leak into the next step.
- On a fresh calibration file, initial operating points come from the calibration tree.

## Considered options

- **Per-experiment copies** (qili-experiments today): rejected, because copies drift apart after the first recalibration.
- **Shared registry projected into the existing sections**, so qili-experiments would not change: rejected once the qili-experiments team agreed to change its contract.
- **Flat keys** such as `q0_operating_point_sweetspot_z`: rejected, because consumers would have to parse key strings and parked neighbours' bus names would end up encoded in keys.
- **`{Parameter: {bus: value}}`** from QHC-1608: replaced by named loops (`z`, `x`) and roles (`drive`, `readout`), which operators can read; Seqtante-Open maps them to bus aliases in one place.
- **Default pulses carried by the operating point**, with sections overriding what differs: rejected in review. Pulses are shared through `Calibration.waveforms` by name, which qili-runtime already reads, so the same pulse is not stored in a second place.
- **A separate `readout` point next to `reset`**: rejected for this tree, because both would hold the same values.

## Consequences

- qili-experiments must change its contract (read operating points from the target level and pulses by name) before it can run from a calibration file written in this shape. The pulse naming convention is agreed with the qili-experiments and qili-runtime developers before either side codes against it.
- The temporary operating-point format on the QHC-1608 branch is superseded, and the T1 node on QHC-1596 builds on it; both restart from this design, reusing parts.
- Applying an operating point and saving calibrated values move out of each experiment function and into `CalibrationNode` (ADR-0004, ADR-0006).
- Changing the readout pulse of a point invalidates its readout levels, so the Rabi step that writes them has to run again.

## Open questions

- **Parking neighbours.** Whether an operating point may set other elements' fluxes, such as the couplers the operators park around q4. Owner: experimentalists. Conflicts between simultaneous targets are out of scope while parallel execution is not supported.
