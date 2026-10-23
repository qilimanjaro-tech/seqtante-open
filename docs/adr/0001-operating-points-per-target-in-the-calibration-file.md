---
status: proposed
---

# Operating points live per target in the calibration file, and experiment sections reference them

Seqtante-Open calibrates the device; qili-experiments, run through qili-runtime and QiliSDK, later operates it from the runcard and the calibration file alone. Today qili-experiments copies each qubit's fluxes and frequencies into every experiment section, so one recalibration would have to find and update every copy, and any copy it missed would silently run at a stale point. We decided that operating points are stored **once per target** in the calibration file, that each experiment section **names** the operating point it runs at and holds only what is specific to it, and that Seqtante-Open writes operating points while qili-experiments changes its contract to read them. The runcard keeps only the baseline; operating-point values never reach it.

## The shape

```yaml
operating_points_version: 1
operating_points:
  q14:
    readout:
      flux: {z: 0.615, x: 0.27}
      drive:   {lo: 3.2e9,  if: -110e6}
      readout: {lo: 6.38e9, if: 63.4e6, amplitude: 0.9, duration: 2000, ringup: 3000, attenuation: 48,
                level_0: ..., level_1: ...}
    sweet_spot: {...}
    reset: {...}
experiments:
  rabi:
    q14: {operating_point: readout, readout: {amplitude: 0.4}, drive: {...}}
predistortion:
  fast_flux_q14_z: {b: [...], a: [...]}
# AC crosstalk lives in qililab's existing Calibration.crosstalk_matrix_ac
```

- Operating-point names are `sweet_spot`, `readout` and `reset`; targets are named `q14`, `c2_3`.
- Drive and readout store `lo` and `if` separately, because the instruments have a dedicated setting for each. Changing an operating point's LO means recalibrating its IF.
- An operating point carries the **default pulses** for experiments run there; an experiment section overrides only what differs.
- A calibration step writes its result to the operating point by default, and to the experiment section only when the calibration tree says so. Rabi at the readout point writes the |0⟩ and |1⟩ readout levels, which LZ uses to normalise.
- Sweet-spot recalibration corrects the crosstalk flux offset, so the sweet spot sits at its nominal flux and every operating point of that qubit moves with it. The operators currently store the absolute flux instead.
- AC crosstalk and IIR predistortion are properties of the flux lines, so they sit at device level rather than inside the LZ section.
- On a fresh calibration file, initial operating points come from the calibration tree config, and the run refuses to start if a step needs an operating point that nothing provides.

## Considered options

- **Per-experiment copies** (qili-experiments today): rejected, because copies drift apart after the first recalibration.
- **Shared registry projected into the existing sections**, so qili-experiments would not change: rejected once the qili-experiments team agreed to change its contract.
- **Flat keys** such as `q0_operating_point_sweetspot_z`: rejected, because consumers would have to parse key strings and parked neighbours' bus names would end up encoded in keys.
- **`{Parameter: {bus: value}}`** from QHC-1608: replaced by named loops (`z`, `x`) and roles (`drive`, `readout`), which operators can read; Seqtante-Open maps them to bus aliases in one place.
- **Named waveforms in `Calibration.waveforms`** for per-point pulses: rejected for now, because no consumer reads them.

## Consequences

- qili-experiments must change its contract before it can run from a calibration file written in this shape.
- The temporary operating-point format on the QHC-1608 branch is superseded, and the T1 node on QHC-1596 builds on it, so both need reworking.
- Applying an operating point and saving calibrated values move out of each experiment function and into `CalibrationNode`; experiments only measure, fit and return.

## Open questions

- **Parking neighbours.** Whether an operating point may set other elements' fluxes, such as the couplers the operators park around q4, and how two simultaneous targets that set the same bus to different values are handled.
