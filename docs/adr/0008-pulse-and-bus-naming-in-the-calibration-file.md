---
status: accepted
---

# Pulses are named by purpose per bus, and bus keys keep qililab's existing convention

ADR-0001 stores operating points per target and has each experiment section name the operating point it runs at and the pulses it plays, with pulses living as named waveforms in qililab's `Calibration.waveforms`. It left two things to a cross-team agreement with the qili-experiments and qili-runtime developers, to be settled before either side codes against the format: how pulses are named, and the exact shape of an experiment section. We decided that **bus keys keep the hardware/crosstalk convention already in qililab**, that **pulse names are by purpose, `lower_snake` and unique per bus, carrying their operating point as a `-`-joined suffix where a purpose is calibrated at more than one point**, and that a **section names only its operating point and its pulses (role → name)**. The operating point holds no pulse references; the pulse name is where the point lives. Agreed on QHC-1796 (Juanjo Martinez with Ely Farrés, Natàlia Padilla and Oscar).

## The shape

```yaml
# Calibration.waveforms — keyed by BUS, then by PULSE NAME
waveforms:
  readout_q14:
    readout-reset: Square(...)      # readout pulse calibrated at the reset point
    # readout-<point>: Square(...)  # one per operating point in scope (e.g. the point t2 runs at)
  drive_q14:
    pi-reset: Drag(...)             # pi gate calibrated at the reset point
    # pi-<point>: Drag(...)
  flux_z_q14: {}                    # no pulse played by name in the December 1q tree

experiments:
  rabi:
    q14:
      operating_point: reset
      pulses: {readout: readout-reset, drive: pi-reset}
```

- **Bus keys** keep qililab's convention, not a new one. Drive and readout are `<line>_<target>` (`drive_q14`, `readout_q14`). Flux is **axis-first**: `flux_<axis>_<target>`, axis ∈ {`x`, `y`, `z`} — `flux_z_q14`, `flux_z_c2_3`, not `flux_<target>_<axis>`. Flux is axis-first because `CrosstalkMatrix.flux_to_bias`, the bus sorter, the fit-manager target parsing and the runcards all read `flux_z_q1`; the target-first spelling (`flux_q1_z`) would break crosstalk calibration. The `(target, line) → bus alias` mapping lives in one place in Seqtante-Open; no consumer encodes bus names. *(Updated 2026-11-03: this ADR first recorded the target-first spelling `flux_q1_z`; an internal change to qililab's flux bus naming flipped it to axis-first `flux_z_q1`, and this ADR and the code follow it.)*
- **Pulse names** are short, stable, `lower_snake`, unique within their bus, and named by **purpose** — what the pulse does — not by shape; the shape is the waveform object (`Square`, `Drag`). Where the same purpose is calibrated at more than one operating point, the name carries the point as a suffix joined by `-`: `<purpose>-<operating_point>`. The separator is `-`, not `_`, because `_` already joins words within a purpose (`pi_half`, `readout_long`), so `-` keeps the split between purpose and point unambiguous: `readout-reset`, `pi-reset`, `pi_half-reset` (and `x90-reset` when those gates arrive). A tree with two operating points therefore holds one readout and one pi pulse per point. Recalibrating writes new parameters into the same named waveform; the name does not change.
- **Experiment sections** are keyed `experiments.<experiment>.<target>` and hold only the operating-point name, a `pulses` map (role → pulse name, role ∈ {`readout`, `drive`, `flux`}), and fields genuinely specific to the section. `pulses` is sparse: each experiment registers a default per role and a section lists only the roles it overrides; a section running every default needs no `pulses` block. The named pulse must exist under the bus that `(target, role)` resolves to.
- **The operating point holds no pulse references** (ADR-0001). Descriptiveness across points comes from the name suffix, not from putting pulses back on the operating point, so that decision stands.
- **Parse rules**, so the three repos parse identically: a pulse name matches `^[a-z][a-z0-9_]*(-[a-z0-9_]+)?$` — a `lower_snake` purpose, optionally followed by `-<operating_point>`; roles are the closed set `{drive, readout, flux}`; an unknown role or a pulse name absent from the resolved bus is a load-time error naming the section; for flux, bus-key parsing reads the axis first (the axis, when present, follows `flux_`, then the target).

## Considered options

- **Shape names, or deferring the shape-vs-purpose choice** (name December by shape, `drag`, decide later): rejected. Purpose names are more descriptive, and with t2 in scope the tree already holds more than one calibrated pulse per bus, so the scheme is fixed now rather than renamed later across three repositories.
- **`_` as the purpose/point separator** (`pi_reset`): rejected, because `_` already separates words inside a purpose (`pi_half`, `readout_long`), so a reader and a parser cannot tell where the purpose ends; `-` makes the split unambiguous.
- **Pulse references on the operating point**, keyed by use there (`operating_points.<target>.<point>.waveforms: {drive: pi-reset, readout: readout-reset}`): rejected. It reopens ADR-0001's decision to keep pulses out of the operating point and the one-way-door format; the operating-point suffix in the pulse name gives the same descriptiveness without it.
- **Flat keys** (`q14_readout-reset`): rejected, consistent with ADR-0001 — consumers would parse key strings and neighbours' names would leak into keys.

## Consequences

- The calibration-file format stays a one-way door: qili-experiments and qili-runtime read it, so this vocabulary is fixed before either codes against it (the calibration-file model in QHC-1797 codes against it directly).
- The December drive pulse is named `pi-reset` (purpose) rather than `drag` (shape). ADR-0001's shape YAML shows `pulses: {readout: readout_long, drive: drag}`; under this ADR those values read `{readout: readout-reset, drive: pi-reset}`. The calibration-tree step names (`t2__drag`, `rabi__amplitude_drag_drive`) are step identifiers, not pulse names, and do not change.
- The December 1q tree plays no named flux pulse (flux moves are operating-point values, not `Calibration.waveforms` entries), so `flux_z_q14` is empty and the `flux` role is absent from sections for December.

## Open questions

- **Named flux pulses.** Whether the qili-experiments or qili-runtime developers need `flux` in the `pulses` map for a later tree. None is needed for December; left open until a tree plays a flux pulse by name.
