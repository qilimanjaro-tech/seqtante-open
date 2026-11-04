---
status: accepted
---

# Sweet-spot recalibration stores the measured flux bias, not a crosstalk-offset correction

ADR-0001 proposed that sweet-spot recalibration corrects the crosstalk flux offset, so the sweet spot sits at its nominal flux and every operating point of the qubit moves with it, and noted the operators instead store the absolute flux today — leaving the choice to its own ADR. We decided to **keep the operators' approach**: sweet-spot recalibration does **not** modify the crosstalk offset. It records where the qubit's sweet spot is at that moment, as a measured flux bias, which may differ from one day to the next. Confirmed by Gonzalo Troncoso on the EuroQCS December final-design review (2026-11-02).

## The decision

- The sweet-spot operating point stores the **measured flux bias** (`flux.z`, `flux.x`) found by the recalibration, as a value in the calibration file.
- Sweet-spot recalibration leaves the crosstalk matrix and its offset untouched. There is no coupling whereby recalibrating the sweet spot shifts the other operating points of the qubit.
- `flux.z` is therefore read and written as a plain measured value; the calibration-file model (QHC-1799) stores it as such, with no offset-correction arithmetic.

## Considered options

- **Correct the crosstalk offset** (ADR-0001's proposal): rejected. It would make every operating point of a qubit depend on the sweet-spot calibration, and it does not match how the operators work; the sweet-spot bias is not stable enough day to day to treat as a fixed nominal with an offset.
- **Store the measured flux bias** (operators' approach): chosen. Each recalibration records the bias it measured, standing on its own.

## Consequences

- The calibration-file model (QHC-1799) stores `flux.z` as a measured value — no absolute-vs-offset ambiguity, no offset-correction machinery.
- Two-tone vs flux (task 7) and the reset-point search (task 9) write the measured sweet-spot bias; they do not touch the crosstalk matrix.
- This supersedes the sweet-spot/offset-correction point in ADR-0001, which anticipated being settled here.
