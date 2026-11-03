---
status: accepted
---

# An operating point parks the fluxes of neighbouring elements

ADR-0001 stores an operating point per target with the target's own `flux: {z, x}`, and left open (its open question on parking) whether a point may also set other elements' fluxes, such as the couplers parked around a qubit. We decided that **it does**: an operating point carries, alongside the target's own flux, the parked fluxes of the elements it holds steady — the attached couplers, and the neighbouring qubits where the operators park them. This is what the operators have done since the June delivery. Confirmed by Gonzalo Troncoso on the EuroQCS December final-design review (2026-11-02).

## The shape

An operating point holds its target's flux and the parked fluxes of its neighbours, each keyed by the element whose flux is parked:

```yaml
operating_points:
  q14:
    reset:
      flux: {z: 0.615, x: 0.27}          # the target's own flux
      parked_flux:                       # fluxes held steady while q14 is at this point
        c13_14: {z: 0.02}
        c14_15: {z: -0.01}
      drive:   {lo: 2.9e9,  if: -100e6}
      readout: {lo: 6.38e9, if: 63.4e6, attenuation: 48}
```

- The parked elements are the couplers attached to the target, and the neighbouring qubits the operators park, named by their target (`c13_14`, `q15`). The calibration-file model (task 2, QHC-1799) fixes the exact key, mapped to a flux bus through the one `(target, line) -> bus alias` table (ADR-0001, ADR-0008); `parked_flux` above is the proposed name and the model task settles it.
- Parked fluxes are set together with the point and restored by the framework after the step, like the point's own flux (ADR-0001, ADR-0004).
- A point parks only the elements it names. Two targets that would park the same neighbour at different values cannot run at once; this does not arise while parallel execution is unsupported (ADR-0001 dropped D28), so conflict resolution between simultaneous targets is out of scope for December.

## Considered options

- **No parking in the file; neighbours parked by a separate mechanism**: rejected. The parked fluxes belong to the operating point — they are part of what "put q14 at its reset point" means — and the operators already treat them that way.
- **A single flat flux map over every element**, the target and its neighbours in one dict: rejected. The target's own flux is read on nearly every path, so keeping it distinct from the parked neighbours keeps the common case direct and the parked set optional.

## Consequences

- The calibration-file operating-point model (QHC-1799) carries parked fluxes for neighbour buses, not only the target's own `flux`. This resolves ADR-0001's open question on parking.
- Applying a point sets its parked fluxes too; the framework restores them after the step (ADR-0004).
- Conflicts between simultaneous targets parking the same neighbour stay out of scope until parallel execution is supported.
