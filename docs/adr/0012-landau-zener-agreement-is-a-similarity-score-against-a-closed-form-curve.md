---
status: accepted
---

# Landau-Zener agreement is a similarity score against a closed-form theory curve

ADR-0002 left two questions open for the Landau-Zener benchmark: where the theory curve comes from (O2) and which number measures agreement with it (O3). Until they were answered, the verdict was "Not evaluated". Theory answered both on 25 November 2026, and gave a reference script and notebook that define the method. We decided:

- **The theory curve is the closed-form two-level Landau-Zener probability.** It is computed in seqtante-open from a few simulated points per qubit, stored in the calibration file.
- **Agreement is measured by a similarity score S.** S is computed three ways and all three are reported. The benchmark value is the `rmse` variant. S is higher-is-better.

seqtante-open reimplements the method in its own style (numpy, scipy, plotly). It does not import Theory's script.

## Theory curve

```
P_LZ(T) = exp(−π² Δ² T / Δε)
```

- **Δ** is the minimum gap and **Δε** the detuning span of the ramp, both in GHz.
- **T** is the duration of the ramp from `z_start` to `z_end`, in ns. The entry and exit ramps are not part of T.
- There is no closed form for Δ as a function of φx. Δ and Δε are therefore precomputed offline for a few φx values, and seqtante-open evaluates the formula at the φx values and durations it measured. No simulation file, no QiliSDK and no Hamiltonian diagonalisation in seqtante-open.

### The simulated points

The simulated points are a table **per qubit** (Theory confirmed the parameters differ per qubit). The table lives in the calibration file, in the qubit's `landau_zener` experiment section. It does not go in the calibration tree: the tree is edited by hand, and a per-qubit table would clutter it.

```yaml
experiments:
  landau_zener:
    q14:
      operating_point: reset
      simulated_points:  # illustrative values, not in the calibration file's flux convention
        - {phi_x: 0.225, z_start: 0.47, z_end: 0.53, min_gap: 0.0572, detuning_span: 1.887}
        - ...
```

Each row is one φx: where the ramp runs and the two numbers the formula needs. The offline simulation gives the effective qubit Hamiltonian `H = σx·X + σz·Z` at each point of the ramp, from which:

| Key | Meaning | Unit |
|---|---|---|
| `phi_x` | x flux of the ramp | Φ₀ (ADR-0011) |
| `z_start`, `z_end` | z flux at the start and end of the linear ramp | Φ₀ (ADR-0011) |
| `min_gap` (Δ) | minimum over the ramp of `E01 = 2·√(σx² + σz²)` | GHz |
| `detuning_span` (Δε) | `2·|σz(z_end) − σz(z_start)|` | GHz |

- The key names are a proposal. The calibration-file model in qili-qprograms settles them (ADR-0001 note: a format change is a qili-qprograms PR plus a `uv.lock` bump here).
- The flux values are in the calibration file's `flux.z`/`flux.x` convention. How the simulation's convention maps to it is an open question (below).
- The closed form assumes ε is linear in φz over the ramp, Δ is constant along it, and the ramp starts and ends far from the anticrossing (`min(|ε_start|, |ε_end|)/Δ ≫ 1`). Whoever produces a table reports these three checks with it. They are not stored in the table.
- The table is seeded by hand from the offline simulations. seqtante-open only reads it. qili-experiments reads the same table, so it is part of the cross-repository contract (ADR-0001).
- The offline simulations use the effective qubit Hamiltonian from qilitranspiler @ [`b7ed3cb`](https://github.com/qilimanjaro-tech/qilitranspiler/blob/b7ed3cbe2a5dc4f73b623516a6cb86f6da2c60f8/src/qilitranspiler/simulation.py) (`_one_qubit_coefficients_at_system`), built on cirqed @ [`e7e015a`](https://github.com/qilimanjaro-tech/cirqed/blob/e7e015ac08a034afe3260d5658f542132618bece/src/cirqed/circuits/fluxoniumx.py). Both are pinned because the helper is private and the qilitranspiler branch moves. They need the device parameters E_C, E_L, E_J,max and the junction asymmetry d. Getting those parameters is out of scope for seqtante-open. They belong to Natàlia Padilla or the experimentalists.

## The sweep

- φx, `z_start` and `z_end` come from the table, not from the calibration tree. The tree sets only the ramp durations (and the usual averaging settings). It replaces the `flux_paths` parameter of the first node (QHC-1808).
- φx is swept as a software loop: one QProgram per table row, each sweeping the ramp durations. φx is stored as a loop of the measurement, so the saved data is one φx × T surface.
- A target whose calibration file has no table cannot be swept, so the step fails for that target and the rest of its pipeline is skipped (ADR-0003). The report says why.

## Normalisation

The measured signal is rotated (`rotate_iq`) and normalised with the |0⟩ and |1⟩ readout levels that the Rabi step at the reset point wrote (ADR-0001):

```
P' = (V − level_0) / (level_1 − level_0)
```

`P'` is compared against the bare `P_LZ`, with no thermal floor.

- Readout levels are thermal mixtures: `level_0 = V0 + p·ΔV`, `level_1 = V0 + (1 − p)·ΔV`, with p the thermal excited population. At the end of the ramp the excited population is `q = p + (1 − 2p)·P_LZ`, so `P' = (q − p)/(1 − 2p) = P_LZ` exactly. The thermal population cancels, and the theory curve needs no temperature.
- If the thermal population during LZ (p_L) differs from the one during Rabi (p_R), then `P' = P_LZ + δ(1 − 2P_LZ)/(1 − 2p_R)`, with δ = p_L − p_R. This is an offset and contrast error. It lowers S. The gap ratio (below) is not affected.
- We have no single-shot readout, no quantum-jump measurement and no temperature, and the LZ ramps themselves heat the fridge. We accept this risk until a single-shot readout calibration gives thresholds.

## The score

For each φx (index k) and duration T_ki, the residual is `r_ki = P'_ki − P_LZ(T_ki)`. The distance D pools every point of the φx × T surface:

```
D² = Σ_k v_k Σ_i w_ki r_ki²  =  Σ_k v_k D_k²
S  = 1 − D / D_ref
```

- `w_ki` are trapezoid weights in log T within each φx, normalised to 1. The distance approximates an integral over the curve, and LZ data is taken on a log-spaced duration grid. D_k is the weighted RMS residual of one φx.
- `v_k` are trapezoid weights over the measured φx values, normalised to 1. They are 1 when there is a single φx. Each φx contributes through `v_k` only, whatever its number of points.
- S is pooled, not an average of one S per φx. There is one D_ref for the whole surface, and the square root is taken after the sum, so a φx that agrees badly weighs more than it would in a plain average.
- S = 1 means the data match the theory curve. S = 0 means they are no closer than the reference D_ref. S can be negative.

The three variants differ in D or in D_ref:

| Variant | D | D_ref (S = 0) |
|---|---|---|
| `rmse` | D as above | mean D of uniform random values in [0, 1] at the same points (20 000 draws, fixed seed) |
| `rmse_excess` | `D_ex = √max(D² − σ̂², 0)`: the noise of the data removed | the same, on the uniform draws |
| `shape` | D as above | D of the flat curve at the weighted mean of each φx's theory curve |

### Reported alongside

- **σ̂**, the noise of the data, and **`fit_r2`**, the fit quality. Both come from fitting the LZ decay `floor + A·exp(−(T − T₀)/τ)` to the measured data at each φx, with T₀ fixed at the shortest duration (so the fit has three free parameters), and pooling over φx: `fit_r2 = 1 − ΣSS_res/ΣSS_tot`, `σ̂ = √(ΣSS_res / Σ(n_k − 3))`. If the decay is not resolved at some φx, σ̂ falls back to the scatter between successive points, as in Theory's script.
- **The gap ratio** at each φx, `√(τ_theory/τ_measured)`, from the same decay fit applied to the theory curve and to the data. It reads the agreement physically (τ ∝ 1/Δ²) and is not affected by offset or contrast errors.
- **The uncertainty of S**, propagated from σ̂ through D with D_ref and σ̂ held fixed, and `σ_S = σ_D/D_ref`. With `W = v_k·w_ki`:
  - `rmse` and `shape`: `σ_D = σ̂·√(Σ W² r²)/D`.
  - `rmse_excess`: `σ_D = σ̂·√(Σ W² r²)/D_ex`. It is undefined when `D_ex = 0`.

## Benchmark value and verdict

- The benchmark value is the `rmse` variant of S, dimensionless and **higher-is-better**. `analyze` returns it as `value` with its uncertainty, and the run scores it against the tree thresholds with `verdict_for` (QHC-1848). Its thresholds are plain numbers (ADR-0002, ADR-0003).
- The report shows all three S, σ̂, `fit_r2`, the gap ratio per φx, and the measured curves against the theory curves.
- The database stores `score_s` (the benchmark S) and `rmse` (its distance D), as decided by Natàlia Padilla. The fitted-parameter names of QHC-1497 may need an update for them.
- The benchmark writes nothing to the calibration file (ADR-0002).

## Considered options

- **Reduced χ² with a binomial σ per point** (ADR-0002's candidate for O3): rejected. Theory chose S. The binomial σ √(p(1 − p)/N) assumes single-shot counts, and an averaged IQ readout does not give them.
- **A simulated curve in a data file referenced from the tree** (ADR-0002's candidate for O2): rejected. The closed form needs two numbers per φx. A file of curves would have to match the measured φx and durations exactly, or be interpolated.
- **The full unitary simulation in seqtante-open**: rejected. It needs QiliSDK and cirqed, which seqtante-open does not depend on.
- **One S per φx, averaged**: rejected. It needs one D_ref per φx, and a φx that agrees badly weighs less than in the pooled S (see The score).
- **The table in the calibration tree**: rejected. It is per qubit and not edited by operators.
- **A thermal floor in the theory curve** (as in Theory's notebook): rejected. It applies to the raw excited population. On readout-level-normalised data the thermal population cancels (see Normalisation).

## Consequences

- This settles ADR-0002's two Landau-Zener open questions (O2, O3). It supersedes ADR-0003's χ²ᵣ example and its "χ²ᵣ is better lower" direction for Landau-Zener.
- `NODE_REGISTRY["landau_zener"]` flips to `higher_is_better`, unit `None`.
- qili-qprograms needs the `landau_zener` section model with the table. seqtante-open bumps `uv.lock` for it. qili-experiments codes against the same table.
- The QHC-1808 node loses its `flux_paths` parameter and reads the table instead.
- The node's `analyze` returns the `rmse` S as `value`, so the verdict comes from `verdict_for`. Thresholds go into the EuroQCS tree once the scale is agreed.
- The verdict is no longer "Not evaluated" by design, once a target has a table and thresholds.

## Open questions

- **Confirmation of `rmse` as the benchmark variant.** Pending Andrés's confirmation. Owner: Natàlia Padilla, by 30 November 2026.
- **Flux convention of the table.** cirqed's φz has the linear x → z crosstalk compensated, and puts the degeneracy point at φz = 0.5. Whoever seeds the table must write the values in the calibration file's `flux.z`/`flux.x` convention. How the two map is not yet written down. Owner: Natàlia Padilla with the experimentalists, by 30 November 2026.
