Updated Seqtante-Open to qililab 0.35.2, now required as `>=0.35.2,<0.36`, and fixed what the new version broke at runtime.

- Fits read the stored runcard from `measurement.platform`. qililab exposes the `platform_before` column under that name now, so `FluxoniumSingleToneModel`, `FluxoniumSingleToneFluxModel`, `FluxoniumTwoToneModel` and `FluxoniumTwoToneFluxModel` raised `AttributeError` when `lo` was left unset.
- The `offset_calibration`, `single_tone`, `two_tone` and `two_tone_vs_flux` nodes no longer pass `autocalibration=True`. `StreamArray` now warns on every call that sets the deprecated flag, and raises a `ValueError` when it doesn't match the manager's schema.
- A missing `measurement_id` now raises qililab's `IndexError` listing the missing ids, instead of the `LookupError` from `FittingClass`.
