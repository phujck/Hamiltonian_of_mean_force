# HMF Debug Log (Codex v1)

## Objective
Align compact-v5 populations/coherences with ordered-model sweeps using sign/scaling diagnostics.

## Changes Tested
1. Legacy baseline: transverse channel sign flipped and fixed `scale=0.25` (historical standalone behavior).
2. Sign correction: restore manuscript-consistent signs for `Sigma_+` and `Sigma_-`.
3. Running renormalization: apply `r = 1 / (1 + chi_raw / chi_cap)`, with `chi_cap = kappa * (beta*omega_q/2)`.
4. Final parameters: `scale=1.0400`, `kappa=0.9400`.

## Error Sources Identified
- Fixed 1/4 scale under-corrects low-to-mid coupling and over-regularizes only one regime.
- Sign mismatch in transverse channels drives wrong direction in generic-angle sweeps.
- Unbounded raw `chi ~ g^2` causes diagonal over-polarization at strong coupling.

## Sweep RMSE Summary

| sweep | model | rmse_p00 | rmse_coh | max_abs_p00 | max_abs_coh |
|---|---:|---:|---:|---:|---:|
| angle | legacy | 0.021467 | 0.010439 | 0.048478 | 0.018433 |
| angle | running | 0.025432 | 0.036042 | 0.051286 | 0.068871 |
| coupling | legacy | 0.165135 | 0.240675 | 0.201182 | 0.292179 |
| coupling | running | 0.024452 | 0.045133 | 0.030517 | 0.056714 |
| temperature | legacy | 0.483790 | 0.000000 | 0.624638 | 0.000000 |
| temperature | running | 0.031260 | 0.000000 | 0.052041 | 0.000000 |
