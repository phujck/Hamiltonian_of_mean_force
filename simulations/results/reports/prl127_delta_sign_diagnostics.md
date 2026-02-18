# PRL127 Delta Sign Diagnostics

## Purpose
Test whether the finite-model mismatch is primarily caused by a simple sign error in `\Delta` (with optional overall kernel scaling).

## Method
- Benchmark: `prl127_qubit_analytic_bridge`
- Safe execution: `run_safe.ps1` (thread-limited)
- New diagnostic controls in code:
  - `--v4-kernel-scale`
  - `--v4-delta-sign-x`
  - `--v4-delta-sign-z`
- Compared against exact reduced-state ED across the same coupling scan.

## Small/Intermediate Coupling Scan (`lambda_max=0.4`)

| Run | scale | sx | sz | sym max dist | eq67 max dist |
|---|---:|---:|---:|---:|---:|
| `prl127_signcheck_base` | 1.0 | +1 | +1 | 0.130964 | 0.288376 |
| `prl127_signcheck_scale05` | 0.5 | +1 | +1 | **0.114526** | **0.270880** |
| `prl127_signcheck_scale05_zflip` | 0.5 | +1 | -1 | 0.204393 | 0.376527 |
| `prl127_signcheck_scale05_xflip` | 0.5 | -1 | +1 | 0.245010 | 0.401113 |
| `prl127_signcheck_scale05_xzflip` | 0.5 | -1 | -1 | 0.340840 | 0.501596 |

Result: simple sign flips worsen agreement. A global scale reduction helps somewhat.

## Full Scan (`lambda_max=8`)

| Run | scale | sym max dist | sym end dist | eq67 max dist | eq67 end dist |
|---|---:|---:|---:|---:|---:|
| `prl127_signcheck_l8_base` | 1.0 | 0.130970 | 0.069431 | 0.288177 | 0.219053 |
| `prl127_signcheck_l8_scale05` | 0.5 | **0.104495** | 0.069431 | **0.261176** | 0.219053 |

Result: same pattern at large coupling; scaling improves peak mismatch, but endpoint mismatch remains unchanged.

## Interpretation
1. The dominant issue is not a single `\Delta_x/\Delta_z` sign mistake in the compressed model.
2. There is likely an amplitude/mapping mismatch in the compressed kernel channel (`K_eff` normalization/effective prefactor), since scaling helps.
3. Endpoint invariance suggests the ultrastrong asymptote structure is intact, but finite-coupling interpolation is miscalibrated.
4. Remaining gap is consistent with missing ordered-time structure in the symmetric compression (ordered model still best).

## Artifacts
- Data:
  - `simulations/results/data/prl127_signcheck_base_summary.csv`
  - `simulations/results/data/prl127_signcheck_scale05_summary.csv`
  - `simulations/results/data/prl127_signcheck_scale05_zflip_summary.csv`
  - `simulations/results/data/prl127_signcheck_scale05_xflip_summary.csv`
  - `simulations/results/data/prl127_signcheck_scale05_xzflip_summary.csv`
  - `simulations/results/data/prl127_signcheck_l8_base_summary.csv`
  - `simulations/results/data/prl127_signcheck_l8_scale05_summary.csv`
- Figures:
  - `simulations/results/figures/prl127_signcheck_base_alignment.png`
  - `simulations/results/figures/prl127_signcheck_scale05_alignment.png`
  - `simulations/results/figures/prl127_signcheck_l8_base_alignment.png`
  - `simulations/results/figures/prl127_signcheck_l8_scale05_alignment.png`
