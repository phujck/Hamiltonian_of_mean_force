# PRL127 Analytic Bridge Diagnostics

## Run Configuration
- `beta`: 1.0
- `omega_q`: 3.0
- `theta`: 0.25
- `lambda_range`: [0.0, 8.0] with 49 points
- `n_modes`: 2, `n_cut`: 4
- `v4_lambda_map`: lam2
- `v4_kernel_mode`: formula
- `v4_kernel_scale`: 1.000000
- `v4_delta_sign_x`: 1.000
- `v4_delta_sign_z`: 1.000
- `q_reorg`: 45.514959

## Kernel Consistency Checks
- `int_0^beta K_0(u) du` (direct): `91.029920`
- `2 q_reorg` (expected): `91.029918`
- Relative mismatch for `int K`: `1.651e-08`
- `K_eff` base from closed formula: `275.142844`
- `K_eff` base from direct Eq.63 double integral: `275.142992`
- Relative mismatch for `K_eff`: `5.373e-07`

## Key Metrics
- Sym model max trace distance: `0.130970`
- Sym model endpoint trace distance: `0.069431`
- Eq67 model max trace distance: `0.288177`
- Ordered model max trace distance: `0.019701`
- Sym non-Hermiticity max: `1.665e-16`
- Eq67 non-Hermiticity max: `7.165e-01`
- Sym field MAE (hx, hy, hz): `(4.9539, 0.0000, 13.3930)`
- Ordered field MAE (hx, hy, hz): `(0.0137, 0.0000, 0.0033)`
- Max Bloch radius (exact, sym, ordered): `(0.905148, 1.000000, 0.905148)`
- Note: large field discrepancies can be amplified when |r| approaches 1, because h ~ arctanh(|r|)/|r|.
- First nonzero coupling sample: `g=0.166667`, `lambda_paper=0.027778`, `K_eff=7.642857`

## Reference Plots
![Alignment](../figures/prl127_signcheck_l8_base_alignment.png)

![Diagnostics](../figures/prl127_signcheck_l8_base_diagnostics.png)

![Field Components](../figures/prl127_signcheck_l8_base_fields.png)

## Data Files
- Scan CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_signcheck_l8_base_scan.csv`
- Summary CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_signcheck_l8_base_summary.csv`
