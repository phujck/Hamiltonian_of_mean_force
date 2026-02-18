# PRL127 Analytic Bridge Diagnostics

## Run Configuration
- `beta`: 1.0
- `omega_q`: 3.0
- `theta`: 0.25
- `lambda_range`: [0.0, 4.0] with 401 points
- `n_modes`: 2, `n_cut`: 4
- `v4_lambda_map`: lam2
- `v4_kernel_mode`: formula
- `q_reorg`: 45.514959

## Kernel Consistency Checks
- `int_0^beta K_0(u) du` (direct): `91.029920`
- `2 q_reorg` (expected): `91.029918`
- Relative mismatch for `int K`: `1.651e-08`
- `K_eff` base from closed formula: `275.142844`
- `K_eff` base from direct Eq.63 double integral: `275.142992`
- Relative mismatch for `K_eff`: `5.373e-07`

## Key Metrics
- Sym model max trace distance: `0.130964`
- Sym model endpoint trace distance: `0.070853`
- Eq67 model max trace distance: `0.288376`
- Ordered model max trace distance: `0.021051`
- Sym non-Hermiticity max: `2.776e-16`
- Eq67 non-Hermiticity max: `7.165e-01`
- Sym field MAE (hx, hy, hz): `(4.8069, 0.0000, 12.9307)`
- Ordered field MAE (hx, hy, hz): `(0.0185, 0.0000, 0.0043)`
- Max Bloch radius (exact, sym, ordered): `(0.905148, 1.000000, 0.905148)`
- Note: large field discrepancies can be amplified when |r| approaches 1, because h ~ arctanh(|r|)/|r|.
- First nonzero coupling sample: `g=0.010000`, `lambda_paper=0.000100`, `K_eff=0.027514`

## Reference Plots
![Alignment](../figures/prl127_scaling_probe4_alignment.png)

![Diagnostics](../figures/prl127_scaling_probe4_diagnostics.png)

![Field Components](../figures/prl127_scaling_probe4_fields.png)

## Data Files
- Scan CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_scaling_probe4_scan.csv`
- Summary CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_scaling_probe4_summary.csv`
