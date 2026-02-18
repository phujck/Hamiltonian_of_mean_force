# PRL127 Analytic Bridge Diagnostics

## Run Configuration
- `beta`: 1.0
- `omega_q`: 3.0
- `theta`: 0.25
- `lambda_range`: [0.0, 8.0] with 49 points
- `n_modes`: 2, `n_cut`: 4
- `v4_lambda_map`: lam2
- `v4_kernel_mode`: formula

## Key Metrics
- Sym model max trace distance: `0.130970`
- Sym model endpoint trace distance: `0.069431`
- Eq67 model max trace distance: `0.288177`
- Ordered model max trace distance: `0.019125`
- Sym non-Hermiticity max: `1.665e-16`
- Eq67 non-Hermiticity max: `7.165e-01`
- Sym field MAE (hx, hy, hz): `(4.9539, 0.0000, 13.3930)`
- Ordered field MAE (hx, hy, hz): `(0.0127, 0.0000, 0.0030)`
- Max Bloch radius (exact, sym, ordered): `(0.905148, 1.000000, 0.905148)`
- Note: large field discrepancies can be amplified when |r| approaches 1, because h ~ arctanh(|r|)/|r|.

## Reference Plots
![Alignment](../figures/prl127_bridge_lam2_alignment.png)

![Diagnostics](../figures/prl127_bridge_lam2_diagnostics.png)

![Field Components](../figures/prl127_bridge_lam2_fields.png)

## Data Files
- Scan CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_bridge_lam2_scan.csv`
- Summary CSV: `C:\Users\gerar\VScodeProjects\Hamiltonian_of_mean_force\simulations\results\data\prl127_bridge_lam2_summary.csv`
