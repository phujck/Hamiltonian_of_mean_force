# TeX Exact-Form Consistency (Codex v14)

Checks whether Eq. (rho_matrix_v5) and direct `exp(-beta H_Q) exp(Delta)` agree numerically.

| sweep | max_tex_vs_direct_entry_diff | mean_tex_vs_direct_entry_diff | rmse_tex_minus_ordered_p00 | rmse_direct_minus_ordered_p00 | rmse_tex_minus_ordered_coh | rmse_direct_minus_ordered_coh |
|---|---:|---:|---:|---:|---:|---:|
| beta | 9.035e-12 | 1.294e-12 | 0.609205 | 0.609205 | 0.000000 | 0.000000 |
| g | 2.220e-16 | 1.269e-16 | 0.029947 | 0.029947 | 0.106235 | 0.106235 |