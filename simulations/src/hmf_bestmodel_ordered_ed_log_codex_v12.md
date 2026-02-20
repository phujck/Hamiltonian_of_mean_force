# Best Model vs Ordered vs ED (Codex v12)

ED is computed with lightweight settings: n_modes=2, n_cut=3.
Ordered uses lightweight kernel settings from standalone implementation.

| sweep | comparison | rmse_p00 | rmse_coh |
|---|---:|---:|---:|
| angle | best_vs_ed | 0.048116 | 0.049772 |
| angle | best_vs_ordered | 0.007750 | 0.008089 |
| angle | ordered_vs_ed | 0.055148 | 0.045049 |
| coupling | best_vs_ed | 0.013317 | 0.040420 |
| coupling | best_vs_ordered | 0.021567 | 0.001835 |
| coupling | ordered_vs_ed | 0.026365 | 0.040175 |
| temperature | best_vs_ed | 0.135716 | 0.000000 |
| temperature | best_vs_ordered | 0.077869 | 0.000000 |
| temperature | ordered_vs_ed | 0.076741 | 0.000000 |