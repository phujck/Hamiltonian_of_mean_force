# Best Model (Beta-Flow Blended) Codex v24

Fixed branch params: c_base=2.645810, b_base=0.680000, mu_base=0.700000
Blend weight: w(theta,g) = sin(theta)^8 * min(1, (g/0.5)^2)

| sweep | comparison | rmse_p00 | rmse_coh |
|---|---|---:|---:|
| angle | new_vs_ed | 0.126256 | 0.153078 |
| angle | ordered_vs_ed | 0.039059 | 0.031953 |
| angle | raw_vs_ed | 0.290266 | 0.158217 |
| coupling | new_vs_ed | 0.034517 | 0.120314 |
| coupling | ordered_vs_ed | 0.009860 | 0.024483 |
| coupling | raw_vs_ed | 0.023173 | 0.093823 |
| temperature | new_vs_ed | 0.074110 | 0.000000 |
| temperature | ordered_vs_ed | 0.095903 | 0.000000 |
| temperature | raw_vs_ed | 0.583750 | 0.000000 |