# Three-Mode Resonant-Node Test (Codex v41)

Setup: omega_q=1, theta=pi/2, g=0.5, n_modes=3, n_cut=3.
Analytic branch uses the exact same custom discrete kernel nodes as ED.

Best node set by fair metric: `uniform_[0.2,1.0,1.8]` with rmse_ed_vs_disc_p00=0.114395

| case | omega_nodes | rmse_ed_vs_disc_p00 | rmse_ed_vs_disc_coh | p00_at_beta2_ed | p00_at_beta2_an | p00_at_beta8_ed | p00_at_beta8_an |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform_[0.2,1.0,1.8] | 0.200000;1.000000;1.800000 | 0.114395 | 9.413655e-17 | 0.242906 | 0.262271 | 0.186870 | 0.354509 |
| resonant_[0.05,1.0,2.0] | 0.050000;1.000000;2.000000 | 0.132529 | 3.744925e-17 | 0.244957 | 0.280802 | 0.173574 | 0.359335 |
| resonant_[0.05,1.0,1.8] | 0.050000;1.000000;1.800000 | 0.141308 | 3.770893e-17 | 0.235894 | 0.275112 | 0.160106 | 0.357864 |