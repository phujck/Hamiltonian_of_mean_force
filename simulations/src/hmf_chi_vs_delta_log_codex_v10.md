# Chi vs Delta Mismatch Diagnostic (Codex v10)

- fitted k_chi (coupling sweep objective): `0.945000`
- fitted k_delta (coupling sweep objective): `1.095000`

| sweep | variant | k_value | rmse_p00 | rmse_coh | rmse_ratio_rel | ratio_rel_median |
|---|---:|---:|---:|---:|---:|---:|
| coupling | chi_only | 0.945000 | 0.001461 | 0.052012 | 0.015178 | 0.007621 |
| coupling | delta_only | 1.095000 | 0.005280 | 0.021421 | 0.054179 | -0.002108 |
| coupling | baseline | 1.000000 | 0.020853 | 0.003040 | 0.150615 | -0.161505 |
| temperature | baseline | 1.000000 | 0.079879 | 0.000000 | 0.510643 | 0.505805 |
| temperature | chi_only | 0.945000 | 0.198926 | 0.000000 | 0.955291 | -1.000000 |
| temperature | delta_only | 1.095000 | 0.226042 | 0.000000 | 2.116073 | 1.533827 |