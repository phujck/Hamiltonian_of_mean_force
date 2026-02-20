# Functional Difference Diagnostic (Codex v9)

Config: beta=2.0, theta=pi/4, omega_q=2, g in [0.0, 2.0] (21 points)

| target | model | a | b | R2 | RMSE |
|---|---:|---:|---:|---:|---:|
| delta_p00 | a*g^2/(1+b*g^2) | -0.053107 | 1.475000 | 0.990868 | 0.001033 |
| delta_p00 | a*g | -0.017933 | nan | 0.927578 | 0.002910 |
| delta_p00 | a*g^2 | -0.010672 | nan | 0.520472 | 0.007487 |
| ratio_rel_err | a*g^2/(1+b*g^2) | -0.612119 | 2.700000 | 0.982436 | 0.008998 |
| ratio_rel_err | a*g | -0.129222 | nan | 0.822870 | 0.028574 |
| ratio_rel_err | a*g^2 | -0.075333 | nan | 0.108847 | 0.064092 |