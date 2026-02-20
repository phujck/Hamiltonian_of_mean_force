# Diagonal Rapidity Mismatch (Codex v18)

Population channel reparameterized via u and d=atanh(u).

| sweep | rmse_u | rmse_d | scale_median | scale_min | scale_max |
|---|---:|---:|---:|---:|---:|
| beta_theta_pi2_g05 | 0.099384 | 11.826056 | 0.230960 | 0.116571 | 0.970477 |
| g_beta2_theta_pi2 | 0.087344 | 13.481453 | 0.102771 | 0.096387 | 0.954291 |
| g_beta2_theta_pi4 | 0.062598 | 0.140492 | 0.950201 | 0.673858 | 1.051863 |
| fit_g_beta2_theta_pi2 | nan | nan | 1.040503 | 13.450000 | 0.944216 |
| fit_g_beta2_theta_pi4 | nan | nan | 0.909896 | 0.000000 | 0.000000 |

g_beta2_theta_pi2 suppression fit: scale(g) ~ a/(1+b g^2), a=1.040503, b=13.450000, R2=0.944216.
g_beta2_theta_pi4 suppression fit: scale(g) ~ a/(1+b g^2), a=0.909896, b=0.000000, R2=0.000000.