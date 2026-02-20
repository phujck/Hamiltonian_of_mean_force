# Density-Focused Debug Log (Codex v2)

## Scope
Only density components were used for calibration and validation: rho_00, rho_11, |rho_01|.

## Fitted Running Parameters
- scale = 1.200000
- kappa = 0.940000
- model: chi_eff = chi_raw / (1 + chi_raw / (kappa * beta*omega_q/2))

## RMSE Summary

| sweep | model | rmse_p00 | rmse_p11 | rmse_coh | max_abs_p00 | max_abs_p11 | max_abs_coh |
|---|---:|---:|---:|---:|---:|---:|---:|
| angle | legacy | 0.021467 | 0.021467 | 0.010439 | 0.048478 | 0.048478 | 0.018433 |
| angle | running | 0.021801 | 0.021801 | 0.042067 | 0.041905 | 0.041905 | 0.079049 |
| coupling | legacy | 0.165135 | 0.165135 | 0.240675 | 0.201182 | 0.201182 | 0.292179 |
| coupling | running | 0.025589 | 0.025589 | 0.046827 | 0.030918 | 0.030918 | 0.057214 |
| temperature | legacy | 0.483790 | 0.483790 | 0.000000 | 0.624638 | 0.624638 | 0.000000 |
| temperature | running | 0.025142 | 0.025142 | 0.000000 | 0.042560 | 0.042560 | 0.000000 |