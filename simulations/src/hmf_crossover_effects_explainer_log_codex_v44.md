# Crossover Effects Explainer (Codex v44)

## Bandwidth crossover evidence
| n_cut | best_bandwidth | best_delta | best_rmse_ed_vs_disc_p00 | rmse_ed_vs_ordered_p00_at_best |
|---:|---:|---:|---:|---:|
| 4 | 3.200 | 1.600 | 0.076900 | 0.029841 |
| 5 | 3.200 | 1.600 | 0.064376 | 0.044776 |
| 6 | 3.200 | 1.600 | 0.058418 | 0.053808 |

## Interpretation metrics
- crossover_bandwidth_ref_ncut6: 3.200000
- ncut6_slope_signflip_bandwidth: 3.550000
- cutoff_rmse46_at_crossover_bw: 0.026334
- cutoff_rmse46_at_max_bw: 0.045128
- disc_vs_cont_population_rmse_gap_coupling: 0.007850
- disc_vs_cont_population_rmse_gap_temperature: -0.040259
- disc_vs_cont_population_rmse_gap_angle: -0.027722

## Regime picks used in beta panel
- narrow: delta=0.800, bandwidth=1.600
- crossover: delta=1.600, bandwidth=3.200
- broad: delta=3.000, bandwidth=4.900

Generated figure:
- hmf_crossover_effects_explainer_codex_v44.png