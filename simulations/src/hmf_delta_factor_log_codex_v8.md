# Delta-Factor and Coherence Diagnostic (Codex v8)

- Inferred constant diagonal factor from theta=pi/2: `k_const = 1.003970`
- Sign-flip and extra-g^2 variants included.

| variant | rmse_p00_vs_ordered | rmse_coh_vs_ordered | ratio_over_ordered_median | ratio_over_ordered_max |
|---|---:|---:|---:|---:|
| plus_const_k | 0.020440 | 0.001939 | 0.841304 | 1.000000 |
| plus_baseline | 0.021165 | 0.001723 | 0.835103 | 1.000000 |
| minus_signflip | 0.134978 | 0.208212 | 0.012378 | 1.000000 |
| plus_extra_g2 | 0.257358 | 0.121329 | 1.000000 | 13.924932 |