# Ordered-Exact Density Agreement (Codex v3)

## Core Question
How closely do ordered and exact models agree, and does direct Delta matrix multiplication help?

## Matrix-Exponent Diagnostics
- Max |Tr(A B)/(Tr(A)Tr(B)) - 1| across scan: `9.999939e-01` (product-of-traces assumption is generally not valid for 2x2 AB).
- Max |expm(Delta) - V exp(D) V^-1| elementwise: `2.575717e-14`.
- RMSE trace-distance between compact AB and closed compact constructor: `4.676933e-16`.

## Summary Metrics

| sweep | model | rmse_p00 | rmse_p11 | rmse_coh | rmse_trace_distance | ratio_over_exact_median | ratio_over_exact_min | ratio_over_exact_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| angle | compact_ab_vs_exact | 0.361436 | 0.361436 | 0.231793 | 0.429376 | 2.310615 | 1.000000 | 1030116086899.778320 |
| angle | ordered_vs_exact | 0.102769 | 0.102769 | 0.081358 | 0.131075 | 1.805235 | 1.000000 | 2.734463 |
| coupling | compact_ab_vs_exact | 0.050269 | 0.050269 | 0.146850 | 0.155216 | 1.181088 | 0.997620 | 4.253243 |
| coupling | ordered_vs_exact | 0.029161 | 0.029161 | 0.054092 | 0.061452 | 1.158803 | 1.000000 | 2.147870 |
| temperature | compact_ab_vs_exact | 0.754479 | 0.754479 | 0.000000 | 0.754479 | 4143920266984271.500000 | 1.135176 | 4686021538092944.000000 |
| temperature | ordered_vs_exact | 0.153691 | 0.153691 | 0.000000 | 0.153691 | 2.252771 | 1.108752 | 2.744270 |