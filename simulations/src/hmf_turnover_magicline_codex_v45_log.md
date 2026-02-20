# Turnover Magic-Line Formulation (Codex v45)

## Formulated condition
For the bandwidth scan with lower-bound clamping `omega_min = max(omega_floor, omega_q - delta)`,
the turnover onset is where the resonance pinning breaks and detuning becomes nonzero:

$B^* = 2(\omega_q - \omega_{\mathrm{floor}})$

with `B = omega_max - omega_min`.

Equivalent piecewise detuning predictor (n_modes=3, uniform spacing):

$\dfrac{\delta_\mathrm{detune}(B)}{\omega_q} = \dfrac{\max\!\left(0,\ \omega_{\mathrm{floor}} + \dfrac{B}{2} - \omega_q\right)}{\omega_q}$

## Numbers from current CSVs
- omega_q: 2.000000
- omega_floor: 0.100000
- magic_bandwidth_B_star: 3.800000
- empirical_B_min_ncut6: 3.200000
- empirical_rmse_min_ncut6: 0.058418
- empirical_B_slope_zero_ncut6: 3.376999
- gap_B_star_minus_B_min: 0.600000
- gap_B_star_minus_B_slope0: 0.423001

Generated figure:
- hmf_turnover_magicline_codex_v45.png