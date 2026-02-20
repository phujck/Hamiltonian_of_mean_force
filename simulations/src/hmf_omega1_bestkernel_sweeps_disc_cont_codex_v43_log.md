# Omega_q=1 Best-Kernel Sweeps: Discrete vs Continuous Analytic (Codex v43)

Setup fixed for fairness:
- omega_q=1, n_modes=2, n_cut=6, omega window=[0.2,1.8]
- ED uses the same 2-mode window.
- Analytic discrete: n_modes=2 on this same window.
- Analytic continuous: n_modes=40 on this same window.

| sweep | rmse_disc_p00 | rmse_cont_p00 | rmse_disc_coh | rmse_cont_coh | max_abs_disc_dp00 | max_abs_cont_dp00 |
|---|---:|---:|---:|---:|---:|---:|
| angle | 0.010259 | 0.037981 | 2.586059e-02 | 1.963643e-02 | 0.016678 | 0.068908 |
| coupling | 0.022318 | 0.014469 | 3.923910e-02 | 2.542103e-02 | 0.030275 | 0.022147 |
| temperature | 0.010855 | 0.051115 | 7.436870e-16 | 7.591345e-16 | 0.018267 | 0.094680 |