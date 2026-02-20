# v5 Ordered-Model Writeup

This note documents the active v5 simulation models and their kernel definitions.
It is specific to:

- `simulations/src/hmf_v5_regime_generate.py`
- `simulations/src/hmf_v5_qubit_core.py`
- `simulations/src/prl127_qubit_analytic_bridge.py`
- `simulations/src/prl127_qubit_benchmark.py`

`symmetric`-kernel comparators are intentionally excluded from the v5 regime pipeline.

## 1) Shared bath discretization

Both ED and theory paths use the same discretized spectral density:

- `J(omega) = Q * tau_c * omega * exp(-tau_c * omega)`
- mode grid `omega_k` over `[omega_min, omega_max]`
- `g_k^2 = J(omega_k) * Delta_omega`

Code:

- `prl127_qubit_benchmark.py:spectral_density_exp`
- `prl127_qubit_benchmark.py:build_static_operators`
- `hmf_v5_qubit_core.py:_discrete_bath`

## 2) ED reference ("exact" finite model)

The ED reference state is built directly from the finite Hamiltonian and then traced:

- `H_tot(g) = H_S + H_B + g * H_I`
- `rho_tot = exp(-beta H_tot) / Tr exp(-beta H_tot)`
- `rho_Q^ED = Tr_B rho_tot`

Code:

- `hmf_v5_qubit_core.py:exact_reduced_state`

Note on counterterm:

- In the active v5 path, no explicit `g^2` counterterm is used.
- For this qubit coupling `X = cos(theta) sigma_z - sin(theta) sigma_x`, `X^2 = I`, so a counterterm would only add a scalar shift and not change `rho_Q`.

## 3) Kernel used by theory paths

The base imaginary-time kernel (`g=1`) is

`K0(u) = sum_k g_k^2 * cosh[omega_k (beta/2 - u)] / sinh(beta omega_k/2)`, with `u in [0,beta]`.

Code:

- `hmf_v5_qubit_core.py:kernel_profile_base`
- same formula in `prl127_qubit_analytic_bridge.py:_kernel_profile`

Coupling is applied as:

- `K(u; g) = g^2 K0(u)`

## 4) Ordered model (active nonperturbative comparator)

The ordered model evaluates the Gaussian-noise average of the time-ordered product directly, without reducing to closed channel formulas.

### 4.1 Time covariance from kernel

On a time grid `{tau_i}`:

- `u_ij = min(|tau_i - tau_j|, beta - |tau_i - tau_j|)`
- `C_ij = K0(u_ij)`

Code:

- `prl127_qubit_analytic_bridge.py:_build_ordered_quadrature_context`

### 4.2 KL + Gauss-Hermite quadrature

The covariance matrix is eigendecomposed:

- `C ~= V diag(lambda) V^T`

A finite KL rank is retained; Gaussian variables are integrated by tensor-product Gauss-Hermite nodes.

Code:

- `prl127_qubit_analytic_bridge.py:_build_ordered_quadrature_context`

### 4.3 Ordered propagator average

For each node, a noise history `xi(tau_n)` is constructed and inserted into:

- `W[xi] = T_prod_n exp[-Delta_tau * xi_n * f_tilde(tau_n)]`

with the qubit step closed analytically as:

- `exp[-v f_tilde] = cosh(v) I - sinh(v) f_tilde`

Then:

- `rho_bar = exp(-beta H_Q) * <W[xi]>`
- project to Hermitian PSD, normalize.

Code:

- `prl127_qubit_analytic_bridge.py:finite_hmf_ordered_gaussian_state`

## 5) Compact channel model (from ordered-kernel moments)

The compact model computes:

- `K(omega) = int_0^beta du K0(u) exp(omega u)`
- `R(omega) = int_0^beta du (beta-u) K0(u) exp(omega u)`

then channels:

- `Sigma_+^0, Sigma_-^0, Delta_z^0` and `chi0 = sqrt(Delta_z0^2 + Sigma_+0 Sigma_-0)`
- scale by `g^2`
- build `rho_Q^th` from the closed v5 matrix form.

Code:

- `hmf_v5_qubit_core.py:laplace_k0`
- `hmf_v5_qubit_core.py:resonant_r0`
- `hmf_v5_qubit_core.py:compute_v5_base_channels`
- `hmf_v5_qubit_core.py:v5_theory_state`

## 6) "Kernel in exact case" vs "kernel in ordered case"

ED does not explicitly integrate `K(u)` at runtime. It uses `(omega_k, g_k)` in `H_B` and `H_I`, then diagonalizes `H_tot`.

The ordered and compact theory paths explicitly build `K0(u)` from the same `(omega_k, g_k)` and then integrate/average over it.

So the difference is computational route, not bath input:

- ED route: finite Hamiltonian -> thermal state -> partial trace.
- Ordered route: same bath -> covariance kernel -> ordered Gaussian average.
- Compact route: same bath -> Laplace/resonant kernel moments -> closed qubit matrix.

## 7) Where the ordered prediction is in code

In the v5 regime pipeline, the ordered prediction is exactly this object:

- `rho_th = finite_hmf_ordered_gaussian_state(g, ordered_ctx)`

Code path:

1. `hmf_v5_regime_generate.py:_run_single_beta`
2. `ordered_ctx = _build_ordered_quadrature_context(...)`
3. `_theory_state_from_g(g)` selects:
   - `compact`: `v5_theory_state(...)`
   - `ordered`: `finite_hmf_ordered_gaussian_state(...)`
4. `rho_ed` is built independently via `exact_reduced_state(...)`
5. comparison metric is `trace_distance(rho_ed, rho_th)`

So when `--theory-model ordered` is used, the theory curve is the ordered Gaussian state produced by
`finite_hmf_ordered_gaussian_state`, not ED.

## 8) ED vs ordered: complete input/assumption difference ledger

This section lists every place the two pipelines are not the same object.

### 8.1 State-construction principle

- ED ("exact" finite model):
  - builds full finite Hamiltonian matrix, diagonalizes, thermalizes, traces bath.
  - formula: `rho_Q^ED = Tr_B[exp(-beta H_tot)] / Z`.
- Ordered:
  - builds a stochastic imaginary-time propagator average from kernel covariance.
  - formula: `rho_Q^ord ~ exp(-beta H_Q) < T exp[-int xi(t) f_tilde(t) dt] >_xi`.

### 8.2 Object being integrated

- ED integrates over bath Hilbert space exactly (for given finite `n_modes`, `n_cut`).
- Ordered integrates over Gaussian-noise histories (KL+Gauss-Hermite quadrature).

### 8.3 Numerical truncations are different

- ED truncations:
  - `n_modes` (bath discretization)
  - `n_cut` (Fock truncation per mode)
- Ordered truncations:
  - `ordered_time_slices` (time discretization of ordered product)
  - `ordered_kl_rank` (KL rank truncation of covariance)
  - `ordered_gh_order` (Gaussian quadrature order)
  - `ordered_max_nodes` (node cap)

These are independent approximation controls; matching one set does not imply matching the other.

### 8.4 Kernel usage

- ED does not explicitly integrate `K(u)` during state construction.
  - it uses `(omega_k, g_k)` in Hamiltonian operators.
- Ordered explicitly constructs covariance from `K0(u)` and uses it as noise covariance.

Both use the same underlying discretized bath data, but through different computational pipelines.

### 8.5 Time ordering treatment

- ED: exact in operator time ordering (inside full matrix exponential).
- Ordered:
  - time ordering is approximated by a discrete product over `n_time_slices`.
  - each step assumes piecewise-constant noise on that slice.

### 8.6 Positivity enforcement

- ED output is PSD by construction from `exp(-beta H_tot)`.
- Ordered output is post-processed by projection to Hermitian PSD due finite quadrature/grid errors.
  - this is a numerical stabilization step, not part of the analytic definition.

### 8.7 Counterterm convention (current v5 path)

- Active v5 ED path uses `H_tot = H_S + H_B + g H_I` (no explicit counterterm).
- Ordered path also uses kernel construction without an explicit `g^2` counterterm term in `H_tot`.
- For this qubit coupling (`X^2 = I`), adding/removing the counterterm only shifts total energy by a scalar and does not change `rho_Q`.

### 8.8 Practical consequence

Even with the same bath discretization and same `(beta, g, theta)`, ED and ordered can differ because:

1. they solve different numerical problems,
2. they have different truncation controls,
3. one is a full finite-Hilbert thermal trace, the other is a Gaussian-history quadrature.

This is exactly why the ordered model is a prediction/comparator, not an identity to ED at finite numerical resolution.

## 9) Minimal reproducibility checklist

To make ED-vs-ordered comparisons meaningful at a point `(beta, g, theta)`, report:

- ED: `n_modes`, `n_cut`
- Ordered: `ordered_time_slices`, `ordered_kl_rank`, `ordered_gh_order`, `ordered_max_nodes`
- shared bath params: `omega_min`, `omega_max`, `q_strength`, `tau_c`
- observed metrics: `D(rho_ED, rho_ordered)`, plus `mx,my,mz` and coherence for each

Without this full tuple, disagreement cannot be interpreted cleanly.
