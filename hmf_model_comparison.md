# HMF Model Comparison: v5 Compact vs Ordered Gaussian

This scratchpad compares the two theoretical predictions for the Mean-Force Hamiltonian $H_{MF}$ in the single-qubit limit.

## 1. Model Definitions

### A. The Ordered Gaussian Model (Reference)
This model computes the specific Gaussian-averaged time-ordered exponential:
$$ \bar{\rho} = e^{-\beta H_Q} \left\langle \mathcal{T} \exp\left( -\int_0^\beta d\tau \, \xi(\tau) \tilde{f}(\tau) \right) \right\rangle_\xi $$
- **Implementation**: `finite_hmf_ordered_gaussian_state` in `prl127_qubit_analytic_bridge.py`
- **Method**: Numerical quadrature on a time grid.
- **Status**: Exact for the spin-boson model with Gaussian bath (up to discretization error).
- **H_MF Extraction**: $H_{MF} = -\frac{1}{\beta} \log(\rho_{ordered})$.

### B. The v5 Compact Model (Analytic Hypothesis)
This model postulates a closed-form solution based on the su(2) Lie algebra closure of the second cumulant.
- **Ansatz**: $ \bar{\rho} = e^{-\beta H_Q} \exp( \Delta ) $
- **Exponent**: $\Delta = \Delta_0 I + \Delta_z \sigma_z + \Sigma_+ \sigma_+ + \Sigma_- \sigma_-$
- **Channels**:
  - $\Delta_z$: derived from Resonant Kernel $R(\omega_q)$
  - $\Sigma_\pm$: derived from Green's Kernel $G^>(0, \omega_q)$
- **H_MF Extraction**: Algebraic diagonalization of the 2x2 matrix.

## 2. Analytic Comparison of Components

We decompose $H_{MF}$ into Pauli components:
$$ H_{MF} = h_x \sigma_x + h_z \sigma_z $$
(assuming $h_y=0$ by gauge choice/symmetry).

| Component | v5 Compact Formula (Current) | Ordered Model |
|---|---|---|
| **Structure** | Closed form, depends on 3 scalar kernels | Numerical functional integral |
| **Input Data** | $K(0), K(\pm\omega_q), R(\pm\omega_q)$ | Full $K(\tau)$ profile |
| **$\theta=\pi/2$** | Predicts $\Delta_x=0 \implies h_x=0$ (pure $h_z$) | Reference shows mixing to $I/2$ ($h_x, h_z \to 0$) |

## 3. Comparison Plan

We will run a temperature scan at fixed coupling $g$ and angle $\theta$ to compare the effective fields $h_x, h_z$.

### Scan Parameters
- **System**: $\omega_q = 2.0$
- **Coupling**: $g = 2.5$ (strong coupling)
- **Angles**:
  1. $\theta = 0.25$ (Generic case)
  2. $\theta = \pi/2$ (Pathological case)
- **Temperature**: $\beta \in [0.1, 5.0]$

### Outputs
- Plot 1: $h_z(\beta)$ comparison
- Plot 2: $h_x(\beta)$ comparison
- Plot 3: Trace distance $D(\rho_{v5}, \rho_{ordered})$ vs $\beta$

## 4. Results

[To be populated by simulation script]
