"""
Diagnostic: HMF v5 compact model vs ordered model vs ED at theta=pi/2.

Key reference: ultrastrong limit at theta=pi/2 must approach I/2 (paper eq 8, cos(pi/2)=0).
Run from simulations/src/ directory.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from prl127_qubit_benchmark import BenchmarkConfig
from hmf_v5_qubit_core import (
    build_ed_context, exact_reduced_state,
    compute_v5_base_channels, coupling_channels, v5_theory_state,
    bloch_components, kernel_profile_base, laplace_k0, resonant_r0,
)
from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state,
    _bloch_components,
    ultrastrong_closed_qubit_state,
)

IDENTITY_2 = np.eye(2, dtype=complex)

# ── Parameters ──────────────────────────────────────────────────────────────
BETA     = 2.0
OMEGA_Q  = 2.0
THETA    = np.pi / 2           # pure transverse coupling
G_VALUES = [0.0, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0]

CFG = BenchmarkConfig(
    beta=BETA, omega_q=OMEGA_Q, theta=THETA,
    lambda_min=0.0, lambda_max=max(G_VALUES),
    lambda_points=len(G_VALUES),
    n_modes=4, n_cut=3,
    omega_min=0.1, omega_max=10.0,
    q_strength=5.0, tau_c=0.5,
    output_prefix="hmf_pi2_diag",
)
a = 0.5 * BETA * OMEGA_Q  # beta*omega_q/2

# ── Pre-build contexts ────────────────────────────────────────────────────
ed_ctx   = build_ed_context(CFG)
base_ch  = compute_v5_base_channels(CFG, n_kernel_grid=4001)
ordered_ctx = _build_ordered_quadrature_context(
    CFG, n_time_slices=120, kl_rank=4, gh_order=4, max_nodes=500000,
)

# Ultrastrong reference
rho_us = ultrastrong_closed_qubit_state(CFG)   # should be I/2 at theta=pi/2

print("=" * 70)
print(f"theta=pi/2 diagnostic  beta={BETA}  omega_q={OMEGA_Q}  a={a:.4f}")
print(f"Ultrastrong reference (paper eq 8): diag should be [{rho_us[0,0].real:.4f}, {rho_us[1,1].real:.4f}]")
print("=" * 70)

# ── Base-channel values ───────────────────────────────────────────────────
print("\nBase channels (g=1):")
print(f"  Sigma_+0 = {base_ch.sigma_plus0:.6f}   (should be 0 since c=0)")
print(f"  Sigma_-0 = {base_ch.sigma_minus0:.6f}   (should be 0 since c=0)")
print(f"  Delta_z0 = {base_ch.delta_z0:.6f}")
print(f"  chi0     = {base_ch.chi0:.6f}   (= |Delta_z0| at pi/2)")
print(f"  R0_plus  = {base_ch.r0_plus:.6f}")
print(f"  R0_minus = {base_ch.r0_minus:.6f}")
print(f"  K(0)     = {base_ch.k0_zero:.6f}")
print(f"  K(+wq)   = {base_ch.k0_plus:.6f}")
print(f"  K(-wq)   = {base_ch.k0_minus:.6f}")

# Numerical check: does R^- equal the integral from compute_delta_coefficients?
from prl127_qubit_analytic_bridge import compute_delta_coefficients
_, dx_num, dy_num, dz_num = compute_delta_coefficients(CFG, n_kernel_grid=4001)
print(f"\nNumerical Delta channels from compute_delta_coefficients (g=1 equivalent):")
print(f"  delta_x (num) = {dx_num:.6f}   (should be 0 since c=0)")
print(f"  delta_y (num) = {dy_num:.6f}   (should be ~0)")
print(f"  delta_z (num) = {dz_num:.6f}")
print(f"  delta_z (v5)  = {base_ch.delta_z0:.6f}")
print(f"  match:         {abs(dz_num - base_ch.delta_z0) < 1e-4}")

# ── Per-coupling comparison ───────────────────────────────────────────────
print("\n" + "-"*70)
print(f"{'g':>6} | {'ED rho11':>9} {'ED rho22':>9} | {'Ord rho11':>10} {'Ord rho22':>10} | {'V5 rho11':>9} {'V5 rho22':>9} | {'chi':>7} {'tanh_chi':>9}")
print("-"*70)

for g in G_VALUES:
    # ED
    rho_ed = exact_reduced_state(ed_ctx, g)
    ed11 = rho_ed[0,0].real
    ed22 = rho_ed[1,1].real

    # Ordered
    rho_ord = finite_hmf_ordered_gaussian_state(g, ordered_ctx)
    ord11 = rho_ord[0,0].real
    ord22 = rho_ord[1,1].real

    # V5 compact
    ch = coupling_channels(base_ch, g)
    rho_v5 = v5_theory_state(CFG, ch)
    v5_11 = rho_v5[0,0].real
    v5_22 = rho_v5[1,1].real
    chi = ch.chi
    tanh_chi = np.tanh(chi) if chi > 0 else 0.0

    print(f"{g:>6.2f} | {ed11:>9.5f} {ed22:>9.5f} | {ord11:>10.5f} {ord22:>10.5f} | {v5_11:>9.5f} {v5_22:>9.5f} | {chi:>7.4f} {tanh_chi:>9.5f}")

# ── Formula check: where does the v5 state go wrong? ─────────────────────
print("\n" + "="*70)
print("V5 formula analysis at theta=pi/2:")
print(f"  chi = g^2 * Delta_z0,  gamma = tanh(chi)/chi")
print(f"  rho_11 = e^{{-a}} * (1 + tanh(chi)) / Z_Q")
print(f"  rho_22 = e^{{+a}} * (1 - tanh(chi)) / Z_Q")
print(f"  Z_Q = 2*(cosh(a) - tanh(chi)*sinh(a))")
print(f"\n  As chi -> inf: tanh(chi) -> 1")
print(f"  Then Z_Q -> 2*(cosh(a) - sinh(a)) = 2*exp(-a) = {2*np.exp(-a):.5f}")
print(f"  rho_11 -> exp(-a)*2 / (2*exp(-a)) = 1.0   [EXCITED STATE DOMINATES - WRONG]")
print(f"  rho_22 -> 0.0")
print(f"\n  CORRECT ultrastrong limit: rho_11 = rho_22 = 0.5 (I/2)")
print(f"\n  Sign check: if sign of tanh(chi) in diagonal were FLIPPED:")
print(f"  rho_11 = e^{{-a}}*(1 - tanh(chi)) / Z_Q'  with Z_Q'=2*(cosh(a)+tanh(chi)*sinh(a))")
print(f"  As chi->inf: rho_11 -> 0,  rho_22 -> 1   [GROUND STATE DOMINATES - ALSO WRONG]")
print(f"\n  CONCLUSION: neither +-sign for tanh(chi) gives I/2 at theta=pi/2.")
print(f"  The error must be in the v5 exponent Delta itself, not just sign convention.")

# ── What v5 formula predicts vs correct at intermediate g ────────────────
print("\n" + "="*70)
print("Intermediate coupling: is v5 moving in the right direction at small g?")
print("Correct: rho_11 should increase toward 0.5 from below as g increases.")
for g in [0.0, 0.3, 0.6]:
    ch = coupling_channels(base_ch, g)
    rho_v5 = v5_theory_state(CFG, ch)
    rho_ed = exact_reduced_state(ed_ctx, g)
    rho_ord = finite_hmf_ordered_gaussian_state(g, ordered_ctx)
    print(f"  g={g:.1f}: v5 rho_11={rho_v5[0,0].real:.5f}  ord rho_11={rho_ord[0,0].real:.5f}  ed rho_11={rho_ed[0,0].real:.5f}")

# ── Cross-check: does the state formula give correct Z_Q? ────────────────
print("\n" + "="*70)
print("Verification: exact Z_Q vs v5 formula Z_Q at g=1")
g = 1.0
ch = coupling_channels(base_ch, g)
dz = ch.delta_z
a_val = 0.5 * BETA * OMEGA_Q
z_exact = 2 * np.cosh(dz - a_val)  # from direct computation
z_v5 = 2 * (np.cosh(a_val) - np.tanh(dz) * np.sinh(a_val))
print(f"  Delta_z = {dz:.5f},  a = {a_val:.5f}")
print(f"  Z_Q from Tr[exp(-bHQ)*exp(Dz*sz)] = 2*cosh(Dz-a) = {z_exact:.5f}")
print(f"  Z_Q from v5 code formula            = {z_v5:.5f}")
print(f"  Match: {abs(z_exact - z_v5) < 1e-4}")

# Actual trace:
rho_bar_11 = np.exp(-a_val + dz)   # exp(-a)*exp(+Dz)
rho_bar_22 = np.exp( a_val - dz)   # exp(+a)*exp(-Dz)
z_from_trace = rho_bar_11 + rho_bar_22
print(f"  Z_Q from direct trace = exp(Dz-a)+exp(a-Dz) = {z_from_trace:.5f}")
rho_11_check = rho_bar_11 / z_from_trace
rho_22_check = rho_bar_22 / z_from_trace
print(f"  rho_11 from exp formula = {rho_11_check:.5f}")
print(f"  rho_22 from exp formula = {rho_22_check:.5f}")
