"""
Diagnostic v3: Targeted tests to isolate the discrepancy.

Test 1: theta=0 (pure dephasing). All operators commute.
  - v5 derivation is trivially exact.
  - Ordered model should agree perfectly.
  - If they don't, the Ordered model has a numerical issue.

Test 2: theta=pi/2 (pure transverse). Max non-commutation.
  - Compare at weak coupling where perturbation theory is exact.

Test 3: Convergence test for the Ordered model. 
  - Increase n_slices, kl_rank, gh_order and check if results change.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm, logm
import itertools

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_MINUS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

def _discrete_bath(beta, omega_q, n_modes, omega_min, omega_max, q_strength, tau_c):
    omegas = np.linspace(omega_min, omega_max, n_modes, dtype=float)
    delta_omega = float(omegas[1] - omegas[0])
    j0 = q_strength * tau_c * omegas * np.exp(-tau_c * omegas)
    g2 = np.maximum(j0, 0.0) * delta_omega
    return omegas, g2

def _kernel_profile(beta, omegas, g2, u_abs):
    kernel = np.zeros_like(u_abs, dtype=float)
    for omega_k, g2_k in zip(omegas, g2):
        denom = np.sinh(0.5 * beta * omega_k)
        if abs(denom) < 1e-14:
            kernel += (2.0 * g2_k) / max(beta * omega_k, 1e-14)
        else:
            kernel += g2_k * np.cosh(omega_k * (0.5 * beta - u_abs)) / denom
    return kernel

def laplace_K(beta, omegas, g2, omega_val, n_grid=2001):
    u = np.linspace(0.0, beta, n_grid)
    ku = _kernel_profile(beta, omegas, g2, u)
    return float(np.trapezoid(ku * np.exp(omega_val * u), u))

def resonant_R(beta, omegas, g2, omega_val, n_grid=2001):
    u = np.linspace(0.0, beta, n_grid)
    ku = _kernel_profile(beta, omegas, g2, u)
    return float(np.trapezoid((beta - u) * ku * np.exp(omega_val * u), u))

def v5_rho(beta, omega_q, theta, g, omegas, g2_arr):
    """Full v5 density matrix from manuscript (Eq. 282)."""
    c, s = np.cos(theta), np.sin(theta)
    g2 = g**2
    
    K0 = laplace_K(beta, omegas, g2_arr, 0.0)
    Kp = laplace_K(beta, omegas, g2_arr, omega_q)
    Km = laplace_K(beta, omegas, g2_arr, -omega_q)
    Rp = resonant_R(beta, omegas, g2_arr, omega_q)
    Rm = resonant_R(beta, omegas, g2_arr, -omega_q)
    
    Sp = g2 * (c * s / omega_q) * ((1.0 + np.exp(beta * omega_q)) * K0 - 2.0 * Kp)
    Sm = g2 * (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * K0 - 2.0 * Km)
    Dz = g2 * (s**2) * 0.5 * (Rp - Rm)
    
    chi_sq = Dz**2 + Sp * Sm
    chi = np.sqrt(max(chi_sq, 0.0))
    gamma = np.tanh(chi) / chi if abs(chi) > 1e-10 else 1.0
    
    a = 0.5 * beta * omega_q
    gDz = gamma * Dz
    gSp = gamma * Sp
    gSm = gamma * Sm
    
    Z_Q = 2.0 * (np.cosh(a) - gDz * np.sinh(a))
    
    rho = np.array([
        [np.exp(-a) * (1.0 + gDz) / Z_Q, np.exp(-a) * gSp / Z_Q],
        [np.exp(a) * gSm / Z_Q,           np.exp(a) * (1.0 - gDz) / Z_Q]
    ], dtype=complex)
    
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho, {'Dz': Dz, 'Sp': Sp, 'Sm': Sm, 'chi': chi, 'gamma': gamma}

def ordered_rho(beta, omega_q, theta, g, omegas, g2_arr,
                n_slices=40, kl_rank=4, gh_order=4):
    """Ordered Gaussian state with configurable resolution."""
    c, s = np.cos(theta), np.sin(theta)
    
    t = np.linspace(0.0, beta, n_slices)
    dt = t[1] - t[0]
    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, beta - u)
    cov = _kernel_profile(beta, omegas, g2_arr, u)
    
    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    eff_rank = min(kl_rank, np.count_nonzero(evals > 1e-14))
    kl_basis = evecs[:, :eff_rank] * np.sqrt(evals[:eff_rank]) if eff_rank > 0 else np.zeros((len(t), 0))
    
    gh_x, gh_w = np.polynomial.hermite.hermgauss(gh_order)
    eta_1d = np.sqrt(2.0) * gh_x
    weights_1d = gh_w / np.sqrt(np.pi)
    
    def x_tilde(tau):
        return c * SIGMA_Z - s * (np.cosh(omega_q * tau) * SIGMA_X + 1.0j * np.sinh(omega_q * tau) * SIGMA_Y)
    
    x_grid = np.array([x_tilde(tau) for tau in t])
    
    w_avg = np.zeros((2, 2), dtype=complex)
    for inds in itertools.product(range(gh_order), repeat=eff_rank):
        ind_arr = np.asarray(inds, dtype=int)
        eta_vec = eta_1d[ind_arr]
        xi = g * (kl_basis @ eta_vec)
        weight = np.prod(weights_1d[ind_arr])
        
        u_op = IDENTITY_2.copy()
        for n in range(n_slices):
            v = dt * xi[n]
            step = np.cosh(v) * IDENTITY_2 - np.sinh(v) * x_grid[n]
            u_op = step @ u_op
        
        w_avg += weight * u_op
    
    hs = 0.5 * omega_q * SIGMA_Z
    evals_h = np.linalg.eigvals(-beta * hs)
    shift = np.max(np.real(evals_h))
    prefactor = expm(-beta * hs - shift * IDENTITY_2)
    rho = prefactor @ w_avg
    
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho

# ── Tests ────────────────────────────────────────────────────────────────────

def test_dephasing():
    """Test 1: theta=0 (pure dephasing, all commuting)."""
    print("="*70)
    print("TEST 1: PURE DEPHASING (theta=0)")
    print("="*70)
    
    beta = 2.0; omega_q = 1.0; theta = 0.001  # near zero (avoid exact zero for numerics)
    Q = 10.0; tau_c = 1.0
    omegas, g2_arr = _discrete_bath(beta, omega_q, 40, 0.1, 10.0, Q, tau_c)
    
    print(f"{'g':>6s}  {'v5_p00':>8s}  {'ord_p00':>8s}  {'diff':>10s}")
    for g_val in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
        rho_v5, ch = v5_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
        rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
        diff = np.real(rho_v5[0,0] - rho_ord[0,0])
        print(f"{g_val:6.2f}  {np.real(rho_v5[0,0]):8.5f}  {np.real(rho_ord[0,0]):8.5f}  {diff:+10.6f}")

def test_convergence():
    """Test 2: Check Ordered model convergence at fixed g."""
    print()
    print("="*70)
    print("TEST 2: ORDERED MODEL CONVERGENCE (theta=pi/4, g=0.5)")
    print("="*70)
    
    beta = 2.0; omega_q = 1.0; theta = np.pi/4
    Q = 10.0; tau_c = 1.0; g_val = 0.5
    omegas, g2_arr = _discrete_bath(beta, omega_q, 40, 0.1, 10.0, Q, tau_c)
    
    rho_v5, ch = v5_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
    print(f"v5 rho_00 = {np.real(rho_v5[0,0]):.6f}  |rho_01| = {np.abs(rho_v5[0,1]):.6f}")
    print(f"v5 channels: Dz={ch['Dz']:.6f}, Sp={ch['Sp']:.6f}, chi={ch['chi']:.6f}, gamma={ch['gamma']:.6f}")
    print()
    
    configs = [
        (20, 3, 3), (40, 4, 4), (60, 5, 5), (80, 6, 6),
        (100, 8, 6), (120, 10, 6), (60, 4, 8), (80, 6, 8),
    ]
    
    print(f"{'slices':>6s}  {'kl':>4s}  {'gh':>4s}  {'ord_p00':>10s}  {'ord_coh':>10s}")
    for ns, kl, gh in configs:
        try:
            rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr,
                                  n_slices=ns, kl_rank=kl, gh_order=gh)
            print(f"{ns:6d}  {kl:4d}  {gh:4d}  {np.real(rho_ord[0,0]):10.6f}  {np.abs(rho_ord[0,1]):10.6f}")
        except Exception as e:
            print(f"{ns:6d}  {kl:4d}  {gh:4d}  ERROR: {e}")

def test_weak_coupling():
    """Test 3: Weak coupling comparison at theta=pi/2 with high-res Ordered."""
    print()
    print("="*70)
    print("TEST 3: WEAK COUPLING (theta=pi/2, g=0.1) with HIGH-RES Ordered")
    print("="*70)
    
    beta = 2.0; omega_q = 1.0; theta = np.pi/2
    Q = 10.0; tau_c = 1.0
    omegas, g2_arr = _discrete_bath(beta, omega_q, 40, 0.1, 10.0, Q, tau_c)
    
    gs = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    print(f"{'g':>6s}  {'v5_p00':>8s}  {'ord_p00':>8s}  {'diff':>10s}  {'v5_coh':>8s}  {'ord_coh':>8s}")
    for g_val in gs:
        rho_v5, ch = v5_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
        # High-res ordered
        rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr,
                              n_slices=80, kl_rank=8, gh_order=6)
        diff = np.real(rho_v5[0,0] - rho_ord[0,0])
        print(f"{g_val:6.3f}  {np.real(rho_v5[0,0]):8.5f}  {np.real(rho_ord[0,0]):8.5f}  {diff:+10.6f}  {np.abs(rho_v5[0,1]):8.5f}  {np.abs(rho_ord[0,1]):8.5f}")

def test_sign_variants():
    """Test 4: Try different sign conventions for Sigma_+/-."""
    print()
    print("="*70)
    print("TEST 4: SIGN VARIANTS (theta=pi/4, g=0.3)")
    print("="*70)
    
    beta = 2.0; omega_q = 1.0; theta = np.pi/4; g_val = 0.3
    Q = 10.0; tau_c = 1.0
    omegas, g2_arr = _discrete_bath(beta, omega_q, 40, 0.1, 10.0, Q, tau_c)
    
    c, s = np.cos(theta), np.sin(theta)
    g2 = g_val**2
    K0 = laplace_K(beta, omegas, g2_arr, 0.0)
    Kp = laplace_K(beta, omegas, g2_arr, omega_q)
    Km = laplace_K(beta, omegas, g2_arr, -omega_q)
    Rp = resonant_R(beta, omegas, g2_arr, omega_q)
    Rm = resonant_R(beta, omegas, g2_arr, -omega_q)
    
    # Reference: high-res ordered
    rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr,
                          n_slices=80, kl_rank=6, gh_order=6)
    
    a = 0.5 * beta * omega_q
    
    # Construct rho for different sign/scale variants
    variants = {
        'POSITIVE Sigma':   (+(c*s/omega_q) * ((1+np.exp(beta*omega_q))*K0 - 2*Kp),
                             +(c*s/omega_q) * ((1+np.exp(-beta*omega_q))*K0 - 2*Km),
                             +(s**2)*0.5*(Rp-Rm)),
        'NEGATIVE Sigma':   (-(c*s/omega_q) * ((1+np.exp(beta*omega_q))*K0 - 2*Kp),
                             -(c*s/omega_q) * ((1+np.exp(-beta*omega_q))*K0 - 2*Km),
                             +(s**2)*0.5*(Rp-Rm)),
        'NEG Dz':           (+(c*s/omega_q) * ((1+np.exp(beta*omega_q))*K0 - 2*Kp),
                             +(c*s/omega_q) * ((1+np.exp(-beta*omega_q))*K0 - 2*Km),
                             -(s**2)*0.5*(Rp-Rm)),
        'ALL NEGATIVE':     (-(c*s/omega_q) * ((1+np.exp(beta*omega_q))*K0 - 2*Kp),
                             -(c*s/omega_q) * ((1+np.exp(-beta*omega_q))*K0 - 2*Km),
                             -(s**2)*0.5*(Rp-Rm)),
    }
    
    print(f"Ordered rho_00={np.real(rho_ord[0,0]):.6f}, coh={np.abs(rho_ord[0,1]):.6f}")
    print()
    
    for name, (Sp0, Sm0, Dz0) in variants.items():
        Sp = g2 * Sp0; Sm = g2 * Sm0; Dz = g2 * Dz0
        chi_sq = Dz**2 + Sp*Sm
        chi = np.sqrt(max(chi_sq, 0.0))
        gamma = np.tanh(chi)/chi if abs(chi) > 1e-10 else 1.0
        gDz = gamma*Dz; gSp = gamma*Sp; gSm = gamma*Sm
        Z_Q = 2.0*(np.cosh(a) - gDz*np.sinh(a))
        
        rho = np.array([
            [np.exp(-a)*(1+gDz)/Z_Q, np.exp(-a)*gSp/Z_Q],
            [np.exp(a)*gSm/Z_Q,      np.exp(a)*(1-gDz)/Z_Q]
        ], dtype=complex)
        rho = 0.5*(rho + rho.conj().T)
        rho /= np.trace(rho)
        
        diff = np.real(rho[0,0]) - np.real(rho_ord[0,0])
        print(f"  {name:20s}: p00={np.real(rho[0,0]):.6f}  coh={np.abs(rho[0,1]):.6f}  diff={diff:+.6f}  Dz={Dz:.4f}  Sp={Sp:.4f}")

if __name__ == "__main__":
    test_dephasing()
    test_convergence()
    test_weak_coupling()
    test_sign_variants()
