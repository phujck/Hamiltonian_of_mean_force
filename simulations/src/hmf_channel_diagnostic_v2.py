"""
Investigation Log and Detailed Channel Diagnostic - Round 2.

Goal: Compare RAW v5 channels to the Ordered model's effective channels
to determine where the discrepancy actually lives:
  (a) In the raw Delta_z, Sigma_+, Sigma_- values themselves?
  (b) In the gamma = tanh(chi)/chi resummation?
  (c) Or in the density matrix construction formula?

We extract the Ordered model's "raw" channels by inverting the
gamma formula: if we know gamma*Dz from the Ordered rho, and we
know the Ordered rho populations directly, we can check consistency.

Also: test what happens if we bypass gamma entirely and directly use 
the exponential product form rho = exp(-beta*H_Q) @ exp(Delta).
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm, logm
import itertools

# ── Constants ───────────────────────────────────────────────────────────────
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_MINUS = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

# ── Bath & Kernel (shared) ──────────────────────────────────────────────────
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

# ── v5 channel computation ──────────────────────────────────────────────────
def v5_raw_channels(beta, omega_q, theta, omegas, g2_arr):
    """Compute v5's RAW channels at g=1 (before g^2 scaling)."""
    c = np.cos(theta)
    s = np.sin(theta)
    
    K0 = laplace_K(beta, omegas, g2_arr, 0.0)
    Kp = laplace_K(beta, omegas, g2_arr, omega_q)
    Km = laplace_K(beta, omegas, g2_arr, -omega_q)
    Rp = resonant_R(beta, omegas, g2_arr, omega_q)
    Rm = resonant_R(beta, omegas, g2_arr, -omega_q)
    
    # Manuscript sign-positive form:
    Sigma_plus0 = (c * s / omega_q) * ((1.0 + np.exp(beta * omega_q)) * K0 - 2.0 * Kp)
    Sigma_minus0 = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * K0 - 2.0 * Km)
    Delta_z0 = (s**2) * 0.5 * (Rp - Rm)
    
    return Sigma_plus0, Sigma_minus0, Delta_z0

# ── Ordered model ───────────────────────────────────────────────────────────
def ordered_rho(beta, omega_q, theta, g, omegas, g2_arr):
    c, s = np.cos(theta), np.sin(theta)
    n_slices = 40
    kl_rank = 4
    gh_order = 4
    
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

# ── Channel extraction ────────────────────────────────────────────────────
def extract_eff_channels(rho, beta, omega_q):
    a = 0.5 * beta * omega_q
    M00 = np.exp(a) * rho[0, 0]
    M11 = np.exp(-a) * rho[1, 1]
    M01 = np.exp(a) * rho[0, 1]
    M10 = np.exp(-a) * rho[1, 0]
    denom = np.real(M00 + M11)
    return {
        'gDz': np.real(M00 - M11) / denom,
        'gSp': np.real(2.0 * M01 / denom),
        'gSm': np.real(2.0 * M10 / denom),
    }

def extract_log_channels(rho, beta, omega_q):
    """Extract Delta from rho via matrix logarithm.
    rho = (1/Z) exp(-beta*H_Q) exp(Delta)
    => exp(Delta) = Z * exp(beta*H_Q) @ rho
    => Delta = logm(exp(beta*H_Q) @ rho) + logZ * I
    The traceless part gives Delta_z, Sigma_+, Sigma_-"""
    a = 0.5 * beta * omega_q
    H_Q = 0.5 * omega_q * SIGMA_Z
    M = expm(beta * H_Q) @ rho  # proportional to exp(Delta)
    # M is proportional to exp(Delta) / Z, but logm gives Delta - log(Z)*I
    # The traceless part is what we want
    try:
        log_M = logm(M)
    except:
        return {'Dz': np.nan, 'Sp': np.nan, 'Sm': np.nan}
    
    # Decompose traceless part
    Dz = np.real(log_M[0, 0] - log_M[1, 1]) / 2.0
    Sp = np.real(log_M[0, 1])  # coefficient of sigma_+
    Sm = np.real(log_M[1, 0])  # coefficient of sigma_-
    return {'Dz': Dz, 'Sp': Sp, 'Sm': Sm}

# ── Main ────────────────────────────────────────────────────────────────────
def run():
    beta = 2.0
    omega_q = 1.0
    theta = np.pi / 4
    Q = 10.0
    tau_c = 1.0
    
    omegas, g2_arr = _discrete_bath(beta, omega_q, 40, 0.1, 10.0, Q, tau_c)
    
    # Get raw v5 channels at g=1
    Sp0, Sm0, Dz0 = v5_raw_channels(beta, omega_q, theta, omegas, g2_arr)
    print(f"Raw v5 channels at g=1:")
    print(f"  Sigma_plus_0  = {Sp0:.6f}")
    print(f"  Sigma_minus_0 = {Sm0:.6f}")
    print(f"  Delta_z_0     = {Dz0:.6f}")
    print(f"  Sp*Sm = {Sp0*Sm0:.6f}")
    print()
    
    gs = np.linspace(0.05, 2.0, 25)
    
    rows = []
    for g_val in gs:
        g2 = g_val**2
        Sp = g2 * Sp0
        Sm = g2 * Sm0
        Dz = g2 * Dz0
        
        chi_sq = Dz**2 + Sp * Sm
        chi = np.sqrt(max(chi_sq, 0.0))
        gamma = np.tanh(chi) / chi if abs(chi) > 1e-10 else 1.0
        
        # Ordered model
        rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
        
        # Extract Ordered model's effective log-space channels
        ord_log = extract_log_channels(rho_ord, beta, omega_q)
        
        # The v5 Delta (before gamma resummation) should match ord_log
        # (since exp(Delta_v5) should = exp(Delta_ord))
        ratio_log_Dz = Dz / ord_log['Dz'] if abs(ord_log['Dz']) > 1e-12 else np.nan
        ratio_log_Sp = Sp / ord_log['Sp'] if abs(ord_log['Sp']) > 1e-12 else np.nan
        
        rows.append({
            'g': g_val, 'g2': g2,
            'v5_Dz': Dz, 'v5_Sp': Sp, 'v5_Sm': Sm,
            'v5_chi': chi, 'v5_gamma': gamma,
            'ord_log_Dz': ord_log['Dz'], 'ord_log_Sp': ord_log['Sp'],
            'ratio_log_Dz': ratio_log_Dz,
            'ratio_log_Sp': ratio_log_Sp,
            'v5_p00': np.nan,  # to fill below
            'ord_p00': np.real(rho_ord[0, 0]),
            'ord_coh': np.abs(rho_ord[0, 1]),
        })
        
        print(f"g={g_val:.3f}  v5_Dz={Dz:+.5f}  ord_Dz={ord_log['Dz']:+.5f}  ratio={ratio_log_Dz:.4f}  |  v5_Sp={Sp:+.5f}  ord_Sp={ord_log['Sp']:+.5f}  ratio={ratio_log_Sp:.4f}")
    
    df = pd.DataFrame(rows)
    df.to_csv('hmf_channel_diagnostic_v2.csv', index=False, float_format='%.8f')
    print("\nSaved hmf_channel_diagnostic_v2.csv")

if __name__ == "__main__":
    run()
