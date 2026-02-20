"""
Channel Diagnostic: Extract effective Delta channels from both v5 and Ordered models.

For each coupling g, we:
1. Compute v5's raw Delta_z, Sigma_+, Sigma_- (and gamma, chi)
2. Compute the Ordered model's rho
3. Extract the Ordered model's effective channels via M = exp(beta*H_Q) @ rho
   The decomposition is: M = (1/Z_Q) * (I + gamma_eff * M_hat)
   So gamma_eff * Delta_z_eff = (M00 - M11) / (M00 + M11)
4. Compare channel-by-channel and output ratios to CSV
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm
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
    """K(omega) = int_0^beta K(u) exp(omega*u) du"""
    u = np.linspace(0.0, beta, n_grid)
    ku = _kernel_profile(beta, omegas, g2, u)
    return float(np.trapezoid(ku * np.exp(omega_val * u), u))

def resonant_R(beta, omegas, g2, omega_val, n_grid=2001):
    """R(omega) = int_0^beta (beta-u) K(u) exp(omega*u) du"""
    u = np.linspace(0.0, beta, n_grid)
    ku = _kernel_profile(beta, omegas, g2, u)
    return float(np.trapezoid((beta - u) * ku * np.exp(omega_val * u), u))

# ── v5 channel computation ──────────────────────────────────────────────────
def v5_channels(beta, omega_q, theta, g, omegas, g2_arr):
    """Compute v5's raw channels and derived quantities."""
    c = np.cos(theta)
    s = np.sin(theta)
    
    K0 = laplace_K(beta, omegas, g2_arr, 0.0)
    Kp = laplace_K(beta, omegas, g2_arr, omega_q)
    Km = laplace_K(beta, omegas, g2_arr, -omega_q)
    Rp = resonant_R(beta, omegas, g2_arr, omega_q)
    Rm = resonant_R(beta, omegas, g2_arr, -omega_q)
    
    # Manuscript Eqs. 205-206 (POSITIVE sign):
    # Sigma_+ = (cs/omega_q) * [(1+exp(beta*omega_q))*K(0) - 2*K(omega_q)]
    # Sigma_- = (cs/omega_q) * [(1+exp(-beta*omega_q))*K(0) - 2*K(-omega_q)]
    Sigma_plus_raw = (c * s / omega_q) * ((1.0 + np.exp(beta * omega_q)) * K0 - 2.0 * Kp)
    Sigma_minus_raw = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * K0 - 2.0 * Km)
    
    # Manuscript Eq. 227: Delta_z = s^2 * R^-
    # R^- = 0.5 * (R(omega_q) - R(-omega_q))
    Delta_z_raw = (s**2) * 0.5 * (Rp - Rm)
    
    # Scale by g^2
    Sp = g**2 * Sigma_plus_raw
    Sm = g**2 * Sigma_minus_raw
    Dz = g**2 * Delta_z_raw
    
    # chi, gamma
    chi_sq = Dz**2 + Sp * Sm
    chi = np.sqrt(max(chi_sq, 0.0))
    gamma = np.tanh(chi) / chi if abs(chi) > 1e-10 else 1.0
    
    return {
        'K0': K0, 'Kp': Kp, 'Km': Km, 'Rp': Rp, 'Rm': Rm,
        'Sigma_plus': Sp, 'Sigma_minus': Sm, 'Delta_z': Dz,
        'chi': chi, 'gamma': gamma,
        'gamma_Dz': gamma * Dz,
        'gamma_Sp': gamma * Sp,
        'gamma_Sm': gamma * Sm,
    }

# ── v5 density matrix ──────────────────────────────────────────────────────
def v5_rho(beta, omega_q, channels):
    """Construct rho from v5 channels using Eq. 282."""
    a = 0.5 * beta * omega_q
    gDz = channels['gamma_Dz']
    gSp = channels['gamma_Sp']
    gSm = channels['gamma_Sm']
    
    Z_Q = 2.0 * (np.cosh(a) - gDz * np.sinh(a))
    
    rho = np.array([
        [np.exp(-a) * (1.0 + gDz) / Z_Q, np.exp(-a) * gSp / Z_Q],
        [np.exp(a) * gSm / Z_Q,           np.exp(a) * (1.0 - gDz) / Z_Q]
    ], dtype=complex)
    
    # Symmetrize
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho

# ── Ordered model ───────────────────────────────────────────────────────────
def ordered_rho(beta, omega_q, theta, g, omegas, g2_arr):
    """Ordered Gaussian state via KL + Gauss-Hermite."""
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
    
    # X_tilde(tau) operators
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
    
    # rho = exp(-beta*H_Q) @ <T exp(...)>
    hs = 0.5 * omega_q * SIGMA_Z
    evals_h = np.linalg.eigvals(-beta * hs)
    shift = np.max(np.real(evals_h))
    prefactor = expm(-beta * hs - shift * IDENTITY_2)
    rho = prefactor @ w_avg
    
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho

# ── Channel extraction from density matrix ──────────────────────────────────
def extract_channels(rho, beta, omega_q):
    """Extract effective gamma*Delta_z, gamma*Sigma_+, gamma*Sigma_- from rho.
    
    From Eq. 282: rho = (1/Z_Q) * diag(e^-a, e^a) @ (I + gamma*M_hat)
    So M = diag(e^a, e^-a) @ rho = (1/Z_Q) * (I + gamma*M_hat)
    
    gamma*Delta_z = (M00 - M11) / (M00 + M11)
    gamma*Sigma_+ = 2*M01 / (M00 + M11)
    gamma*Sigma_- = 2*M10 / (M00 + M11)
    """
    a = 0.5 * beta * omega_q
    
    # M = exp(beta*H_Q) @ rho = diag(e^a, e^-a) @ rho
    M00 = np.exp(a) * rho[0, 0]
    M11 = np.exp(-a) * rho[1, 1]
    M01 = np.exp(a) * rho[0, 1]
    M10 = np.exp(-a) * rho[1, 0]
    
    denom = np.real(M00 + M11)
    
    gDz = np.real(M00 - M11) / denom
    gSp = 2.0 * M01 / denom  # May be complex
    gSm = 2.0 * M10 / denom
    
    return {
        'gamma_Dz': gDz,
        'gamma_Sp': np.real(gSp),
        'gamma_Sm': np.real(gSm),
    }

# ── Main diagnostic ─────────────────────────────────────────────────────────
def run_diagnostic():
    """Run channel comparison across coupling sweep."""
    # Parameters
    beta = 2.0
    omega_q = 1.0
    theta = np.pi / 4
    Q = 10.0
    tau_c = 1.0
    n_modes = 40
    omega_min = 0.1
    omega_max = 10.0
    
    omegas, g2_arr = _discrete_bath(beta, omega_q, n_modes, omega_min, omega_max, Q, tau_c)
    
    gs = np.linspace(0.05, 2.5, 30)
    
    rows = []
    for g_val in gs:
        print(f"  g = {g_val:.3f}")
        
        # v5 channels
        ch = v5_channels(beta, omega_q, theta, g_val, omegas, g2_arr)
        
        # Ordered rho
        rho_ord = ordered_rho(beta, omega_q, theta, g_val, omegas, g2_arr)
        
        # v5 rho (using positive-sign manuscript channels, NO scale factor)
        rho_v5 = v5_rho(beta, omega_q, ch)
        
        # Extract effective channels from both
        ch_v5_eff = extract_channels(rho_v5, beta, omega_q)
        ch_ord_eff = extract_channels(rho_ord, beta, omega_q)
        
        # Ratios
        ratio_Dz = ch['gamma_Dz'] / ch_ord_eff['gamma_Dz'] if abs(ch_ord_eff['gamma_Dz']) > 1e-12 else np.nan
        ratio_Sp = ch['gamma_Sp'] / ch_ord_eff['gamma_Sp'] if abs(ch_ord_eff['gamma_Sp']) > 1e-12 else np.nan
        
        rows.append({
            'g': g_val,
            'g2': g_val**2,
            # Raw v5 channels
            'v5_Dz': ch['Delta_z'],
            'v5_Sp': ch['Sigma_plus'],
            'v5_Sm': ch['Sigma_minus'],
            'v5_chi': ch['chi'],
            'v5_gamma': ch['gamma'],
            # v5 effective (gamma * channel)
            'v5_gDz': ch['gamma_Dz'],
            'v5_gSp': ch['gamma_Sp'],
            'v5_gSm': ch['gamma_Sm'],
            # Ordered effective channels
            'ord_gDz': ch_ord_eff['gamma_Dz'],
            'ord_gSp': ch_ord_eff['gamma_Sp'],
            'ord_gSm': ch_ord_eff['gamma_Sm'],
            # Ratios: v5 / Ordered
            'ratio_gDz': ratio_Dz,
            'ratio_gSp': ratio_Sp,
            # Density matrix elements
            'v5_p00': np.real(rho_v5[0, 0]),
            'ord_p00': np.real(rho_ord[0, 0]),
            'v5_coh': np.abs(rho_v5[0, 1]),
            'ord_coh': np.abs(rho_ord[0, 1]),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv('hmf_channel_diagnostic.csv', index=False, float_format='%.8f')
    print("Saved hmf_channel_diagnostic.csv")
    
    # Print summary
    print("\n=== CHANNEL RATIO SUMMARY (v5 / Ordered) ===")
    print(f"{'g':>6s}  {'ratio_gDz':>12s}  {'ratio_gSp':>12s}  {'v5_p00':>8s}  {'ord_p00':>8s}")
    for _, r in df.iterrows():
        print(f"{r['g']:6.3f}  {r['ratio_gDz']:12.4f}  {r['ratio_gSp']:12.4f}  {r['v5_p00']:8.4f}  {r['ord_p00']:8.4f}")
    
    return df

if __name__ == "__main__":
    run_diagnostic()
