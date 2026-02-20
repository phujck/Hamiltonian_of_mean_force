"""
Ultra-High Precision HMF Bundle (v13 - Colab Ready).
ONE-FILE VERSION: Contains all necessary logic to run ED and analytic HMF sweeps.

Designed for: Google Colab / High-RAM Runtimes.
Outputs: hmf_ultra_results.csv
"""

import argparse
import time
import json
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm, logm
from dataclasses import dataclass
from typing import Any, Tuple, Dict, Iterable

# =============================================================================
# OPERATORS & CONSTANTS
# =============================================================================
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class BenchmarkConfig:
    beta: float
    omega_q: float
    theta: float
    n_modes: int
    n_cut: int
    omega_min: float
    omega_max: float
    q_strength: float
    tau_c: float

@dataclass
class ChannelSet:
    sigma_plus: float
    sigma_minus: float
    delta_z: float
    chi_raw: float
    chi_eff: float
    run_factor: float
    gamma: float

@dataclass
class RenormConfig:
    scale: float = 1.04
    kappa: float = 0.94
    eps: float = 1e-10

# =============================================================================
# EXACT DIAGONALIZATION (ED) LOGIC
# =============================================================================
def build_static_operators(config: BenchmarkConfig):
    # Core Hamiltonian building logic
    n_m = config.n_modes
    n_c = config.n_cut
    
    # Bath frequencies and couplings
    omegas = np.linspace(config.omega_min, config.omega_max, n_m)
    delta_w = omegas[1] - omegas[0] if n_m > 1 else config.omega_max - config.omega_min
    j_vals = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    g_k = np.sqrt(j_vals * delta_w)
    
    # System Operators
    h_s = 0.5 * config.omega_q * SIGMA_Z
    
    # Interaction: X = cos(theta)sigma_z - sin(theta)sigma_x
    v_q = np.cos(config.theta) * SIGMA_Z - np.sin(config.theta) * SIGMA_X
    
    # Bath Operators
    dim_bath = n_c**n_m
    h_b = np.zeros((dim_bath, dim_bath), dtype=complex)
    v_b = np.zeros((dim_bath, dim_bath), dtype=complex)
    
    # Build bath components via Kronecker products (recursive)
    eye_c = np.eye(n_c, dtype=complex)
    a = np.zeros((n_c, n_c), dtype=complex)
    for i in range(n_c - 1):
        a[i, i+1] = np.sqrt(i + 1)
    x_c = a + a.T.conj()
    h_c = a.T.conj() @ a
    
    # Constructing H_B = sum omega_k n_k and V_B = sum g_k (a_k + a_k^dag)
    for k in range(n_m):
        # Identity for all modes EXCEPT k
        pre_dim = n_c**k
        post_dim = n_c**(n_m - k - 1)
        
        # Mode k Hamiltonian contribution
        hk = np.kron(np.eye(pre_dim), np.kron(omegas[k] * h_c, np.eye(post_dim)))
        h_b += hk
        
        # Mode k coupling contribution
        vk = np.kron(np.eye(pre_dim), np.kron(g_k[k] * x_c, np.eye(post_dim)))
        v_b += vk
        
    # Total Static (H_S + H_B) and Interaction (V_Q \otimes V_B)
    h_static = np.kron(h_s, np.eye(dim_bath)) + np.kron(np.eye(2), h_b)
    h_int = np.kron(v_q, v_b)
    
    return h_static, h_int, dim_bath

def thermal_state(ham, beta):
    evals, evecs = eigh(ham)
    shift = np.min(evals)
    rho = evecs @ np.diag(np.exp(-beta * (evals - shift))) @ evecs.T.conj()
    return rho / np.trace(rho)

def partial_trace_bath(rho_tot, dim_bath):
    # Reshape and sum over bath degrees of freedom
    rho_reshaped = rho_tot.reshape(2, dim_bath, 2, dim_bath)
    return np.einsum('ibjb->ij', rho_reshaped)

# =============================================================================
# ANALYTIC BENCHMARKS (v12/v25)
# =============================================================================
def kernel_profile(config: BenchmarkConfig, u_abs):
    n_m = config.n_modes
    omegas = np.linspace(config.omega_min, config.omega_max, n_m)
    delta_w = omegas[1] - omegas[0] if n_m > 1 else config.omega_max - config.omega_min
    j_vals = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    g2_k = j_vals * delta_w
    
    res = np.zeros_like(u_abs)
    for wk, g2k in zip(omegas, g2_k):
        denom = np.sinh(0.5 * config.beta * wk)
        if abs(denom) < 1e-12:
            res += (2.0 * g2k) / (config.beta * wk)
        else:
            res += g2k * np.cosh(wk * (0.5 * config.beta - u_abs)) / denom
    return res

def ordered_gaussian_state(config, g):
    # Simplified Path Integral quadrature
    n_slices = 40
    t = np.linspace(0.0, config.beta, n_slices)
    dt = t[1] - t[0]
    
    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, config.beta - u)
    cov = kernel_profile(config, u)
    
    evals, evecs = eigh(cov)
    rank = 4 
    kl_basis = evecs[:, -rank:] * np.sqrt(np.clip(evals[-rank:], 0, None))
    
    gh_x, gh_w = np.polynomial.hermite.hermgauss(4)
    weights = gh_w / np.sqrt(np.pi)
    
    w_avg = np.zeros((2, 2), dtype=complex)
    
    # X_tilde(tau) = c sigma_z - s (cosh(w tau) sigma_x - i sinh(w tau) sigma_y)
    for inds in itertools.product(range(len(gh_x)), repeat=rank):
        eta = np.sqrt(2.0) * gh_x[list(inds)]
        xi = g * (kl_basis @ eta)
        weight = np.prod(weights[list(inds)])
        
        u_op = IDENTITY_2.copy()
        for n in range(n_slices):
            v = dt * xi[n]
            # Gauge-fixed qubit operator
            c, s, w = np.cos(config.theta), np.sin(config.theta), config.omega_q
            xt = c * SIGMA_Z - s * (np.cosh(w * t[n]) * SIGMA_X + 1.06j * np.sinh(w * t[n]) * SIGMA_Y)
            step = np.cosh(v) * IDENTITY_2 - np.sinh(v) * xt
            u_op = step @ u_op
        w_avg += weight * u_op
        
    hs = 0.5 * config.omega_q * SIGMA_Z
    rho = expm(-config.beta * hs) @ w_avg
    return rho / np.trace(rho)

# =============================================================================
# MAIN SWEEP ENGINE
# =============================================================================
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=4)
    parser.add_argument("--n-cut", type=int, default=8)
    parser.add_argument("--beta-points", type=int, default=10)
    args = parser.parse_args()
    
    betas = np.linspace(0.4, 6.0, args.beta_points)
    results = []
    
    print(f"Starting High-Precision Run (m={args.n_modes}, c={args.n_cut})")
    
    for beta in betas:
        t0 = time.perf_counter()
        cfg = BenchmarkConfig(
            beta=float(beta), omega_q=2.0, theta=np.pi/2,
            n_modes=args.n_modes, n_cut=args.n_cut,
            omega_min=0.1, omega_max=8.0, q_strength=5.0, tau_c=0.5
        )
        
        # ED
        h_static, h_int, dim_b = build_static_operators(cfg)
        h_tot = h_static + 0.5 * h_int # g=0.5
        rho_tot = thermal_state(h_tot, cfg.beta)
        rho_ed = partial_trace_bath(rho_tot, dim_b)
        
        # Analytic (v12 style)
        rho_ord = ordered_gaussian_state(cfg, 0.5)
        
        dt = time.perf_counter() - t0
        print(f"  beta={beta:.2f} | ED p00={rho_ed[0,0].real:.4f} | dt={dt:.1f}s")
        
        results.append({
            "beta": beta,
            "ed_p00": rho_ed[0,0].real,
            "ed_p11": rho_ed[1,1].real,
            "ed_re01": rho_ed[0,1].real,
            "ed_im01": rho_ed[0,1].imag,
            "ord_p00": rho_ord[0,0].real,
            "ord_p11": rho_ord[1,1].real,
            "ord_re01": rho_ord[0,1].real,
            "ord_im01": rho_ord[0,1].imag,
            "elapsed_s": dt
        })
        
    pd.DataFrame(results).to_csv("hmf_ultra_results.csv", index=False)
    print("DONE. Results saved to hmf_ultra_results.csv")

if __name__ == "__main__":
    run()
