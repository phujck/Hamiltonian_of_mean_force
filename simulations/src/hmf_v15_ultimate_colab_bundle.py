"""
ULTIMATE HMF SIMULATION BUNDLE (v15 - "One-Click" Colab Edition)
--------------------------------------------------------------
Features:
- Sparse Matrix Engine (scipy.sparse) for 100,000+ states.
- Truncated Thermal State (Lanczos/eigsh) for memory efficiency.
- Full Density Matrix Recording (p00, p11, re01, im01).
- Integrated Plotting (Matplotlib) - See results instantly in Colab.
- Auto-Run Mode: Just paste and press Shift+Enter.
"""

from __future__ import annotations

import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import kron, diags, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. CORE OPERATORS & PHYSICS
# =============================================================================
SX = csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = csr_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = eye(2, dtype=complex, format='csr')

@dataclass
class PhysicsConfig:
    beta: float = 2.0
    omega_q: float = 2.0
    theta: float = 1.5708  # pi/2
    g: float = 0.5
    # Bath
    n_modes: int = 5
    n_cut: int = 10
    omega_min: float = 0.1
    omega_max: float = 8.0
    q_strength: float = 5.0
    tau_c: float = 0.5
    # Numerical
    n_eig: int = 50
    # Analytic
    scale: float = 1.04
    kappa: float = 0.94

@dataclass
class ChannelSet:
    sigma_plus: float
    sigma_minus: float
    delta_z: float
    chi_raw: float
    chi_eff: float
    run_factor: float
    gamma: float

# =============================================================================
# 2. HAMILTONIAN CONSTRUCTION (SPARSE)
# =============================================================================
def build_hamiltonian(c: PhysicsConfig):
    n_m, n_c = c.n_modes, c.n_cut
    omegas = np.linspace(c.omega_min, c.omega_max, n_m)
    delta_w = omegas[1] - omegas[0] if n_m > 1 else c.omega_max - c.omega_min
    j_vals = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    g_k = np.sqrt(j_vals * delta_w)
    
    # Single mode operators
    a = np.zeros((n_c, n_c), dtype=complex)
    for i in range(n_c - 1): a[i, i+1] = np.sqrt(i + 1)
    x_c = csr_matrix(a + a.T.conj())
    n_c_op = csr_matrix(a.T.conj() @ a)
    
    # System bits
    h_s = 0.5 * c.omega_q * SZ
    v_q = np.cos(c.theta) * SZ - np.sin(c.theta) * SX
    
    # Bath bits (Kronecker sums)
    dim_bath = n_c**n_m
    h_b = csr_matrix((dim_bath, dim_bath), dtype=complex)
    v_b = csr_matrix((dim_bath, dim_bath), dtype=complex)
    
    for k in range(n_m):
        pre, post = n_c**k, n_c**(n_m - k - 1)
        # H_b = sum omega_k n_k
        h_b += kron(eye(pre), kron(omegas[k] * n_c_op, eye(post)), format='csr')
        # V_b = sum g_k x_k
        v_b += kron(eye(pre), kron(g_k[k] * x_c, eye(post)), format='csr')
        
    h_tot = kron(h_s, eye(dim_bath)) + kron(I2, h_b) + c.g * kron(v_q, v_b)
    return h_tot, dim_bath

# =============================================================================
# 3. THERMAL STATE SOLVER
# =============================================================================
def solve_density_matrix(h_tot: csr_matrix, c: PhysicsConfig):
    dim_tot = h_tot.shape[0]
    n_eig = min(c.n_eig, dim_tot - 2)
    
    # Solve for lowest eigenvalues (Lanczos)
    evals, evecs = eigsh(h_tot, k=n_eig, which='SA')
    
    # Boltzmann weighting
    shift = np.min(evals)
    weights = np.exp(-c.beta * (evals - shift))
    z_sum = np.sum(weights)
    
    # Reduced density matrix construction
    dim_bath = dim_tot // 2
    rho_q = np.zeros((2, 2), dtype=complex)
    
    for i in range(n_eig):
        w = weights[i] / z_sum
        psi = evecs[:, i]
        psi0, psi1 = psi[:dim_bath], psi[dim_bath:]
        rho_q[0, 0] += w * np.vdot(psi0, psi0)
        rho_q[1, 1] += w * np.vdot(psi1, psi1)
        rho_q[0, 1] += w * np.vdot(psi0, psi1)
        rho_q[1, 0] += w * np.vdot(psi1, psi0)
        
    return rho_q

# =============================================================================
# 4. ANALYTIC BENCHMARK LOGIC (v12 / v25)
# =============================================================================
def kernel_profile(c: PhysicsConfig, u_abs):
    # Use finer discretization for analytic integrals
    omegas = np.linspace(c.omega_min, c.omega_max, 100)
    dw = omegas[1] - omegas[0]
    j_vals = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    g2_k = j_vals * dw
    
    res = np.zeros_like(u_abs)
    for wk, g2k in zip(omegas, g2_k):
        denom = np.sinh(0.5 * c.beta * wk)
        if abs(denom) < 1e-12:
            res += (2.0 * g2k) / (c.beta * wk)
        else:
            res += g2k * np.cosh(wk * (0.5 * c.beta - u_abs)) / denom
    return res

def ordered_gaussian_state(c: PhysicsConfig):
    n_slices = 40
    t = np.linspace(0.0, c.beta, n_slices)
    dt = t[1] - t[0]
    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, c.beta - u)
    cov = kernel_profile(c, u)
    
    evals, evecs = np.linalg.eigh(cov)
    rank = 4
    kl = evecs[:, -rank:] * np.sqrt(np.clip(evals[-rank:], 0, None))
    
    gx, gw = np.polynomial.hermite.hermgauss(4)
    weights = gw / np.sqrt(np.pi)
    w_avg = np.zeros((2, 2), dtype=complex)
    
    for inds in itertools.product(range(len(gx)), repeat=rank):
        eta = np.sqrt(2.0) * gx[list(inds)]
        xi = c.g * (kl @ eta)
        weight = np.prod(weights[list(inds)])
        
        u_op = np.eye(2, dtype=complex)
        for n in range(n_slices):
            v, tau = dt * xi[n], t[n]
            # X_tilde(tau)
            xt = np.cos(c.theta)*SZ - np.sin(c.theta)*(np.cosh(c.omega_q*tau)*SX + 1.0j*np.sinh(c.omega_q*tau)*SY)
            step = np.cosh(v)*np.eye(2, dtype=complex) - np.sinh(v)*xt
            u_op = step @ u_op
        w_avg += weight * u_op
        
    hs = 0.5 * c.omega_q * SZ
    from scipy.linalg import expm
    rho = expm(-c.beta * hs.toarray()) @ w_avg
    return rho / np.trace(rho)

def solve_best_analytic(c: PhysicsConfig):
    # Simplified compact v5 model with running renormalization
    omegas = np.linspace(c.omega_min, c.omega_max, 200)
    dw = omegas[1] - omegas[0]
    j = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    g2 = j * dw
    
    u = np.linspace(0, c.beta, 501)
    k0 = np.zeros_like(u)
    for wk, gk2 in zip(omegas, g2):
        k0 += gk2 * np.cosh(wk * (0.5 * c.beta - u)) / np.sinh(0.5 * c.beta * wk)
        
    k_plus = np.trapezoid(k0 * np.exp(c.omega_q * u), u)
    k_minus = np.trapezoid(k0 * np.exp(-c.omega_q * u), u)
    k0_val = np.trapezoid(k0, u)
    r_plus = np.trapezoid((c.beta - u) * k0 * np.exp(c.omega_q * u), u)
    r_minus = np.trapezoid((c.beta - u) * k0 * np.exp(-c.omega_q * u), u)
    
    ct, st = np.cos(c.theta), np.sin(c.theta)
    b, wq = c.beta, c.omega_q
    
    s_plus0 = (ct*st/wq) * ((1 + np.exp(b*wq))*k0_val - 2*k_plus)
    s_minus0 = (ct*st/wq) * ((1 + np.exp(-b*wq))*k0_val - 2*k_minus)
    dz0 = st**2 * 0.5 * (r_plus - r_minus)
    
    c2 = c.g**2 * c.scale
    sp, sm, dz = c2*s_plus0, c2*s_minus0, c2*dz0
    
    chi_raw = np.sqrt(max(dz**2 + sp*sm, 0))
    a = 0.5 * b * wq
    chi_cap = max(c.kappa * a, 1e-10)
    run = 1.0 / (1.0 + chi_raw / chi_cap)
    
    chi_eff = run * chi_raw
    gamma = np.tanh(chi_eff)/chi_eff if chi_eff > 1e-9 else 1.0
    
    zq = 2.0 * (np.cosh(a) - gamma * (run*dz) * np.sinh(a))
    rho = np.array([
        [np.exp(-a)*(1 + gamma*run*dz), np.exp(-a)*gamma*run*sp],
        [np.exp(a)*gamma*run*sm, np.exp(a)*(1 - gamma*run*dz)]
    ], dtype=complex) / zq
    return rho

# =============================================================================
# 5. ONE-CLICK EXECUTION & PLOTTING
# =============================================================================
def run_ultimate_simulation():
    print("🚀 [ULTIMATE BUNDLE] Starting Comparison Sweep...")
    
    base_cfg = PhysicsConfig(n_modes=5, n_cut=10, n_eig=50)
    betas = np.linspace(0.4, 6.0, 12)
    results = []
    
    print(f"🔧 Building 100k state Hamiltonian (ED)...")
    h_tot, _ = build_hamiltonian(base_cfg)
    
    for beta in betas:
        t0 = time.perf_counter()
        base_cfg.beta = float(beta)
        
        # 1. ED Result
        rho_ed = solve_density_matrix(h_tot, base_cfg)
        
        # 2. Ordered Gaussian
        rho_ord = ordered_gaussian_state(base_cfg)
        
        # 3. Best Renormalized Analytic
        rho_best = solve_best_analytic(base_cfg)
        
        dt = time.perf_counter() - t0
        print(f"  beta={beta:.2f} | ED={rho_ed[0,0].real:.4f} | Ord={rho_ord[0,0].real:.4f} | Best={rho_best[0,0].real:.4f} | {dt:.1f}s")
        
        results.append({
            "beta": beta,
            "ed_p00": rho_ed[0,0].real,
            "ed_re01": rho_ed[0,1].real,
            "ord_p00": rho_ord[0,0].real,
            "ord_re01": rho_ord[0,1].real,
            "best_p00": rho_best[0,0].real,
            "best_re01": rho_best[0,1].real,
        })
        
    df = pd.DataFrame(results)
    df.to_csv("hmf_high_precision_comparison.csv", index=False)
    
    # 6. INTEGRATED COMPARISON PLOTS
    print("\n📊 Generating Side-by-Side Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Population Comparison
    axes[0].plot(df["beta"], df["ed_p00"], 'ko', label="Exact Diagonalization", markersize=8)
    axes[0].plot(df["beta"], df["ord_p00"], 'r--', label="Ordered Gaussian", alpha=0.7)
    axes[0].plot(df["beta"], df["best_p00"], 'g-', label="Best Analytic (Renorm)", linewidth=2)
    axes[0].set_title(r"Longitudinal Population $\rho_{00}$")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Coherence/Off-diagonal Comparison
    axes[1].plot(df["beta"], df["ed_re01"], 'ko', label="Exact Diagonalization")
    axes[1].plot(df["beta"], df["ord_re01"], 'r--', label="Ordered Gaussian", alpha=0.7)
    axes[1].plot(df["beta"], df["best_re01"], 'g-', label="Best Analytic (Renorm)", linewidth=2)
    axes[1].set_title(r"Transverse Coherence $\mathrm{Re}[\rho_{01}]$")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("🎉 Done! All simulations complete.")

if __name__ == "__main__":
    run_ultimate_simulation()
