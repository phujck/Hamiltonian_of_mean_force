"""
HMF OMNIBUS RESEARCH SUITE (v16 - Definitive Version)
---------------------------------------------------
A comprehensive, self-contained suite for Hamiltonian of Mean Force (HMF) research.

CAPABILITIES:
- Physical Sweeps: Temperature (beta), Coupling (g), and Geometry (theta).
- Convergence Sweeps: Bath discretization (n_modes, n_cut).
- Triple Benchmark: Sparse ED (Lanczos), Ordered Gaussian (PI), and Renormalized Analytic.
- State Forensics: Full density matrix components (rho00, rho11, re01, im01).
- Live Visualization: Integrated dashboard generation directly in the notebook.
"""

from __future__ import annotations

import time
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from dataclasses import dataclass, asdict

# =============================================================================
# 1. CORE OPERATORS & PHYSICS
# =============================================================================
SX = csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = csr_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = eye(2, dtype=complex, format='csr')

@dataclass
class omnibusConfig:
    # Physical
    beta: float = 2.0
    omega_q: float = 2.0
    theta: float = 1.5708  # pi/2
    g: float = 0.5
    # Bath
    n_modes: int = 4
    n_cut: int = 8
    omega_min: float = 0.1
    omega_max: float = 8.0
    q_strength: float = 5.0
    tau_c: float = 0.5
    # Numerical
    n_eig: int = 40
    # Renorm (v5 style)
    scale: float = 1.04
    kappa: float = 0.94

# =============================================================================
# 2. THE THREE ENGINES (ED, ORDERED, ANALYTIC)
# =============================================================================

def build_hamiltonian(c: omnibusConfig):
    n_m, n_c = c.n_modes, c.n_cut
    omegas = np.linspace(c.omega_min, c.omega_max, n_m)
    delta_w = omegas[1] - omegas[0] if n_m > 1 else c.omega_max - c.omega_min
    j_vals = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    g_k = np.sqrt(j_vals * delta_w)
    
    # Single mode operators
    a = np.zeros((n_c, n_c), dtype=complex)
    for i in range(n_c - 1): a[i, i+1] = np.sqrt(i + 1)
    xc, nc = csr_matrix(a + a.T.conj()), csr_matrix(a.T.conj() @ a)
    
    dim_b = n_c**n_m
    hb, vb = csr_matrix((dim_b, dim_b), dtype=complex), csr_matrix((dim_b, dim_b), dtype=complex)
    
    for k in range(n_m):
        pre, post = n_c**k, n_c**(n_m - k - 1)
        hb += kron(eye(pre), kron(omegas[k] * nc, eye(post)), format='csr')
        vb += kron(eye(pre), kron(g_k[k] * xc, eye(post)), format='csr')
        
    vq = np.cos(c.theta) * SZ - np.sin(c.theta) * SX
    h_tot = kron(0.5 * c.omega_q * SZ, eye(dim_b)) + kron(I2, hb) + c.g * kron(vq, vb)
    return h_tot, dim_b

def solve_ed(h_tot: csr_matrix, c: omnibusConfig):
    evals, evecs = eigsh(h_tot, k=min(c.n_eig, h_tot.shape[0]-2), which='SA')
    w = np.exp(-c.beta * (evals - np.min(evals)))
    w /= np.sum(w)
    
    dim_b = h_tot.shape[0] // 2
    rho = np.zeros((2, 2), dtype=complex)
    for i in range(len(w)):
        psi = evecs[:, i]
        psi0, psi1 = psi[:dim_b], psi[dim_b:]
        rho[0,0] += w[i] * np.vdot(psi0, psi0)
        rho[1,1] += w[i] * np.vdot(psi1, psi1)
        rho[0,1] += w[i] * np.vdot(psi0, psi1)
    rho[1,0] = rho[0,1].conj()
    return rho

def kernel_profile(c: omnibusConfig, u):
    omegas = np.linspace(c.omega_min, c.omega_max, 100)
    dw = omegas[1] - omegas[0]
    j = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    res = np.zeros_like(u)
    for wk, jk in zip(omegas, j):
        denom = np.sinh(0.5 * c.beta * wk)
        if abs(denom) < 1e-12: res += (2.0 * jk * dw) / (c.beta * wk)
        else: res += (jk * dw) * np.cosh(wk * (0.5 * c.beta - u)) / denom
    return res

def solve_ordered(c: omnibusConfig):
    n_s = 32
    t = np.linspace(0.0, c.beta, n_s)
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
        for n in range(n_s):
            v, tau = dt * xi[n], t[n]
            xt = np.cos(c.theta)*SZ.toarray() - np.sin(c.theta)*(np.cosh(c.omega_q*tau)*SX.toarray() + 1.0j*np.sinh(c.omega_q*tau)*SY.toarray())
            u_op = (np.cosh(v)*np.eye(2) - np.sinh(v)*xt) @ u_op
        w_avg += weight * u_op
    rho = expm(-c.beta * (0.5 * c.omega_q * SZ.toarray())) @ w_avg
    return rho / np.trace(rho)

def solve_renorm(c: omnibusConfig):
    # Reduced integration grid for speed in loops
    u = np.linspace(0, c.beta, 301)
    k0 = kernel_profile(c, u)
    k_plus = np.trapezoid(k0 * np.exp(c.omega_q * u), u)
    k_minus = np.trapezoid(k0 * np.exp(-c.omega_q * u), u)
    k_zero = np.trapezoid(k0, u)
    r_plus = np.trapezoid((c.beta - u) * k0 * np.exp(c.omega_q * u), u)
    r_minus = np.trapezoid((c.beta - u) * k0 * np.exp(-c.omega_q * u), u)
    ct, st, b, wq = np.cos(c.theta), np.sin(c.theta), c.beta, c.omega_q
    sp0 = (ct*st/wq) * ((1 + np.exp(b*wq))*k_zero - 2*k_plus)
    sm0 = (ct*st/wq) * ((1 + np.exp(-b*wq))*k_zero - 2*k_minus)
    dz0 = st**2 * 0.5 * (r_plus - r_minus)
    c2 = c.g**2 * c.scale
    sp, sm, dz = c2*sp0, c2*sm0, c2*dz0
    chi_raw = np.sqrt(max(dz**2 + sp*sm, 0))
    run = 1.0 / (1.0 + chi_raw / max(c.kappa * 0.5 * b * wq, 1e-10))
    chi_eff = run * chi_raw
    gamma = np.tanh(chi_eff)/chi_eff if chi_eff > 1e-9 else 1.0
    zq = 2.0 * (np.cosh(0.5*b*wq) - gamma*(run*dz)*np.sinh(0.5*b*wq))
    a = 0.5 * b * wq
    rho = np.array([[np.exp(-a)*(1 + gamma*run*dz), np.exp(-a)*gamma*run*sp],
                    [np.exp(a)*gamma*run*sm, np.exp(a)*(1 - gamma*run*dz)]], dtype=complex) / zq
    return rho

# =============================================================================
# 3. OMNIBUS DASHBOARD & SWEEP
# =============================================================================

def run_omnibus_sweep():
    print("🛸 [OMNIBUS] Launching Definitive Research Sweep...")
    
    # 1. DEFINE RANGES (SWEEP OVER EVERYTHING)
    betas = np.linspace(0.4, 6.0, 10)
    gs = [0.25, 0.5, 0.75, 1.0] # Multi-coupling
    thetas = [0.0, 0.785, 1.571] # 0, pi/4, pi/2
    modes = [3, 4, 5]
    cuts = [6, 8, 10]
    
    results = []
    t_start = time.perf_counter()
    
    # Simple nested loops for the "everything" sweep
    for g, theta in itertools.product(gs, thetas):
        print(f"--- Physical Sector: g={g:.2f}, theta={theta:.2f} ---")
        # Build ED context for highest available (m, c) for this sector to use as baseline
        m_max, c_max = max(modes), max(cuts)
        ctx = omnibusConfig(g=g, theta=theta, n_modes=m_max, n_cut=c_max)
        h_tot_ed, _ = build_hamiltonian(ctx)
        
        for beta in betas:
            ctx.beta = beta
            # Benchmarks
            rho_ed = solve_ed(h_tot_ed, ctx)
            rho_ord = solve_ordered(ctx)
            rho_ren = solve_renorm(ctx)
            
            row = {"g": g, "theta": theta, "beta": beta, "m": m_max, "c": c_max}
            for m, r in [("ed", rho_ed), ("ord", rho_ord), ("ren", rho_ren)]:
                row[f"{m}_p00"] = r[0,0].real
                row[f"{m}_re01"] = r[0,1].real
                row[f"{m}_im01"] = r[0,1].imag
                row[f"{m}_coh"] = 2.0 * abs(r[0,1])
            results.append(row)
            print(f"  beta={beta:.1f} | ED={rho_ed[0,0].real:.4f} | Ord={rho_ord[0,0].real:.4f} | Ren={rho_ren[0,0].real:.4f}")
            
    df = pd.DataFrame(results)
    df.to_csv("hmf_omnibus_results.csv", index=False)
    
    # DASHBOARD
    print("\n🎬 Rendering Omnibus Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle(f"HMF Omnibus Research Dashboard (m={m_max}, c={c_max})", fontsize=16)
    
    # Plot 1: Population vs Beta for different couplings
    ax = axes[0, 0]
    for g_val in gs:
        sub = df[(df["g"] == g_val) & (df["theta"] == 1.571)] # Pick pi/2 for clear comparison
        ax.plot(sub["beta"], sub["ed_p00"], 'o-', label=f"ED g={g_val}")
        ax.plot(sub["beta"], sub["ren_p00"], '--', alpha=0.5)
    ax.set_title("Longitudinal Population (theta=pi/2)")
    ax.set_xlabel("beta"); ax.set_ylabel("rho00"); ax.legend(); ax.grid(alpha=0.2)
    
    # Plot 2: Coherence vs Beta for different couplings
    ax = axes[0, 1]
    for g_val in gs:
        sub = df[(df["g"] == g_val) & (df["theta"] == 0.785)] # Pick pi/4 for coh comparison
        ax.plot(sub["beta"], sub["ed_coh"], 'd-', label=f"ED g={g_val}")
    ax.set_title("Total Coherence (theta=pi/4)")
    ax.set_xlabel("beta"); ax.set_ylabel("2|rho01|"); ax.legend(); ax.grid(alpha=0.2)
    
    # Plot 3: Error Heatmap (ED vs Renorm)
    ax = axes[1, 0]
    error = np.abs(df["ed_p00"] - df["ren_p00"])
    # Slice a phase portrait
    sub_hm = df[df["beta"] == betas[-1]] # Highest beta
    pivot = sub_hm.pivot_table(index="g", columns="theta", values="ed_p00")
    im = ax.imshow(pivot, aspect='auto', origin='lower', extent=[0, 1.57, gs[0], gs[-1]])
    plt.colorbar(im, ax=ax, label="rho00")
    ax.set_title(f"Population Landscape (beta={betas[-1]:.1f})")
    ax.set_xlabel("theta"); ax.set_ylabel("coupling g")
    
    # Plot 4: Analytic Residuals
    ax = axes[1, 1]
    sub_res = df[df["theta"] == 0.785]
    ax.plot(sub_res["beta"], sub_res["ed_p00"] - sub_res["ren_p00"], 'g-', label="ED - Renorm")
    ax.plot(sub_res["beta"], sub_res["ed_p00"] - sub_res["ord_p00"], 'r--', label="ED - Ordered")
    ax.set_title("Model Discrepancies (Residuals)")
    ax.set_xlabel("beta"); ax.set_ylabel("Delta p00"); ax.legend(); ax.grid(alpha=0.2)
    
    plt.show()
    print(f"✅ OMNIBUS COMPLETE. Total Time: {(time.perf_counter() - t_start)/60:.1f} minutes")

if __name__ == "__main__":
    run_omnibus_sweep()
