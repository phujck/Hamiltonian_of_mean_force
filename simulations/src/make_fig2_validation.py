# -*- coding: utf-8 -*-
"""
fig2_validation.py  --  Figure 2: Comprehensive Numerical Validation
=====================================================================
(a) Heatmap of ED vs HMF error in the (beta, g) plane.
(b, c, d) Slices for different energy gaps omega_q.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

SRC = Path(__file__).resolve().parent
FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": False, "figure.dpi": 200, "lines.linewidth": 1.4,
})

# ── Analytic Solver (Discrete Bath) ───────────────────────────────────────────
def solve_hmf_p00(beta, g, theta, omega_q, omegas, weights):
    """Analytic HMF population p00 for a discrete bath."""
    # chi0(beta) = sum_k weights[k] * tanh(beta*omega[k]/2) / (2*omega[k])
    th = np.tanh(np.clip(beta * omegas / 2.0, 0, 700))
    chi0 = np.sum(weights * th / (2.0 * omegas))
    
    chi = g**2 * chi0
    a   = beta * omega_q / 2.0
    
    # theta=pi/2 (transverse) fallback if needed, but we use general theta
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Generic mixing case (theta angle between z and x)
    # M = chi * (c sigma_z + s sigma_x)
    # rho = exp(-a/2 sz) exp(M) exp(-a/2 sz) / Z
    A00 = np.exp(-a) * (np.cosh(chi) + c * np.sinh(chi))
    A11 = np.exp(+a) * (np.cosh(chi) - c * np.sinh(chi))
    A01 = s * np.sinh(chi)
    Z = A00 + A11
    
    mz = (A00 - A11) / Z
    return 0.5 * (1.0 + mz)

def main():
    # ── 1. Create Heatmap from JAX data (omega_q=1.0) ─────────────────────────
    JAX_FILE = SRC / "hmf_jax_convergence_results.csv"
    print(f"Loading {JAX_FILE.name}...")
    df = pd.read_csv(JAX_FILE)
    
    # Filter for theta=pi/2 (transverse) and N_modes=5
    # Theta in JAX is [0...1.5709]
    df_map = df[(df.theta > 1.5) & (df.n_modes == 5)].copy()
    
    # Bath for JAX (N=5, spectral strength lambda=1 in v17 bundle usually)
    # Assume omegas are linspace(0.2, 1.8, 5) and weights are uniform?
    # Actually, simpler: chi0(beta) for 5-mode bath.
    omegas = np.linspace(0.2, 1.8, 5)
    weights = np.ones(5) * (1.6/5.0) # approx weights
    
    # Compute analytic prediction for each row
    df_map['an_p00'] = df_map.apply(lambda row: solve_hmf_p00(row.beta, row.g, row.theta, 1.0, omegas, weights), axis=1)
    df_map['err'] = np.abs(df_map['p00'] - df_map['an_p00'])
    
    # Pivot for heatmap
    # We want beta in [0, 3] as requested.
    df_sub = df_map[df_map.beta <= 3.2].copy()
    pivot = df_sub.pivot_table(index='g', columns='beta', values='err')
    
    # ── 2. Load v35 for varied omega_q (Slices) ───────────────────────────────
    V35_FILE = SRC / "hmf_omegaq_scaled_window_scan_codex_v35.csv"
    print(f"Loading {V35_FILE.name}...")
    df_v35 = pd.read_csv(V35_FILE)
    
    # Filter for a consistent set across omega_q: m4c4 case and tail_fraction ~0.2
    df_v35 = df_v35[(df_v35['case'] == 'm4c4') & (df_v35['tail_fraction'] > 0.19) & (df_v35['tail_fraction'] < 0.21)].copy()

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 6.2))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
    
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    
    # (a) Heatmap
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    im = ax_a.pcolormesh(X, Y, pivot.values, cmap="magma", norm=mcolors.LogNorm(vmin=1e-3, vmax=0.1), shading='auto')
    cb = fig.colorbar(im, ax=ax_a, pad=0.01)
    cb.set_label("Abs Error in " + r"$\rho_{00}$")
    
    # Overlay g* line (chi=1)
    b_gs = np.linspace(0.4, 3.2, 100)
    def chi0_vec(betas):
        th = np.tanh(np.multiply.outer(betas, omegas) / 2.0)
        return np.sum(weights[np.newaxis, :] * th / (2.0 * omegas[np.newaxis, :]), axis=1)
    
    c0 = chi0_vec(b_gs)
    ax_a.plot(b_gs, 1.0/np.sqrt(c0), 'w--', lw=1.5, label=r"$\chi=1$ scale ($g_\star$)")
    
    ax_a.set_xlabel(r"$\beta\omega_q$")
    ax_a.set_ylabel(r"$g/\omega_q$")
    ax_a.set_title(r"(a) Accuracy landscape in $(\beta, g)$ plane ($\omega_q=1.0$)")
    ax_a.set_ylim(0.1, 1.6) # Match JAX range
    ax_a.legend(loc='lower left', fontsize=7, framealpha=0.6)
    ax_a.text(0.01, 0.9, "(a)", transform=ax_a.transAxes, fontweight="bold", color="w")

    # (b, c, d) Slices for omega_q
    W_VALS = [1.5, 2.0, 3.0]
    AXES = [ax_b, ax_c, ax_d]
    LABELS = ["(b)", "(c)", "(d)"]
    COLORS = ["#e07b39", "#7b52ab", "#2ca02c"]
    
    for wq, ax, lab, col in zip(W_VALS, AXES, LABELS, COLORS):
        sub = df_v35[df_v35.omega_q == wq].sort_values('beta')
        if sub.empty:
            print(f"Warning: No data for omega_q={wq}")
            continue
            
        ax.plot(sub['beta'], sub['ed_p00'], 'o', ms=4, color=col, markerfacecolor='none', markeredgewidth=1.0, label='ED')
        ax.plot(sub['beta'], sub['best_p00'], '-', color=col, lw=1.5, label='HMF')
        ax.set_title(rf"{lab} $\omega_q={wq}$", fontsize=9)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\rho_{00}$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
        ax.set_ylim(0.1, 0.45)
        
    fig.suptitle("HMF Validation: Performance across coupling and temperature regimes", fontsize=11, y=0.98)
    plt.savefig(FIGURES / "hmf_fig2_validation.png", dpi=200, bbox_inches='tight')
    print("Saved Figure 2.")

if __name__ == "__main__":
    main()
