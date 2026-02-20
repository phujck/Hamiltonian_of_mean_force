# -*- coding: utf-8 -*-
"""
fig2_ed_vs_analytic.py  –  4-Panel Crossover Landscape
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import sys

DATA    = Path(__file__).parent / "data" / "sweeps_v50.csv"
FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": True, "figure.dpi": 200, "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
})

# Crossover calculation imports
sys.path.append(str(Path(__file__).parents[2] / "simulations" / "src"))
from hmf_component_normalized_compare_codex_v5 import _base_channels, BenchmarkConfig

C_ED   = "#2ca02c"   # green
C_DISC = "#1f77b4"   # blue
C_CONT = "#ff7f0e"   # orange

df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows; sweeps: {df.sweep.unique().tolist()}")

# Function to get star values
def get_stars():
    # g_star: beta=2, theta=pi/4, omega_q=1
    cfg_g = BenchmarkConfig(beta=2.0, omega_q=1.0, theta=np.pi/4, n_modes=40, n_cut=5, 
                            omega_min=0.0, omega_max=1.8, q_strength=5.0, tau_c=0.5)
    sp0_g, sm0_g, dz0_g = _base_channels(cfg_g)
    chi0_g = np.sqrt(max(dz0_g**2 + sp0_g*sm0_g, 0))
    g_star = 1.0/np.sqrt(chi0_g) if chi0_g > 0 else 1.0

    # beta_star: g=0.5, theta=pi/2
    g = 0.5; theta = np.pi/2
    target_chi0 = 1.0/(g**2)
    b_range = np.linspace(0.1, 50, 500)
    chi0_vals = []
    for b in b_range:
        cfg = BenchmarkConfig(beta=float(b), omega_q=1.0, theta=theta, n_modes=40, n_cut=1, 
                              omega_min=0.0, omega_max=1.8, q_strength=5.0, tau_c=0.5)
        sp0, sm0, dz0 = _base_channels(cfg)
        chi0_vals.append(np.sqrt(max(dz0**2 + sp0*sm0, 0)))
    beta_star = np.interp(target_chi0, chi0_vals, b_range)
    return g_star, beta_star

G_STAR, B_STAR = get_stars()
print(f"Calculated crossover scales: g_star={G_STAR:.3f}, beta_star={B_STAR:.3f}")

fig, axes = plt.subplots(2, 1, figsize=(3.2, 5.2), sharex=True)
fig.subplots_adjust(hspace=0.2)

SWEEPS = ["coupling", "temperature"]
X_STARS = [G_STAR, B_STAR]
X_LABS = [r"$g/g_\star$", r"$\beta/\beta_\star$"]
FIX_LABS = [r"$\beta\omega_q=2, \theta=\pi/4$", r"$g/\omega_q=0.5, \theta=\pi/2$"]
THE_STARS = [np.pi/2.56, np.pi/2]

for row, (sw, x_star, x_lab, fix_lab, theta) in enumerate(zip(SWEEPS, X_STARS, X_LABS, FIX_LABS, THE_STARS)):
    sub = df[df.sweep == sw].sort_values("param").copy()
    x = sub["param"].values / x_star
    
    ax = axes[row]
    p_ed   = sub["ed_p11"].values
    p_disc = sub["analytic_disc_p11"].values
    p_cont = sub["analytic_cont_p11"].values
    
    ax.plot(x, p_ed,   color=C_ED,   lw=1.5, label="ED", zorder=3)
    ax.plot(x, p_disc, color=C_DISC, lw=1.2, ls="--", label=r"$\mathcal{K}_{\mathrm{disc}}$")
    ax.plot(x, p_cont, color=C_CONT, lw=1.0, ls="--",  label=r"$\mathcal{K}$")
    
    # US limit
    p_inf = np.cos(theta/2)**2
    ax.axhline(p_inf, color="k", lw=0.8, ls=(0, (5, 1)), alpha=0.4, label="US limit")
    
    # Shading
    ax.axvspan(0.1, 1.0, color="#e8f4f8", alpha=0.5, zorder=0)
    ax.axvspan(1.0, 10.0, color="#fff4e6", alpha=0.5, zorder=0)
    ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlim(0.1, 10.0)
    ax.set_ylabel(r"$p_{11}$")
    ax.set_xlabel(x_lab) # Add to both
    ax.text(0.05, 0.9, ["(a)", "(b)"][row], transform=ax.transAxes, fontweight="bold")
    
    # Weak/Strong labels
    ax.text(0.15, 0.08, "Weak", transform=ax.transAxes, fontsize=7, color="#2c7fb8", fontweight="bold")
    ax.text(0.65, 0.08, "Strong", transform=ax.transAxes, fontsize=7, color="#d95f02", fontweight="bold")

    if row == 0:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

plt.savefig(FIGURES / "hmf_fig2_ed_vs_analytic.png", bbox_inches="tight", dpi=250)
plt.savefig(FIGURES / "hmf_fig2_ed_vs_analytic.pdf", bbox_inches="tight")
print("Saved 4-panel figure.")
