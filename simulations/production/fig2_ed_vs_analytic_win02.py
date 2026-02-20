# -*- coding: utf-8 -*-
"""
fig2_ed_vs_analytic_win02.py  –  Comparison Version [win0.2]
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import sys

DATA    = Path(__file__).parent / "data" / "sweeps_v49_win02.csv"
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
C_BARE = "#aaaaaa"   # grey

df = pd.read_csv(DATA)

SWEEP_CONFIG = [
    ("coupling",    "g",    r"$g/\omega_q$",    "beta",  2.0, "(a)", r"$\beta\omega_q=2$"),
    ("temperature", "beta", r"$\beta\omega_q$", "g",     0.5, "(b)", r"$g/\omega_q=0.5$"),
]

fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
fig.subplots_adjust(wspace=0.35)

for ax, (sw, px, xl, fix_col, fix_val, lab, sw_lab) in zip(axes, SWEEP_CONFIG):
    sub = df[df.sweep == sw].copy()
    sub = sub.sort_values(px)
    x   = sub[px].values

    p_ed   = sub["ed_p11"].values
    p_disc = sub["analytic_disc_p11"].values
    p_cont = sub["analytic_cont_p11"].values
    # Bare Gibbs: g=0 or beta=min
    bare = sub[sub.g == 0]["ed_p11"].iloc[0] if sw == "coupling" else sub[sub.beta == sub.beta.min()]["ed_p11"].iloc[0]

    ax.plot(x, p_ed,   color=C_ED,   lw=1.5, label="ED", zorder=3)
    ax.plot(x, p_disc, color=C_DISC, lw=1.2, ls="--", label="Analytic (disc.)")
    ax.plot(x, p_cont, color=C_CONT, lw=1.0, ls=":",  label="Analytic (cont.)")
    ax.axhline(bare, color=C_BARE, lw=0.8, ls="-.", alpha=0.7, label="Bare Gibbs")

    # Add crossover Marker
    if sw == "coupling":
        theta = np.pi/4
        beta  = 2.0
        cfg = BenchmarkConfig(beta=beta, omega_q=1.0, theta=theta, n_modes=40, n_cut=1, 
                              omega_min=0.2, omega_max=1.8, q_strength=5.0, tau_c=0.5)
        sp0, sm0, dz0 = _base_channels(cfg)
        chi0 = np.sqrt(max(dz0**2 + sp0*sm0, 0))
        x_star = 1.0/np.sqrt(chi0) if chi0 > 0 else np.nan
        star_label = r"$g_\star$"
        p_inf = 1-0.5*np.tanh(np.cos(theta))
    else:
        theta = np.pi/2
        g = 0.5
        target_chi0 = 1.0/(g**2)
        b_range = np.linspace(0.2, 20, 200)
        chi0_vals = []
        for b in b_range:
            cfg = BenchmarkConfig(beta=float(b), omega_q=1.0, theta=theta, n_modes=40, n_cut=1, 
                                  omega_min=0.2, omega_max=1.8, q_strength=5.0, tau_c=0.5)
            sp0, sm0, dz0 = _base_channels(cfg)
            chi0_vals.append(np.sqrt(max(dz0**2 + sp0*sm0, 0)))
        x_star = np.interp(target_chi0, chi0_vals, b_range)
        star_label = r"$\beta_\star$"
        p_inf = np.cos(theta/2)**2

    # Add Ultrastrong limit line
    ax.axhline(p_inf, color="k", lw=0.8, ls=(0, (5, 1)), alpha=0.5, label="Ultrastrong limit")

    if not np.isnan(x_star) and x.min() < x_star < x.max():
        ax.axvline(x_star, color="k", ls=":", lw=1.0, alpha=0.6)
        ax.text(x_star, ax.get_ylim()[0] + 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 
                star_label, ha="right", fontsize=8, color="k", alpha=0.7)

    ax.set_xlabel(xl)
    ax.set_ylabel(r"$p_{00}$")
    ax.set_xlim(x.min(), x.max())
    ax.text(0.05, 0.93, lab, transform=ax.transAxes, fontsize=9, fontweight="bold")
    if ax is axes[0]:
        ax.legend(fontsize=7, framealpha=0.85, loc="lower left")

    ax.text(0.97, 0.05, sw_lab, transform=ax.transAxes, fontsize=7.5,
            ha="right", color="k", alpha=0.7)

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig2_ed_vs_analytic_win02{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")
plt.close(fig)
