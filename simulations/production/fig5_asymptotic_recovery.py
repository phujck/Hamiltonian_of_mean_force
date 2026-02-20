# -*- coding: utf-8 -*-
"""
fig5_asymptotic_recovery.py  –  Figure 5: US recovery with inverse parameters
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
C_BARE = "#aaaaaa"   # grey

df = pd.read_csv(DATA)

# Find g_star for the coupling sweep
theta_g = np.pi/4; beta_g = 2.0
cfg_g = BenchmarkConfig(beta=beta_g, omega_q=1.0, theta=theta_g, n_modes=40, n_cut=1, omega_min=0.0, omega_max=1.8, q_strength=5.0, tau_c=0.5)
sp0_g, sm0_g, dz0_g = _base_channels(cfg_g)
g_star = 1.0/np.sqrt(np.sqrt(max(dz0_g**2 + sp0_g*sm0_g, 0)))

# For beta sweep, beta_star is harder; let's just use 1/beta
# Actually, let's plot vs g_star / g and 1/beta

fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
fig.subplots_adjust(wspace=0.35)

# Panel (a): US recovery in coupling
ax = axes[0]
sub = df[df.sweep == "coupling"].sort_values("g")
x = g_star / sub["g"].values 
ax.plot(x, sub["ed_p11"], color=C_ED, lw=1.5, label="ED", zorder=3)
ax.plot(x, sub["analytic_disc_p11"], color=C_DISC, lw=1.2, ls="--", label="Analytic (disc.)")
ax.plot(x, sub["analytic_cont_p11"], color=C_CONT, lw=1.0, ls=":",  label="Analytic (cont.)")
ax.axhline(np.cos(theta_g/2)**2, color="k", lw=0.8, ls=(0, (5, 1)), alpha=0.5, label="US limit")
ax.set_xlabel(r"$g_\star / g$")
ax.set_ylabel(r"$p_{00}$")
ax.set_xlim(0, 2)
ax.axvline(1.0, color="k", ls=":", alpha=0.3)
ax.text(0.05, 0.93, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

# Panel (b): US recovery in temperature (cold limit)
ax = axes[1]
sub = df[df.sweep == "temperature"].sort_values("beta")
x = 1.0 / sub["beta"].values
ax.plot(x, sub["ed_p11"], color=C_ED, lw=1.5, label="ED", zorder=3)
ax.plot(x, sub["analytic_disc_p11"], color=C_DISC, lw=1.2, ls="--", label="Analytic (disc.)")
ax.plot(x, sub["analytic_cont_p11"], color=C_CONT, lw=1.0, ls=":",  label="Analytic (cont.)")
ax.axhline(np.cos(np.pi/4)**2, color="k", lw=0.8, ls=(0, (5, 1)), alpha=0.5) 
ax.set_xlabel(r"$1/\beta\omega_q$")
ax.set_ylabel(r"$p_{00}$")
ax.set_xlim(0, 1)
ax.text(0.05, 0.93, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
ax.legend(fontsize=6, loc="lower right")

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig5_us_recovery{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")
plt.close(fig)
