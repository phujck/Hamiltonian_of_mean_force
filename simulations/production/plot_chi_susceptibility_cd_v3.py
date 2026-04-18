# -*- coding: utf-8 -*-
"""
plot_chi_susceptibility_cd_v3.py
--------------------------------
Regenerates the (c,d) susceptibility panel with v3 nomenclature:
  (c) d varphi_Q / dg
  (d) d r_Q / dg

Output:
  manuscript/figures/hmf_fig1_chi_theory_cd_v3.{pdf,png}
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import get_chi0, dphi_dg, dr_dg

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": True,
    "figure.dpi": 300,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
})

BETAS = [3.0, 4.0, 5.0]
COLORS = ["#e07b39", "#7b52ab", "#1a6ea8"]
LABELS = [r"$\beta\omega_q=3$", r"$\beta\omega_q=4$", r"$\beta\omega_q=5$"]

XI_MAX = 2.0
xi = np.linspace(0.01, XI_MAX, 450)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)

for ax in axes:
    ax.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
    ax.axvspan(0, 1, color="#e8f4f8", alpha=0.45, zorder=0)
    ax.axvspan(1, XI_MAX, color="#fff4e6", alpha=0.45, zorder=0)
    ax.set_xlim(0, XI_MAX)

# (c) d varphi_Q / dg
ax = axes[0]
for b, col, lab in zip(BETAS, COLORS, LABELS):
    chi0 = get_chi0(b)[0]
    gstar = 1.0 / np.sqrt(chi0)
    g_vals = xi * gstar
    y = dphi_dg(g_vals, b)
    ax.plot(xi, y, color=col, label=lab)
    idx = np.argmax(y)
    ax.plot(xi[idx], y[idx], "o", color=col, ms=5)

ax.set_xlabel(r"$g/g_\star(\beta)$")
ax.set_ylabel(r"$\partial_g\varphi_Q$")
ax.legend(loc="upper right", framealpha=0.9)
ax.text(0.03, 0.95, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

# (d) d r_Q / dg
ax = axes[1]
for b, col in zip(BETAS, COLORS):
    chi0 = get_chi0(b)[0]
    gstar = 1.0 / np.sqrt(chi0)
    g_vals = xi * gstar
    y = dr_dg(g_vals, b)
    ax.plot(xi, y, color=col)
    idx = np.argmax(np.abs(y))
    ax.plot(xi[idx], y[idx], "o", color=col, ms=5)

ax.set_xlabel(r"$g/g_\star(\beta)$")
ax.set_ylabel(r"$\partial_g r_Q$")
ax.text(0.03, 0.95, "(d)", transform=ax.transAxes, va="top", fontweight="bold")

out = FIGURES / "hmf_fig1_chi_theory_cd_v3.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
