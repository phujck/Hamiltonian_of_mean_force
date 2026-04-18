# -*- coding: utf-8 -*-
"""
plot_ultrastrong_dual.py
------------------------
Two-panel figure for the ultrastrong duality section:
(a) Direct influence-state flows r_S(g) in the Bloch disk
(b) The same flows after the duality map, showing convergence
    toward the Cresser-Anders coupling-basis endpoints
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import get_chi0, OMEGA_Q

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "figure.dpi": 240,
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
})

THETA = np.pi / 4
BETAS = [0.5, 1.0, 2.0, 3.0]
COLORS = ["#c66b5a", "#e0a882", "#7a6aac", "#3a78b0"]
LABELS = [
    r"$\beta\omega_q=0.5$",
    r"$\beta\omega_q=1.0$",
    r"$\beta\omega_q=2.0$",
    r"$\beta\omega_q=3.0$",
]
N_G = 1200


def unit_disk_boundary(n=500):
    x = np.linspace(0.0, 1.0, n)
    z = -np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
    return x, z


def influence_trajectory(beta):
    chi0, dz0, sx0 = get_chi0(beta, theta=THETA)
    gstar = 1.0 / np.sqrt(chi0)
    g = np.linspace(0.0, 6.0 * gstar, N_G)
    chi = g**2 * chi0
    r_s = np.tanh(chi)

    n_s = np.array([sx0, -dz0], dtype=float)
    n_s /= np.linalg.norm(n_s)

    mx = r_s * n_s[0]
    mz = r_s * n_s[1]

    i_star = int(np.argmin(np.abs(g - gstar)))
    return mx, mz, i_star


def dual_map(mx, mz, beta):
    t_ca = np.tanh(0.5 * beta * OMEGA_Q * abs(np.cos(THETA)))
    n_ca = np.array([abs(np.sin(THETA)), -abs(np.cos(THETA))], dtype=float)

    r = np.sqrt(mx**2 + mz**2)
    mx_v = r * t_ca * n_ca[0]
    mz_v = r * t_ca * n_ca[1]

    mx_ca = t_ca * n_ca[0]
    mz_ca = t_ca * n_ca[1]
    return mx_v, mz_v, mx_ca, mz_ca


# ---------------------------------------------------------------------------
# Combined two-panel figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.2), constrained_layout=True)

for ax in (ax1, ax2):
    xb, zb = unit_disk_boundary()
    ax.plot(xb, zb, color="black", lw=1.0)
    ax.fill_between(xb, zb, 0.02, color="#f8f8f8", zorder=0)
    ax.plot([0, 1], [0, 0], color="black", lw=0.5, ls=":", alpha=0.3)
    ax.plot([0, 0], [-1, 0], color="black", lw=0.5, ls=":", alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.02, 0.02)
    ax.set_aspect("equal", adjustable="box")

# Panel (a): direct flows
for beta, col, lab in zip(BETAS, COLORS, LABELS):
    mx, mz, i_star = influence_trajectory(beta)
    ax1.plot(mx, mz, color=col, label=lab)
    ax1.plot(mx[0], mz[0], "o", color=col, ms=4.5,
             markeredgecolor="white", markeredgewidth=0.6)
    ax1.plot(mx[i_star], mz[i_star], "*", color=col, ms=8.0,
             markeredgecolor="white", markeredgewidth=0.5)
    ax1.plot(mx[-1], mz[-1], "s", color=col, ms=4.5,
             markeredgecolor="white", markeredgewidth=0.6)

ax1.plot([], [], "ko", ms=4.0, label=r"$g=0$")
ax1.plot([], [], "k*", ms=7.0, label=r"$g=g_\star$")
ax1.plot([], [], "ks", ms=4.0, label=r"$g\to\infty$")
ax1.legend(loc="upper right", ncol=2, framealpha=0.92,
           handlelength=1.3, columnspacing=0.5)
ax1.set_xlabel(r"$m_{S,\perp}$")
ax1.set_ylabel(r"$m_{S,z}$")
ax1.set_title(r"\textbf{(a)} Direct chart", loc="left")

# Panel (b): dual-mapped flows, normalized by t_CA for data collapse
# All trajectories collapse to r(g) along the coupling axis n_ca
n_ca = np.array([abs(np.sin(THETA)), -abs(np.cos(THETA))], dtype=float)
# Draw the coupling direction ray
ax2.plot([0, n_ca[0]], [0, n_ca[1]], color="gray", lw=0.8, ls="--", alpha=0.5)
ax2.text(n_ca[0]*0.55 + 0.06, n_ca[1]*0.55 + 0.06,
         r"$\hat{\mathbf f}$", fontsize=9, color="gray", ha="left")

LSTYLES = ["-", "--", "-.", ":"]
for beta, col, lab, ls in zip(BETAS, COLORS, LABELS, LSTYLES):
    mx, mz, i_star = influence_trajectory(beta)
    r = np.sqrt(mx**2 + mz**2)  # tanh(g^2 chi0), 0 to 1
    # Dual: normalized to unit endpoint on coupling axis
    mx_v = r * n_ca[0]
    mz_v = r * n_ca[1]

    ax2.plot(mx_v, mz_v, color=col, label=lab, ls=ls, lw=1.8)
    ax2.plot(mx_v[0], mz_v[0], "o", color=col, ms=4.5,
             markeredgecolor="white", markeredgewidth=0.6)
    ax2.plot(mx_v[i_star], mz_v[i_star], "*", color=col, ms=8.0,
             markeredgecolor="white", markeredgewidth=0.5)
    ax2.plot(mx_v[-1], mz_v[-1], "s", color=col, ms=4.5,
             markeredgecolor="white", markeredgewidth=0.6)

# CA endpoint at the unit-normalised direction
ax2.plot(n_ca[0], n_ca[1], "D", color="black", ms=6.0,
         markeredgecolor="black", markeredgewidth=0.5, zorder=10)
ax2.text(n_ca[0]+0.05, n_ca[1]+0.05,
         r"CA", fontsize=8, color="black", fontweight="bold")

ax2.legend(loc="upper right", ncol=2, framealpha=0.92,
           handlelength=1.3, columnspacing=0.5)
ax2.set_xlabel(r"$m_{S,\perp}^{\vee}/t_{\rm CA}$")
ax2.set_ylabel(r"$m_{S,z}^{\vee}/t_{\rm CA}$")
ax2.set_title(r"\textbf{(b)} Dual chart (normalised)", loc="left")

out = FIGURES / "hmf_ultrastrong_dual.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")

