# -*- coding: utf-8 -*-
"""
plot_bloch_disk_portrait.py
---------------------------
v3 limits figures:
1) hmf_bloch_disk_portrait: coupling-driven g-sweeps of the symmetrised
   influence state rho_S in the coupling plane.
2) hmf_bloch_disk_dual_ca: the same sweeps after a representative dual map
   used to compare with the Cresser-Anders ultrastrong chart.
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
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
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
    # Cresser-Anders radius in the coupling chart
    t_ca = np.tanh(0.5 * beta * OMEGA_Q * abs(np.cos(THETA)))
    n_ca = np.array([abs(np.sin(THETA)), -abs(np.cos(THETA))], dtype=float)

    # map preserves sweep parameter and reorients to coupling chart
    r = np.sqrt(mx**2 + mz**2)
    mx_v = r * t_ca * n_ca[0]
    mz_v = r * t_ca * n_ca[1]

    mx_ca = t_ca * n_ca[0]
    mz_ca = t_ca * n_ca[1]
    return mx_v, mz_v, mx_ca, mz_ca


# ---------------------------------------------------------------------------
# Figure A: direct influence flows
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
xb, zb = unit_disk_boundary()
ax.plot(xb, zb, color="black", lw=1.0)
ax.fill_between(xb, zb, 0.02, color="#f8f8f8", zorder=0)
ax.plot([0, 1], [0, 0], color="black", lw=0.5, ls=":", alpha=0.3)
ax.plot([0, 0], [-1, 0], color="black", lw=0.5, ls=":", alpha=0.3)

for beta, col, lab in zip(BETAS, COLORS, LABELS):
    mx, mz, i_star = influence_trajectory(beta)
    ax.plot(mx, mz, color=col, label=lab)
    ax.plot(mx[0], mz[0], "o", color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(mx[i_star], mz[i_star], "*", color=col, ms=8.0, markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(mx[-1], mz[-1], "s", color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.6)

ax.plot([], [], "ko", ms=4.0, label=r"$g=0$")
ax.plot([], [], "k*", ms=7.0, label=r"$g=g_\star$")
ax.plot([], [], "ks", ms=4.0, label=r"$g\to\infty$")
ax.legend(loc="upper right", ncol=2, framealpha=0.9, handlelength=1.5, columnspacing=0.6)

ax.set_xlim(0, 1.0)
ax.set_ylim(-1.02, 0.0)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$m_{S,\perp}$")
ax.set_ylabel(r"$m_{S,z}$")

out = FIGURES / "hmf_bloch_disk_portrait.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Figure B: dual-mapped flows toward Cresser-Anders chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
xb, zb = unit_disk_boundary()
ax.plot(xb, zb, color="black", lw=1.0)
ax.fill_between(xb, zb, 0.02, color="#f8f8f8", zorder=0)
ax.plot([0, 1], [0, 0], color="black", lw=0.5, ls=":", alpha=0.3)
ax.plot([0, 0], [-1, 0], color="black", lw=0.5, ls=":", alpha=0.3)

for beta, col, lab in zip(BETAS, COLORS, LABELS):
    mx, mz, i_star = influence_trajectory(beta)
    mx_v, mz_v, mx_ca, mz_ca = dual_map(mx, mz, beta)

    ax.plot(mx_v, mz_v, color=col, label=lab)
    ax.plot(mx_v[0], mz_v[0], "o", color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(mx_v[i_star], mz_v[i_star], "*", color=col, ms=8.0, markeredgecolor="white", markeredgewidth=0.5)
    ax.plot(mx_v[-1], mz_v[-1], "s", color=col, ms=4.5, markeredgecolor="white", markeredgewidth=0.6)
    ax.plot(mx_ca, mz_ca, "D", color=col, ms=4.8, markeredgecolor="black", markeredgewidth=0.4)

ax.plot([], [], "kD", ms=4.5, label=r"CA endpoint")
ax.legend(loc="upper right", framealpha=0.9)

ax.set_xlim(0, 1.0)
ax.set_ylim(-1.02, 0.0)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$m_{S,\perp}^{\vee}$")
ax.set_ylabel(r"$m_{S,z}^{\vee}$")

out2 = FIGURES / "hmf_bloch_disk_dual_ca.png"
fig.savefig(out2, bbox_inches="tight")
fig.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out2}")
print(f"Saved -> {out2.with_suffix('.pdf')}")
