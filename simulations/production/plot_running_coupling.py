# -*- coding: utf-8 -*-
"""
plot_running_coupling.py  --  Figure B: Running coupling and temperature portrait
==================================================================================
Three-panel publication figure illustrating how the bath susceptibility chi_0(beta)
encodes a temperature-dependent running coupling, and how this drives the qubit
state across the weak-to-strong coupling crossover as the system cools.

  Panel (a): chi_0(beta) [bath susceptibility] and chi(beta, g) = g^2 * chi_0(beta)
             for three coupling values.  The chi=1 crossover is marked, along with
             the crossover temperature beta_star(g) for each g.

  Panel (b): Bloch angle phi(beta) = arctan2(mx, -mz) for the same three couplings.
             phi = 0 is the bare axis; coupling kicks phi away from zero near beta_star.

  Panel (c): Bloch radius r(beta) for the same three couplings, with the bare
             thermal value r_bare = tanh(beta*omega_q/2) overlaid.
             Crossing r_bare from above = coupling purifies; below = degrades.

Bath model and coupling: Ohmic, theta=pi/4.  All analytic, no simulation data.

Output: manuscript/figures/hmf_running_coupling.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import bloch_ohmic, get_chi0, OMEGA_Q

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Publication-quality rcParams ─────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 8,
    "axes.labelsize"    : 9,
    "axes.titlesize"    : 8,
    "legend.fontsize"   : 7,
    "xtick.labelsize"   : 7,
    "ytick.labelsize"   : 7,
    "text.usetex"       : True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "figure.dpi"        : 200,
    "lines.linewidth"   : 1.2,
    "axes.linewidth"    : 0.7,
    "xtick.major.width" : 0.7,
    "ytick.major.width" : 0.7,
    "xtick.major.size"  : 3.0,
    "ytick.major.size"  : 3.0,
})

# ── Coupling values and palette ───────────────────────────────────────────────
# Three g values straddling the crossover region
G_VALUES    = [0.20, 0.50, 0.90]   # g / omega_q
G_COLORS    = ["#1a9641", "#984ea3", "#d95f02"]
G_LINESTY   = ["-",       "--",      "-."     ]
G_LABELS    = [r"$g=0.20\,\omega_q$", r"$g=0.50\,\omega_q$", r"$g=0.90\,\omega_q$"]

# beta grid (fine enough for smooth curves)
BETA_ARR    = np.linspace(0.4, 9.0, 280)

# ── Pre-compute chi0(beta) on the grid ───────────────────────────────────────
print("Computing chi0 on beta grid...")
chi0_arr = np.array([get_chi0(b)[0] for b in BETA_ARR])
print(f"  chi0 range: [{chi0_arr.min():.4f}, {chi0_arr.max():.4f}]")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7),
                         gridspec_kw=dict(wspace=0.45, left=0.08, right=0.97,
                                         top=0.91, bottom=0.16))

# ── Shared weak/strong shading helper ────────────────────────────────────────
def shade_weak_strong(ax, beta_star_list, ybot=None, ytop=None):
    """Lightly shade axes around each beta_star: no hard boundaries since
    different g values have different crossovers; just add the chi=1 reference."""
    pass  # handled per-axis below


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): chi(beta) for each g, plus chi_0(beta) baseline
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Bath susceptibility chi_0(beta) as baseline (g-independent shape)
ax.plot(BETA_ARR * OMEGA_Q, chi0_arr,
        color="black", lw=0.9, ls=":", alpha=0.60,
        label=r"$\chi_0(\beta)$ (norm.)", zorder=2)

beta_star_all = []  # collect for annotations
for g, col, ls, lab in zip(G_VALUES, G_COLORS, G_LINESTY, G_LABELS):
    chi_arr = g**2 * chi0_arr

    ax.plot(BETA_ARR * OMEGA_Q, chi_arr, color=col, ls=ls, lw=1.3,
            label=lab, zorder=3)

    # beta_star: interpolate where chi = 1
    try:
        idx = np.where(np.diff(np.sign(chi_arr - 1.0)))[0][0]
        beta_star = float(BETA_ARR[idx] +
                          (1.0 - chi_arr[idx]) /
                          (chi_arr[idx+1] - chi_arr[idx]) *
                          (BETA_ARR[idx+1] - BETA_ARR[idx]))
    except IndexError:
        beta_star = np.nan
    beta_star_all.append(beta_star)

    if np.isfinite(beta_star):
        ax.plot(beta_star * OMEGA_Q, 1.0, "o", color=col, ms=5,
                markeredgewidth=0.6, markeredgecolor="white", zorder=5)

# chi = 1 reference
ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5, zorder=1)
ax.text(0.42 * OMEGA_Q, 1.04, r"$\chi = 1$", fontsize=6.5, color="#444444",
        va="bottom")

# Weak / strong regions: shade based on g=0.50 crossover
bs_mid = beta_star_all[1]  # g=0.50
if np.isfinite(bs_mid):
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0.3,
               bs_mid * OMEGA_Q, color="#ddeeff", alpha=0.35, zorder=0)
    ax.axvspan(bs_mid * OMEGA_Q, 9.0 * OMEGA_Q,
               color="#fff3e0", alpha=0.35, zorder=0)

ax.set_xlabel(r"$\beta\omega_q$")
ax.set_ylabel(r"$\chi = g^2\chi_0(\beta)$")
ax.set_xlim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.set_ylim(0, min(3.5, 1.15 * (G_VALUES[-1]**2 * chi0_arr.max())))
ax.legend(fontsize=5.8, loc="upper left", framealpha=0.90,
          handlelength=1.5, borderpad=0.4, labelspacing=0.25)
ax.text(0.04, 0.97, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)

# Annotate "cooling drives crossover"
ax.annotate("cooling",
            xy=(0.89, 0.52), xytext=(0.60, 0.35),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=5.8, color="#555555",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7,
                            mutation_scale=5))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Bloch angle phi(beta)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

# Bare axis reference phi=0
ax.axhline(0.0, color="black", ls=":", lw=0.7, alpha=0.40, zorder=1)

for g, col, ls, lab, beta_star in zip(G_VALUES, G_COLORS, G_LINESTY, G_LABELS,
                                       beta_star_all):
    # Compute phi(beta) by calling bloch_ohmic at each beta point
    phis = []
    for beta in BETA_ARR:
        phi_val, _, _, _ = bloch_ohmic(np.array([g]), beta)
        phis.append(float(phi_val[0]))
    phi_arr = np.degrees(np.array(phis))   # convert to degrees

    ax.plot(BETA_ARR * OMEGA_Q, phi_arr, color=col, ls=ls, lw=1.3,
            label=lab, zorder=3)

    # beta_star vertical marker
    if np.isfinite(beta_star):
        ax.axvline(beta_star * OMEGA_Q, color=col, ls=":", lw=0.7, alpha=0.55)
        ax.plot(beta_star * OMEGA_Q,
                np.degrees(float(bloch_ohmic(np.array([g]), beta_star)[0][0])),
                "o", color=col, ms=5,
                markeredgewidth=0.6, markeredgecolor="white", zorder=5)

ax.set_xlabel(r"$\beta\omega_q$")
ax.set_ylabel(r"$\varphi$ (degrees)")
ax.set_xlim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.legend(fontsize=5.8, loc="upper left", framealpha=0.90,
          handlelength=1.5, borderpad=0.4, labelspacing=0.25)
ax.text(0.04, 0.97, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)
# Label: "bare thermal: phi=0"
ax.text(0.97, 0.06, r"bare: $\varphi=0$", fontsize=5.8, color="#555555",
        ha="right", va="bottom", transform=ax.transAxes)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (c): Bloch radius r(beta)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[2]

# Bare thermal purity r_bare = tanh(a)
r_bare_arr = np.tanh(BETA_ARR * OMEGA_Q / 2.0)
ax.plot(BETA_ARR * OMEGA_Q, r_bare_arr,
        color="black", ls="--", lw=1.0, alpha=0.55,
        label=r"$r_{\rm bare} = \tanh a$", zorder=2)

for g, col, ls, lab, beta_star in zip(G_VALUES, G_COLORS, G_LINESTY, G_LABELS,
                                       beta_star_all):
    rs = []
    for beta in BETA_ARR:
        _, r_val, _, _ = bloch_ohmic(np.array([g]), beta)
        rs.append(float(r_val[0]))
    r_arr = np.array(rs)

    ax.plot(BETA_ARR * OMEGA_Q, r_arr, color=col, ls=ls, lw=1.3,
            label=lab, zorder=3)

    # beta_star marker
    if np.isfinite(beta_star):
        ax.axvline(beta_star * OMEGA_Q, color=col, ls=":", lw=0.7, alpha=0.55)

# Annotate purity enhancement at high T (small beta)
# At small beta all colored curves are ABOVE r_bare; point to the gap near left edge
ax.annotate(r"$r > r_{\rm bare}$: bath purifies",
            xy=(0.75, 0.48), xytext=(2.4, 0.37),
            fontsize=5.5, color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.6,
                            mutation_scale=4),
            ha="center")

ax.set_xlabel(r"$\beta\omega_q$")
ax.set_ylabel(r"$r = |\mathbf{v}|$")
ax.set_xlim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.set_ylim(0, 1.03)
ax.legend(fontsize=5.8, loc="upper left", framealpha=0.90,
          handlelength=1.5, borderpad=0.4, labelspacing=0.25)
ax.text(0.04, 0.97, "(c)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)


# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_running_coupling.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
