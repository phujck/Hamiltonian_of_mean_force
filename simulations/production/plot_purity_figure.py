# -*- coding: utf-8 -*-
"""
plot_purity_figure.py  --  Figure 6: Purity of the mean-force state
====================================================================
Consolidated two-panel figure on bath-induced purification.

  Panel (a): Bloch radius r(g/g_star, beta) for four temperatures.
             Each curve lies strictly above its bare thermal reference
             r_bare = tanh(beta*omega_q/2) (dotted horizontal), confirming
             the universal purification lemma.

  Panel (b): Purity excess Delta(r^2) = r^2(g,beta) - tanh^2(a) in the
             full (g, beta) plane (heatmap). Enhancement peaks at large g
             and high T; vanishes at low T (bare state already near-pure).
             White dashed contour marks the chi=1 crossover locus.

Bath model and coupling: Ohmic J(omega) = alpha*omega*exp(-omega/omega_c),
theta = pi/4. All analytic, no simulation data.

Output: manuscript/figures/hmf_purity_figure.{pdf,png}
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

# ── Publication-quality rcParams ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"        : "serif",
    "font.size"          : 8,
    "axes.labelsize"     : 9,
    "axes.titlesize"     : 8,
    "legend.fontsize"    : 7,
    "xtick.labelsize"    : 7,
    "ytick.labelsize"    : 7,
    "text.usetex"        : True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "figure.dpi"         : 200,
    "lines.linewidth"    : 1.2,
    "axes.linewidth"     : 0.7,
    "xtick.major.width"  : 0.7,
    "ytick.major.width"  : 0.7,
    "xtick.major.size"   : 3.0,
    "ytick.major.size"   : 3.0,
})

THETA_VAL = np.pi / 4

# ── Panel (a): temperature values and palette ─────────────────────────────────
# Same muted palette as bloch_disk_portrait panel (a)
BETAS_A      = [0.5,       1.0,       2.0,       3.0      ]
COLORS_A     = ["#c66b5a", "#e0a882", "#7a6aac", "#3a78b0"]
LINESTYLES_A = ["-",       "--",      "-.",      ":"      ]
LABELS_A     = [r"$\beta\omega_q=0.5$", r"$\beta\omega_q=1.0$",
                r"$\beta\omega_q=2.0$", r"$\beta\omega_q=3.0$"]

N_XI = 6000   # points along g/g_star axis

# ── Panel (b): heatmap grid ───────────────────────────────────────────────────
N_G_MAP    = 110
N_BETA_MAP = 90
G_ARR      = np.linspace(0.02, 1.40, N_G_MAP)
BETA_ARR   = np.linspace(0.35, 8.0,  N_BETA_MAP)

# ── Pre-compute chi0 and g_star for panel (a) ─────────────────────────────────
print("Computing chi0 for panel (a) temperatures...")
chi0_a  = []
gstar_a = []
for beta in BETAS_A:
    c0, _, _ = get_chi0(beta, theta=THETA_VAL)
    chi0_a.append(c0)
    gstar_a.append(1.0 / np.sqrt(c0) if c0 > 0 else np.nan)
    print(f"  beta={beta}: chi0={c0:.5f}, g_star={gstar_a[-1]:.4f}")

# ── Pre-compute heatmap data for panel (b) ────────────────────────────────────
print("Computing purity heatmap grid...")
purity_grid = np.zeros((N_BETA_MAP, N_G_MAP))
chi0_map    = np.zeros(N_BETA_MAP)
gstar_map   = np.zeros(N_BETA_MAP)

for i, beta in enumerate(BETA_ARR):
    c0, _, _ = get_chi0(beta, theta=THETA_VAL)
    chi0_map[i]  = c0
    gstar_map[i] = 1.0 / np.sqrt(c0) if c0 > 0 else np.nan
    _, r_arr, _, _ = bloch_ohmic(G_ARR, beta, theta=THETA_VAL)
    purity_grid[i, :] = r_arr**2
    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{N_BETA_MAP} rows done")

# Bare purity baseline and excess
r_bare_arr  = np.tanh(BETA_ARR * OMEGA_Q / 2.0)
purity_bare = r_bare_arr**2
delta_purity = purity_grid - purity_bare[:, np.newaxis]

GG, BB = np.meshgrid(G_ARR, BETA_ARR * OMEGA_Q)
CHI_MAP = GG**2 * chi0_map[:, np.newaxis]

print(f"Delta-purity range: [{delta_purity.min():.5f}, {delta_purity.max():.5f}]")
print(f"Bath always purifies: {(delta_purity >= 0).all()}")

# ── Figure layout: side-by-side panels ────────────────────────────────────────
# ── Figure layout: vertically stacked panels ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(3.8, 6.2),
                         gridspec_kw=dict(hspace=0.38, left=0.15, right=0.92,
                                          top=0.95, bottom=0.08))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): r(g/g_star) for four temperatures
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

xi_arr = np.geomspace(1e-2, 10.0, N_XI)

for beta, col, ls, lab, gstar in zip(BETAS_A, COLORS_A, LINESTYLES_A,
                                      LABELS_A, gstar_a):
    a = beta * OMEGA_Q / 2.0
    g_arr = xi_arr * gstar
    _, r_arr, _, _ = bloch_ohmic(g_arr, beta, theta=THETA_VAL)

    ax.plot(xi_arr, r_arr, color=col, ls=ls, lw=1.3, label=lab, zorder=3)

    # Bare thermal purity reference (dotted horizontal)
    r_bare = abs(np.tanh(a))
    ax.axhline(r_bare, color=col, ls=":", lw=0.7, alpha=0.55, zorder=2)

# Crossover marker and shading
ax.axvline(1.0, color="black", ls="--", lw=0.9, alpha=0.55, zorder=1)
ax.text(1.08, 0.06, r"$g_\star$", fontsize=8.5, color="#333333", va="bottom")
ax.axvspan(1e-2, 1.0, color="#ddeeff", alpha=0.40, zorder=0)
ax.axvspan(1.0, 10.0, color="#fff3e0", alpha=0.40, zorder=0)
ax.text(0.58, 0.18, r"\textbf{Weak}", fontsize=8.0, color="#2c7fb8",
        transform=ax.transAxes, ha="center")
ax.text(0.82, 0.18, r"\textbf{Strong}", fontsize=8.0, color="#d95f02",
        transform=ax.transAxes, ha="center")

ax.legend(fontsize=6.2, framealpha=0.88, loc="lower left",
          handlelength=1.8, borderpad=0.5, labelspacing=0.3)

ax.set_xscale("log")
ax.set_xlim(1e-2, 10)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$g / g_\star(\beta)$")
ax.set_ylabel(r"$r$")
ax.text(0.04, 0.96, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Delta(r^2) heatmap in (g, beta) plane
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

vmax_dp = delta_purity.max() * 1.02

pcm = ax.pcolormesh(GG, BB, delta_purity,
                    cmap="YlOrRd",
                    vmin=0.0, vmax=vmax_dp,
                    shading="auto", rasterized=True)
cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(r"$\Delta r^2 = r^2 - \tanh^2(\beta\omega_q/2)$", fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)

# chi=1 crossover locus (black dashed for contrast)
cs_chi = ax.contour(GG, BB, CHI_MAP, levels=[1.0],
                    colors=["black"], linewidths=[1.1],
                    linestyles=["--"], alpha=0.6, zorder=3)
ax.clabel(cs_chi, fmt=r"$\chi=1$", fontsize=7.0, colors="black",
          inline=True, inline_spacing=4)

# Annotations (moved and recolored black)
ax.text(0.06, 0.18, r"$\Delta r^2 \approx 0$",
        transform=ax.transAxes, fontsize=5.8,
        color="black", ha="left", va="center")
ax.annotate(r"bath purifies",
            xy=(1.30, 2.0),
            xytext=(1.10, 5.0),
            fontsize=5.8, color="black", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                            mutation_scale=5))

ax.set_xlabel(r"$g/\omega_q$")
ax.set_ylabel(r"$\beta\omega_q$")
ax.set_xlim(G_ARR[0], G_ARR[-1])
ax.set_ylim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.text(0.04, 0.96, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9, color="black")


# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_purity_figure.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
