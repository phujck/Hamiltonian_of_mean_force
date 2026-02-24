# -*- coding: utf-8 -*-
"""
plot_purity_landscape.py  --  Figure C: Purity landscape in the (g, beta) plane
=================================================================================
Two-panel heatmap figure showing how the qubit purity r^2 = |v|^2 depends on
coupling strength g and inverse temperature beta.

  Panel (a): Purity r^2(g, beta) -- the absolute purity of the mean-force state.
             Contours:
               - White dashed: chi=1 crossover locus g_star(beta).
               - Black solid:  Purity-inversion boundary where r^2 = tanh^2(a),
                               i.e., coupling has no net effect on purity.
                               Above/left: coupling ENHANCES purity.
                               Below/right: coupling DEGRADES purity.

  Panel (b): Signed excess purity  Delta(r^2) = r^2(g,beta) - tanh^2(a).
             Positive (red):  bath coupling increases purity above bare value.
             Negative (blue): bath coupling decreases purity below bare value.
             The zero-crossing (black contour) is the same purity-inversion
             boundary as panel (a).

Key result: At high T (small beta), any coupling INCREASES purity because
the bare state is near the maximally-mixed center of the Bloch disk.
At low T (large beta), coupling DECREASES purity from the near-pure ground state.

Bath model and coupling: Ohmic, theta=pi/4.  All analytic, no simulation data.

Output: manuscript/figures/hmf_purity_landscape.{pdf,png}
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

# ── Grid parameters ───────────────────────────────────────────────────────────
N_G    = 110
N_BETA = 90
G_ARR    = np.linspace(0.02, 1.40, N_G)        # g / omega_q
BETA_ARR = np.linspace(0.35, 8.0,  N_BETA)     # beta * omega_q values (divided by omega_q)

# ── Compute purity grid ───────────────────────────────────────────────────────
print("Computing purity grid...")
purity_grid = np.zeros((N_BETA, N_G))
chi0_arr    = np.zeros(N_BETA)
gstar_arr   = np.zeros(N_BETA)

for i, beta in enumerate(BETA_ARR):
    chi0, _, _ = get_chi0(beta)
    chi0_arr[i]  = chi0
    gstar_arr[i] = 1.0 / np.sqrt(chi0) if chi0 > 0 else np.nan
    _, r_arr, _, _ = bloch_ohmic(G_ARR, beta)
    purity_grid[i, :] = r_arr**2

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_BETA} done")

# Bare purity (g=0 baseline): tanh^2(beta * omega_q / 2)
r_bare_arr   = np.tanh(BETA_ARR * OMEGA_Q / 2.0)
purity_bare  = r_bare_arr**2          # shape (N_BETA,)
delta_purity = purity_grid - purity_bare[:, np.newaxis]   # shape (N_BETA, N_G)

print(f"Purity range:       [{purity_grid.min():.4f}, {purity_grid.max():.4f}]")
print(f"Delta-purity range: [{delta_purity.min():.4f}, {delta_purity.max():.4f}]")
print(f"Delta-purity always positive: {(delta_purity > 0).all()}")

# ── Axis meshgrid (physical units: g and beta*omega_q) ───────────────────────
GG, BB = np.meshgrid(G_ARR, BETA_ARR * OMEGA_Q)

# chi contour data: chi(g, beta) = g^2 * chi0(beta)
CHI_MAP = GG**2 * chi0_arr[:, np.newaxis]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1),
                         gridspec_kw=dict(wspace=0.46, left=0.08, right=0.97,
                                         top=0.94, bottom=0.14))


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): Absolute purity r^2
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

pcm = ax.pcolormesh(GG, BB, purity_grid, cmap="viridis",
                    vmin=0.0, vmax=1.0, shading="auto", rasterized=True)
cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(r"$r^2 = |\mathbf{v}|^2$", fontsize=8)
cbar.ax.tick_params(labelsize=6.5)

# chi=1 crossover locus (white dashed)
cs_chi = ax.contour(GG, BB, CHI_MAP, levels=[1.0],
                    colors=["white"], linewidths=[1.2],
                    linestyles=["--"], zorder=3)
ax.clabel(cs_chi, fmt=r"$\chi=1$", fontsize=6.0, colors="white",
          inline=True, inline_spacing=4)

# Purity-inversion boundary (r^2 = tanh^2(a)) -- draw if it exists
# For narrow bandwidth (OMEGA_MAX=2omega_q), this contour may not appear
# because the longitudinal bath response dominates and coupling always purifies.
try:
    cs_inv = ax.contour(GG, BB, delta_purity, levels=[0.0],
                        colors=["black"], linewidths=[1.4],
                        linestyles=["-"], zorder=4)
    if cs_inv.allsegs[0]:  # contour exists
        ax.clabel(cs_inv, fmt=r"$r^2 = r_{\rm bare}^2$", fontsize=5.5,
                  colors="black", inline=True, inline_spacing=3)
except Exception:
    pass

# "Max purification" annotation at high-T, large-g corner
ax.annotate("max\npurification",
            xy=(G_ARR[-1]*0.85, BETA_ARR[2] * OMEGA_Q),
            xytext=(G_ARR[-1]*0.55, BETA_ARR[5] * OMEGA_Q),
            fontsize=5.8, color="white", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7,
                            mutation_scale=5))

ax.set_xlabel(r"$g/\omega_q$")
ax.set_ylabel(r"$\beta\omega_q$")
ax.set_xlim(G_ARR[0], G_ARR[-1])
ax.set_ylim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9, color="white")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Purity enhancement Delta(r^2) = r^2 - tanh^2(a)
#
# For Ohmic bath with OMEGA_MAX=2*omega_q, the longitudinal channel (dz0)
# dominates so delta_purity > 0 everywhere: coupling always purifies.
# Use sequential colormap.  For broader bandwidths, transverse response (sx0)
# grows and a sign change (coupling degrades purity) can appear.
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

vmax_dp = delta_purity.max() * 1.02

pcm2 = ax.pcolormesh(GG, BB, delta_purity, cmap="YlOrRd",
                     vmin=0.0, vmax=vmax_dp,
                     shading="auto", rasterized=True)
cbar2 = fig.colorbar(pcm2, ax=ax, fraction=0.046, pad=0.03)
cbar2.set_label(r"$r^2 - \tanh^2 a$", fontsize=8)
cbar2.ax.tick_params(labelsize=6.5)

# chi=1 crossover (white dashed)
cs_chi2 = ax.contour(GG, BB, CHI_MAP, levels=[1.0],
                     colors=["white"], linewidths=[1.2],
                     linestyles=["--"], zorder=3)
ax.clabel(cs_chi2, fmt=r"$\chi=1$", fontsize=6.0, colors="white",
          inline=True, inline_spacing=4)

# Contour of delta_r^2 at half its maximum value (to show the gradient)
half_max = delta_purity.max() / 2.0
cs_half = ax.contour(GG, BB, delta_purity, levels=[half_max],
                     colors=["black"], linewidths=[0.8],
                     linestyles=[":"], zorder=4)

# Annotation: large purification at high T, small g
ax.text(0.25, 0.18, r"$\Delta r^2 \approx 0$" + "\n(cold)",
        transform=ax.transAxes, fontsize=5.8,
        color="#222222", ha="center", va="center")
ax.annotate(r"bath purifies",
            xy=(G_ARR[int(0.7*N_G)], BETA_ARR[5] * OMEGA_Q),
            xytext=(G_ARR[int(0.45*N_G)], BETA_ARR[12] * OMEGA_Q),
            fontsize=5.8, color="#222222", ha="center",
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.7,
                            mutation_scale=5))

ax.set_xlabel(r"$g/\omega_q$")
ax.set_ylabel(r"$\beta\omega_q$")
ax.set_xlim(G_ARR[0], G_ARR[-1])
ax.set_ylim(BETA_ARR[0] * OMEGA_Q, BETA_ARR[-1] * OMEGA_Q)
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9, color="#222222")


# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_purity_landscape.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
