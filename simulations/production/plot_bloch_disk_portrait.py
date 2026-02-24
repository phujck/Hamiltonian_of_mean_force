# -*- coding: utf-8 -*-
"""
plot_bloch_disk_portrait.py  --  Figure A: Coupling-plane Bloch disk portrait
==============================================================================
Two-panel publication figure showing how the qubit mean-force state traces a
trajectory in the Bloch disk as coupling strength g increases from 0.

  Panel (a): State trajectory (mx, mz) in the coupling plane for 4 temperatures.
             The state starts at the bare thermal point on the -z axis and sweeps
             toward a temperature-dependent attractor as chi -> inf.
             A dotted line marks the bare coupling direction r (upper-left),
             showing explicitly that the attractor direction differs from r.

  Panel (b): Bloch radius r = |v| vs g/g_star(beta), showing coupling-induced
             purity changes. Dashed horizontals give bare r_bare = tanh(a) at
             each temperature; the crossover g/g_star = 1 is marked.

Bath model and coupling: Ohmic, theta = pi/4 (locked). Uses analytic formulas only.

Output: manuscript/figures/hmf_bloch_disk_portrait.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import bloch_ohmic, get_chi0, OMEGA_Q

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Publication-quality rcParams (PRL two-column style) ──────────────────────
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
    "xtick.minor.width" : 0.5,
    "ytick.minor.width" : 0.5,
    "xtick.major.size"  : 3.0,
    "ytick.major.size"  : 3.0,
})

# ── Temperature values and palette ───────────────────────────────────────────
# ColorBrewer RdBu-4: hot (red) -> cool (dark blue), colorblind-safe
BETAS       = [0.4,       0.8,       1.6,       3.2      ]
COLORS      = ["#d73027", "#f4a582", "#92c5de", "#2166ac"]
LINESTYLES  = ["-",       "--",      "-.",      ":"      ]
LABELS      = [r"$\beta\omega_q=0.4$",  r"$\beta\omega_q=0.8$",
               r"$\beta\omega_q=1.6$",    r"$\beta\omega_q=3.2$"  ]

N_G = 600  # resolution along g trajectory

# ── Pre-compute chi0 and g_star for each temperature ─────────────────────────
print("Computing chi0 for each beta...")
chi0_vals  = []
gstar_vals = []
for beta in BETAS:
    chi0, _, _ = get_chi0(beta)
    chi0_vals.append(chi0)
    gstar_vals.append(1.0 / np.sqrt(chi0) if chi0 > 0 else np.nan)
    print(f"  beta={beta}: chi0={chi0:.5f}, g_star={gstar_vals[-1]:.4f}")


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.8, 3.2))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0],
                        wspace=0.40, left=0.08, right=0.97,
                        top=0.94, bottom=0.14)
ax_disk  = fig.add_subplot(gs[0])
ax_purity = fig.add_subplot(gs[1])

ax = ax_disk

# — Background disk —
disk_bg = patches.Circle((0, 0), radius=1.0,
                          facecolor="#f7f7f7", edgecolor="black",
                          linewidth=0.9, zorder=0)
ax.add_patch(disk_bg)

# — Axis reference lines (thin dotted) —
ax.plot([-1.12, 1.12], [0, 0], color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)
ax.plot([0, 0], [-1.12, 0.15],  color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)

# — System axis label —
ax.text(0.04,  0.96, r"$\mathbf{n}_s$",       ha="left", va="top",
        fontsize=8, color="#444444")
ax.text(0.04, -0.96, r"$-\mathbf{n}_s$",      ha="left", va="bottom",
        fontsize=8, color="#444444")
ax.text(0.96, -0.04, r"$\hat{\mathbf{r}}_\perp$", ha="right", va="top",
        fontsize=8, color="#444444")

# — Bare coupling direction r (thin dashed, neutral gray; shows the naively
#   expected ultrastrong attractor direction per Cresser-Anders for reference) —
# For theta=pi/4: r_perp component = +sin(pi/4) along hat_r_perp,
# r_par component = +cos(pi/4) along n_s.  But in Bloch space the coupling
# vector is r = (-1/sqrt2, 0, 1/sqrt2) -- sign of r_x is negative because
# f = (sigma_z - sigma_x)/sqrt2, so it points to upper-LEFT in the (mx,mz) plane.
r_coupling = np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2)])
ax.annotate("", xy=0.72*r_coupling, xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#888888",
                            lw=0.8, mutation_scale=6),
            zorder=2)
ax.text(0.72*r_coupling[0] - 0.06, 0.72*r_coupling[1] + 0.06,
        r"$\hat{\mathbf{r}}$", ha="right", va="bottom",
        fontsize=8, color="#888888")

# — Trajectories —
for beta, col, ls, lab, gstar in zip(BETAS, COLORS, LINESTYLES, LABELS, gstar_vals):
    a = beta * OMEGA_Q / 2.0
    # Sweep g from 0 to 4*g_star for good saturation
    g_arr = np.linspace(0.0, 4.0 * gstar, N_G)
    phi, r, mx, mz = bloch_ohmic(g_arr, beta)

    # Main trajectory
    ax.plot(mx, mz, color=col, ls=ls, lw=1.3, label=lab, zorder=3)

    # Directional arrowhead at ~40% of the trajectory
    idx_arrow = int(0.40 * N_G)
    ax.annotate("",
                xy=(mx[idx_arrow + 3], mz[idx_arrow + 3]),
                xytext=(mx[idx_arrow], mz[idx_arrow]),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=0.0, mutation_scale=7),
                zorder=4)

    # g=0: starting point
    ax.plot(mx[0], mz[0], "o", color=col, ms=4.5, zorder=6,
            markeredgewidth=0.6, markeredgecolor="white")

    # g=g_star: crossover (star marker)
    idx_star = N_G // 4   # 25% along = exactly g_star since g goes to 4*g_star
    ax.plot(mx[idx_star], mz[idx_star], "*", color=col, ms=7, zorder=6,
            markeredgewidth=0.5, markeredgecolor="white")

    # g→∞ attractor (large-g limit: use g = 20*g_star)
    g_inf = np.array([20.0 * gstar])
    _, _, mx_inf, mz_inf = bloch_ohmic(g_inf, beta)
    ax.plot(mx_inf[0], mz_inf[0], "s", color=col, ms=4.5, zorder=6,
            markeredgewidth=0.6, markeredgecolor="white")

# — Marker legend entries —
ax.plot([], [], "ko",  ms=4.5, markeredgewidth=0.5, markeredgecolor="white",
        label=r"$g=0$")
ax.plot([], [], "k*",  ms=7.0, markeredgewidth=0.5, markeredgecolor="white",
        label=r"$g=g_\star$")
ax.plot([], [], "ks",  ms=4.5, markeredgewidth=0.5, markeredgecolor="white",
        label=r"$g\to\infty$")

ax.legend(fontsize=6.2, loc="upper right", framealpha=0.90,
          ncol=2, columnspacing=0.7, handlelength=1.5,
          borderpad=0.5, labelspacing=0.3)

ax.set_xlim(-0.22, 1.12)
ax.set_ylim(-1.10, 0.18)
ax.set_aspect("equal")
ax.set_xlabel(r"$m_x = \langle\sigma_x\rangle$")
ax.set_ylabel(r"$m_z = \langle\sigma_z\rangle$")
ax.text(0.03, 0.97, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)

# Annotate: "coupling drives state off axis"
ax.annotate(r"increasing $g$",
            xy=(0.28, -0.58), xytext=(0.52, -0.30),
            fontsize=6.5, color="#555555",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7,
                            mutation_scale=5),
            ha="center")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): Bloch radius r(g/g_star)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_purity

xi_arr = np.linspace(0.0, 3.5, N_G)   # g / g_star

for beta, col, ls, lab, gstar in zip(BETAS, COLORS, LINESTYLES, LABELS, gstar_vals):
    a = beta * OMEGA_Q / 2.0
    g_arr = xi_arr * gstar
    phi, r_arr, mx, mz = bloch_ohmic(g_arr, beta)

    # Main r(g/g_star) curve
    ax.plot(xi_arr, r_arr, color=col, ls=ls, lw=1.3, label=lab, zorder=3)

    # Bare thermal purity reference
    r_bare = abs(np.tanh(a))
    ax.axhline(r_bare, color=col, ls=":", lw=0.7, alpha=0.55, zorder=2)

# First bare-level annotation (just once)
ax.text(3.38, abs(np.tanh(BETAS[0] * OMEGA_Q / 2)) + 0.02,
        r"$r_{\rm bare}$", fontsize=6.0, color="#555555", va="bottom",
        ha="right")

# Crossover marker
ax.axvline(1.0, color="black", ls="--", lw=0.9, alpha=0.55, zorder=1)
ax.text(1.04, 0.04, r"$g_\star$", fontsize=7, color="#333333", va="bottom")

# Weak / strong shading
ax.axvspan(0, 1.0, color="#ddeeff", alpha=0.45, zorder=0)
ax.axvspan(1.0, 3.5, color="#fff3e0", alpha=0.45, zorder=0)
ax.text(0.14, 0.06, "Weak",   fontsize=7, color="#2c7fb8", fontweight="bold",
        transform=ax.transAxes)
ax.text(0.60, 0.06, "Strong", fontsize=7, color="#d95f02", fontweight="bold",
        transform=ax.transAxes)

ax.legend(fontsize=6.2, loc="lower right", framealpha=0.90,
          handlelength=1.5, borderpad=0.5, labelspacing=0.3)
ax.set_xlabel(r"$g / g_\star(\beta)$")
ax.set_ylabel(r"$r = |\mathbf{v}|$")
ax.set_xlim(0, 3.5)
ax.set_ylim(0, 1.04)
ax.set_xticks([0, 1, 2, 3])
ax.text(0.03, 0.97, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)

# Note: all curves lie above their r_bare line (bath always purifies).
# Annotate the effect at high T where the gap is most visible.
ax.annotate(r"$r > r_{\rm bare}$ always",
            xy=(0.8, 0.62), xytext=(1.8, 0.46),
            fontsize=5.8, color="#555555",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7,
                            mutation_scale=5),
            ha="center")


# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_bloch_disk_portrait.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
