# -*- coding: utf-8 -*-
"""
plot_ergotropy.py  --  Figure: Ergotropy susceptibility and coherence work
==========================================================================
Two-panel figure, both plotted against g/g_star(beta).

  Panel (a): Ergotropy susceptibility  d_g W_erg  [physical units: omega_q]
             for several beta values.  Peak height increases with beta;
             peak position locks near g/g_star ~ 1 (the chi=1 crossover).

  Panel (b): Coherence-enabled ergotropy  Delta W_coh = 2*W_erg/omega_q
             for the same beta values (= r + mz, since mz < 0 throughout).
             Grows from zero at g=0 and increases with beta.

Bath: Ohmic, theta=pi/4.  Output: manuscript/figures/hmf_ergotropy.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import bloch_ohmic, get_chi0

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

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

# ── Beta values and palette ────────────────────────────────────────────────────
BETAS      = [1.0,       2.0,       3.0,       4.0      ]
COLORS     = ["#c66b5a", "#e0a882", "#7a6aac", "#3a78b0"]
LINESTYLES = ["-",       "--",      "-.",      ":"      ]
LABELS     = [r"$\beta\omega_q=1$", r"$\beta\omega_q=2$",
              r"$\beta\omega_q=3$", r"$\beta\omega_q=4$"]

N_XI = 1500
XI   = np.geomspace(5e-3, 8.0, N_XI)   # g / g_star

# ── Compute curves ─────────────────────────────────────────────────────────────
print("Computing Bloch curves...")
curves = []
for beta in BETAS:
    chi0, _, _ = get_chi0(beta, theta=THETA_VAL)
    gstar = 1.0 / np.sqrt(chi0)
    g_arr = XI * gstar

    phi, r, mx, mz = bloch_ohmic(g_arr, beta, theta=THETA_VAL)

    # δW_coh = 0.5 * (r + mz)  [assuming ω_q=1 and absorbing factor of 2]
    delta_W = 0.5 * (r + mz)
    # ∂_g δW_coh
    dW_dg = np.gradient(delta_W, g_arr)

    curves.append(dict(
        beta=beta, gstar=gstar,
        delta_W=delta_W, dW_dg=dW_dg,
    ))
    pk = np.max(dW_dg[XI > 0.1])
    xi_pk = XI[np.argmax(dW_dg * (XI > 0.1))]
    print(f"  beta={beta}: gstar={gstar:.4f}, "
          f"dW_max={delta_W.max():.5f}, "
          f"peak dW_dg={pk:.5f} @ g/g*={xi_pk:.3f}")

# ── Figure: two stacked panels ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(3.8, 5.6),
                          gridspec_kw=dict(hspace=0.40,
                                           left=0.16, right=0.97,
                                           top=0.97, bottom=0.09))

for ax in axes:
    ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.40, zorder=1)
    ax.axvspan(5e-3, 1.0, color="#ddeeff", alpha=0.28, zorder=0)
    ax.axvspan(1.0,  8.0, color="#fff3e0", alpha=0.32, zorder=0)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): ∂_g W_erg susceptibility
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

for i, c in enumerate(curves):
    # smooth away edge noise
    mask = (XI >= 0.04) & (XI <= 7.5)
    ax.plot(XI[mask], c["dW_dg"][mask],
            color=COLORS[i], ls=LINESTYLES[i], lw=1.3,
            label=LABELS[i], zorder=3)

ax.axhline(0, color="gray", lw=0.5, ls=":", zorder=1)
ax.set_xscale("log")
ax.set_xlim(5e-3, 8.0)
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r"$g / g_\star(\beta)$")
ax.set_ylabel(r"$\partial_g \delta W_{\rm coh}$")
ax.text(1.06, ax.get_ylim()[1]*0.92, r"$g_\star$",
        fontsize=8, color="#333333", va="top")
ax.legend(loc="upper right", fontsize=6.5, framealpha=0.88,
          handlelength=1.6, borderpad=0.4, labelspacing=0.25)
ax.text(0.04, 0.96, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)
ax.text(0.30, 0.52, r"\textbf{Weak}", fontsize=7.5, color="#2c7fb8",
        transform=ax.transAxes, ha="center")
ax.text(0.85, 0.52, r"\textbf{Strong}", fontsize=7.5, color="#d95f02",
        transform=ax.transAxes, ha="center")

# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): ΔW_coh = (2/ω_q) W_erg
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

for i, c in enumerate(curves):
    mask = XI >= 5e-3
    ax.plot(XI[mask], c["delta_W"][mask],
            color=COLORS[i], ls=LINESTYLES[i], lw=1.3,
            label=LABELS[i], zorder=3)

ax.set_xscale("log")
ax.set_xlim(5e-3, 8.0)
ax.set_ylim(bottom=0.0)
ax.set_xlabel(r"$g / g_\star(\beta)$")
ax.set_ylabel(r"$\delta W_{\rm coh}$")
ax.text(1.06, ax.get_ylim()[1]*0.05, r"$g_\star$",
        fontsize=8, color="#333333", va="bottom")
ax.text(0.04, 0.96, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", fontsize=9)

# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_ergotropy.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
