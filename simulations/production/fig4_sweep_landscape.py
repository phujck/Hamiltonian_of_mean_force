# -*- coding: utf-8 -*-
"""
fig4_sweep_landscape.py  –  Figure 4: Full parameter-sweep landscape
=====================================================================
2×3 grid: top row = raw p00; bottom row = signed residuals (disc - ED, cont - ED)
across coupling, temperature, and angle sweeps.

Data required: data/sweeps_v43.csv
  Columns used: sweep, g, beta, theta, ed_p00,
                analytic_disc_p00, analytic_cont_p00

Output: ../../manuscript/figures/hmf_fig4_sweep_landscape.pdf + .png
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA    = Path(__file__).parent / "data" / "sweeps_v43.csv"
FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": False, "figure.dpi": 200, "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
})

C_ED   = "#2ca02c"
C_DISC = "#1f77b4"
C_CONT = "#ff7f0e"

# (sweep, x_col, x_label, fixed_col, fixed_val, g_star_approx or None)
SWEEP_CONFIG = [
    ("coupling",    "g",    r"$g/\omega_q$",    "beta",  2.0,  0.14),  # g_star(beta=2)~0.14
    ("temperature", "beta", r"$\beta\omega_q$", "g",     0.5,  None),
    ("angle",       "theta",r"$\theta$ (rad)",  "beta",  2.0,  None),
]

TOP_LABELS = ["(a)", "(c)", "(e)"]
BOT_LABELS = ["(b)", "(d)", "(f)"]

df = pd.read_csv(DATA)

fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), sharex="col")
fig.subplots_adjust(hspace=0.36, wspace=0.42)

for col, ((sw, px, xl, fix_col, fix_val, gstar), tl, bl) in enumerate(
        zip(SWEEP_CONFIG, TOP_LABELS, BOT_LABELS)):

    sub = df[df.sweep == sw].copy()
    uniq = sub[fix_col].unique()
    best = uniq[np.argmin(np.abs(uniq - fix_val))]
    sub  = sub[np.isclose(sub[fix_col], best, atol=0.01)].sort_values(px)

    x    = sub[px].values
    ed   = sub["analytic_disc_p00" if len(sub["ed_p00"].dropna()) == 0
               else "ed_p00"].values     # fallback guard
    ed   = sub["ed_p00"].values
    disc = sub["analytic_disc_p00"].values
    cont = sub["analytic_cont_p00"].values

    # Top: raw p00
    ax_t = axes[0, col]
    ax_t.plot(x, ed,   color=C_ED,   lw=1.5, label="ED")
    ax_t.plot(x, disc, color=C_DISC, lw=1.2, ls="--", label="Disc.")
    ax_t.plot(x, cont, color=C_CONT, lw=1.0, ls=":",  label="Cont.")
    if gstar is not None:
        ax_t.axvline(gstar, color="k", lw=0.8, ls=":", alpha=0.5,
                     label=r"$g_\star$")
    ax_t.set_ylabel(r"$p_{00}$")
    ax_t.set_title(sw.capitalize() + " sweep", fontsize=8, pad=3)
    ax_t.text(0.05, 0.93, tl, transform=ax_t.transAxes, fontsize=8, fontweight="bold")
    if col == 0:
        ax_t.legend(fontsize=6, framealpha=0.85, loc="upper right")

    # Bottom: signed residuals
    ax_b = axes[1, col]
    ax_b.axhline(0, color="k", lw=0.6)
    ax_b.plot(x, disc - ed, color=C_DISC, lw=1.2, ls="--", label="Disc. - ED")
    ax_b.plot(x, cont - ed, color=C_CONT, lw=1.0, ls=":",  label="Cont. - ED")
    if gstar is not None:
        ax_b.axvline(gstar, color="k", lw=0.8, ls=":", alpha=0.5)
    ax_b.set_xlabel(xl)
    ax_b.set_ylabel(r"$\Delta p_{00}$")
    ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax_b.text(0.05, 0.93, bl, transform=ax_b.transAxes, fontsize=8, fontweight="bold")
    if col == 0:
        ax_b.legend(fontsize=6, framealpha=0.85, loc="lower right")

    # Fixed-value annotation
    fix_str = (rf"$\beta\omega_q={best:.1f}$" if fix_col == "beta"
               else rf"$g={best:.2f}$")
    ax_t.text(0.97, 0.05, fix_str, transform=ax_t.transAxes,
              fontsize=6, ha="right", color="k", alpha=0.6)

fig.suptitle(
    r"Parameter-sweep landscape: disc.\ (blue dashed) and cont.\ (orange dotted) "
    r"vs ED"
    "\n"
    r"$\omega_q=1$, $N_\omega=2$, $n_{\max}=6$, window $[0.2,\,1.8]$",
    fontsize=8, y=1.02)

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig4_sweep_landscape{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")
plt.close(fig)
