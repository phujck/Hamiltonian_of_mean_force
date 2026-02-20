# -*- coding: utf-8 -*-
"""
fig3_bandwidth_convergence.py  –  Figure 3: ED simulability crossover
======================================================================
Three panels showing the non-monotone convergence of ED as a function of
spectral window bandwidth and temperature.

Data required: data/bandwidth_v36.csv
  Columns used: bandwidth, beta, n_cut, d_ed_ord_p00

Setup: omega_q=2, g=0.5, N_modes=3, varying n_cut in {4, 5, 6}
Magic bandwidth: B* = 2*(omega_q - omega_floor) = 2*(2 - 0.1) = 3.8
Empirical RMSE minimum at B=3.2 (from hmf_crossover_effects_explainer_summary_codex_v44.csv)
Output: ../../manuscript/figures/hmf_fig3_bandwidth_convergence.pdf + .png
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA    = Path(__file__).parent / "data" / "bandwidth_v36.csv"
FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": False, "figure.dpi": 200, "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
})

B_STAR   = 3.8    # magic bandwidth = 2*(omega_q - omega_floor) = 2*(2 - 0.1)
B_OPT    = 3.2    # empirical RMSE minimum
NCUT_COLORS = {4: "#aec6f5", 5: "#5799d2", 6: "#1f4e91"}
TEMP_COLORS = {0.6: "#e377c2", 2.48: "#8c564b", 5.3: "#17becf"}

df = pd.read_csv(DATA)
bw_vals   = sorted(df["bandwidth"].unique())
beta_vals = sorted(df["beta"].unique())
ncut_vals = [4, 5, 6]

print(f"Loaded {len(df)} rows; bandwidths={bw_vals}; betas (sample)={beta_vals[:5]}")

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6))
fig.subplots_adjust(wspace=0.44)

# ── Panel (a): RMSE vs bandwidth, averaged over all beta, for ncut=4,5,6 ─────
ax = axes[0]
for nc in ncut_vals:
    sub = df[df.n_cut == nc]
    rmse_bw = []
    for bw in bw_vals:
        rows = sub[np.isclose(sub["bandwidth"], bw, atol=0.01)]
        rmse_bw.append(np.sqrt(np.mean(rows["d_ed_ord_p00"].values**2)))
    ax.plot(bw_vals, rmse_bw, color=NCUT_COLORS[nc], lw=1.3, marker="o", ms=3.5,
            label=rf"$n_{{\max}}={nc}$")

ax.axvline(B_STAR, color="#d62728", lw=1.2, ls="--", label=rf"$B^*={B_STAR}$")
ax.axvline(B_OPT,  color="#2ca02c", lw=1.0, ls="-.", label=rf"RMSE min $B={B_OPT}$")
ax.set_xlabel(r"Bandwidth $B$ ($\omega_q$ units)")
ax.set_ylabel(r"RMSE($p_{00}$)")
ax.set_title("(a) RMSE vs bandwidth", fontsize=8, pad=3)
ax.legend(fontsize=6, framealpha=0.85, loc="upper left")
ax.set_xlim(bw_vals[0] - 0.1, bw_vals[-1] + 0.1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

# ── Panel (b): Cutoff sensitivity |Delta(ncut=4) - Delta(ncut=6)| vs BW ──────
ax = axes[1]
sub4 = df[df.n_cut == 4]
sub6 = df[df.n_cut == 6]

for b_val, col in TEMP_COLORS.items():
    sens, bws = [], []
    for bw in bw_vals:
        r4 = sub4[(np.isclose(sub4["bandwidth"], bw, atol=0.01)) &
                  (np.isclose(sub4["beta"], b_val, atol=0.05))]
        r6 = sub6[(np.isclose(sub6["bandwidth"], bw, atol=0.01)) &
                  (np.isclose(sub6["beta"], b_val, atol=0.05))]
        if len(r4) and len(r6):
            sens.append(abs(r4["d_ed_ord_p00"].values[0] -
                            r6["d_ed_ord_p00"].values[0]))
            bws.append(bw)
    ax.plot(bws, sens, color=col, lw=1.2, marker="s", ms=2.8,
            label=rf"$\beta\omega_q={b_val}$")

ax.axvline(B_OPT, color="#2ca02c", lw=1.0, ls="-.", alpha=0.7)
ax.axvline(B_STAR, color="#d62728", lw=1.2, ls="--", alpha=0.7)
ax.set_xlabel(r"Bandwidth $B$")
ax.set_ylabel(r"$|p_{00}(n_{\max}=4)-p_{00}(n_{\max}=6)|$")
ax.set_title("(b) Cutoff sensitivity", fontsize=8, pad=3)
ax.legend(fontsize=6, framealpha=0.85)
ax.set_xlim(bw_vals[0] - 0.1, bw_vals[-1] + 0.1)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

# ── Panel (c): Residual vs beta at optimal bandwidth B=3.2 ───────────────────
ax = axes[2]
for nc in ncut_vals:
    sub = df[(df.n_cut == nc) & np.isclose(df["bandwidth"], B_OPT, atol=0.05)]
    sub = sub.sort_values("beta")
    ax.plot(sub["beta"].values, np.abs(sub["d_ed_ord_p00"].values),
            color=NCUT_COLORS[nc], lw=1.3, marker="o", ms=3.5,
            label=rf"$n_{{\max}}={nc}$")

ax.set_xlabel(r"$\beta\omega_q$")
ax.set_ylabel(r"$|\Delta p_{00}|$ at $B=3.2$")
ax.set_title(f"(c) Error vs temperature (B={B_OPT})", fontsize=8, pad=3)
ax.legend(fontsize=6, framealpha=0.85)

fig.suptitle(
    r"ED simulability crossover: $\omega_q=2$, $g=0.5$, $N_\omega=3$",
    fontsize=9, y=1.02)

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig3_bandwidth_convergence{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")
plt.close(fig)
