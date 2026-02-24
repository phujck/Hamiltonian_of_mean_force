# -*- coding: utf-8 -*-
"""
fig3_bandwidth_convergence.py  –  Figure 3: ED simulability crossover (REFINED)
======================================================================
Single panel showing the cutoff sensitivity Delta p11 
as a function of spectral window bandwidth in units of omega_q.

Data required: data/bandwidth_v36.csv
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

TEMP_COLORS = {0.6: "#e377c2", 2.48: "#8c564b", 5.3: "#17becf"}

df = pd.read_csv(DATA)
# Convert bandwidth to units of omega_q (omega_q=2 in this data)
OMEGA_Q = 2.0
df["bw_scaled"] = df["bandwidth"] / OMEGA_Q

bw_vals   = sorted(df["bw_scaled"].unique())
beta_vals = sorted(df["beta"].unique())
ncut_vals = [4, 5, 6]

print(f"Loaded {len(df)} rows; scaled bandwidths={bw_vals}")

fig, ax = plt.subplots(figsize=(3.4, 2.6))

# ── Cutoff sensitivity |Delta(ncut=4) - Delta(ncut=6)| vs BW ──────
sub4 = df[df.n_cut == 4]
sub6 = df[df.n_cut == 6]

for b_val, col in TEMP_COLORS.items():
    sens, bws = [], []
    for bw in bw_vals:
        r4 = sub4[(np.isclose(sub4["bw_scaled"], bw, atol=0.01)) &
                  (np.isclose(sub4["beta"], b_val, atol=0.05))]
        r6 = sub6[(np.isclose(sub6["bw_scaled"], bw, atol=0.01)) &
                  (np.isclose(sub6["beta"], b_val, atol=0.05))]
        if len(r4) and len(r6):
            # Difference in p00 or p11 (both give same delta as p00+p11=1)
            sens.append(abs(r4["d_ed_ord_p00"].values[0] -
                            r6["d_ed_ord_p00"].values[0]))
            bws.append(bw)
    ax.plot(bws, sens, color=col, lw=1.2, marker="s", ms=2.8,
            label=rf"$\beta\omega_q={b_val}$")

ax.set_xlabel(r"Bandwidth $B$ (units of $\omega_q$)")
ax.set_ylabel(r"$\Delta p_{11}$")
ax.legend(fontsize=7, framealpha=0.85, loc="upper left")
ax.set_xlim(bw_vals[0] - 0.05, bw_vals[-1] + 0.05)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig3_bandwidth_convergence{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")
plt.close(fig)
