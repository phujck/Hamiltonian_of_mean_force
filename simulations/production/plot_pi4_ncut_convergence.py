
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.usetex": True, "figure.dpi": 200, "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
})

DATA_FILE = Path("data/pi4_ncut_convergence.csv")
df = pd.read_csv(DATA_FILE)

# CRITICAL FIX: The CSV contains a sweep over g. 
# We must filter for a single coupling strength to avoid "spikes" caused by mixing different g-manifolds.
TARGET_G = 0.05
df = df[np.isclose(df['g'], TARGET_G)]

fig, ax = plt.subplots(figsize=(5.5, 4))

# Selection of lines to plot
lines = [
    (2, 6, "n=2, c=6 (Low)", "#e41a1c"),
    (2, 20, "n=2, c=20", "#377eb8"),
    (2, 60, "n=2, c=60", "#4daf4a"),
    (4, 4, "n=4, c=4", "#984ea3"),
    (4, 6, "n=4, c=6", "#ff7f00"),
]

groups = df.groupby(['n_modes', 'n_cut'])

for n_m, n_c, label, col in lines:
    if (n_m, n_c) in groups.groups:
        sub = groups.get_group((n_m, n_c)).sort_values("beta")
        dim = sub["dim"].iloc[0]
        ax.plot(sub["beta"], sub["ed_p11"], label=f"{label} [D={dim}]", color=col)

# Reference Line: Highest fidelity in the set for this g
best_row = df.loc[df['dim'].idxmax()]
n_m_best, n_c_best = int(best_row['n_modes']), int(best_row['n_cut'])
sub_best = groups.get_group((n_m_best, n_c_best)).sort_values("beta")
ax.plot(sub_best["beta"], sub_best["ed_p11"], 'k--', lw=1.8, label=f"Max Fidelity (n={n_m_best}, c={n_c_best})", zorder=5)

ax.set_xlabel(r"Inverse Temperature $\beta$")
ax.set_ylabel(r"Excited State Population $p_{11}$")
ax.set_title(rf"ED Convergence to Mean-Force Limit ($\theta=\pi/4, g={TARGET_G}$)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(alpha=0.3, ls=":")
ax.set_ylim(0.5, 1.0) # Population is biased towards excited state in this regime

plt.tight_layout()
plt.savefig("../../manuscript/figures/hmf_ncut_convergence.png", dpi=300)
plt.savefig("../../manuscript/figures/hmf_ncut_convergence.pdf")
print(f"Generated clean convergence plot for g={TARGET_G}")
