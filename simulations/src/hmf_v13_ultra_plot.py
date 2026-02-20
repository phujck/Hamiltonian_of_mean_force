"""
Standalone plotting and analysis script for Ultra-High Precision HMF scans (v13).
Loads multi-dimensional convergence data and physical sweeps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ultra(args):
    data_path = Path(args.input_csv)
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    df = df[df["status"] == "ok"].copy()
    
    out_dir = data_path.parent
    prefix = data_path.stem
    
    # Analysis 1: Temperature Sweep for Different Discretizations
    # Pick a fixed g and theta
    g_fix = df["g"].unique()[0]
    theta_fix = df["theta"].unique()[0]
    
    sub = df[(df["g"] == g_fix) & (df["theta"] == theta_fix)].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # Panel A: Population vs Beta
    ax = axes[0]
    # Plot analytic reference once
    ref = sub.sort_values("beta").drop_duplicates("beta")
    ax.plot(ref["beta"], ref["best_p00"], color="black", linewidth=2.5, label="Best Analytic (v12)")
    
    for n_modes in sorted(sub["n_modes"].unique()):
        for n_cut in sorted(sub["n_cut"].unique()):
            # Only plot a few "interesting" ones to avoid clutter
            if n_cut in [6, 12]:
                line = sub[(sub["n_modes"] == n_modes) & (sub["n_cut"] == n_cut)].sort_values("beta")
                ax.plot(line["beta"], line["ed_p00"], "--", label=f"ED m{n_modes}c{n_cut}", alpha=0.7)
                
    ax.set_title(f"Population Sweep (g={g_fix:.2f}, theta={theta_fix:.2f})")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel B: RMSE Convergence vs n_cut
    ax = axes[1]
    for n_modes in sorted(sub["n_modes"].unique()):
        conv_rows = []
        for n_cut, grp in sub[sub["n_modes"] == n_modes].groupby("n_cut"):
            rmse = np.sqrt(np.mean((grp["ed_p00"] - grp["best_p00"])**2))
            conv_rows.append({"n_cut": n_cut, "rmse": rmse})
        conv_df = pd.DataFrame(conv_rows).sort_values("n_cut")
        ax.plot(conv_df["n_cut"], conv_df["rmse"], "o-", label=f"n_modes={n_modes}")
        
    ax.set_yscale("log")
    ax.set_title("Convergence of Population RMSE")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("RMSE (ED vs Best Analytic)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    
    fig.savefig(out_dir / f"{prefix}_convergence.png", dpi=180)
    print(f"Saved: {prefix}_convergence.png")
    
    # Analysis 2: Heatmap of RMSE at highest beta
    beta_max = sub["beta"].max()
    hm_data = sub[sub["beta"] == beta_max].copy()
    pivot = hm_data.pivot_table(index="n_modes", columns="n_cut", values="ed_p00", aggfunc=lambda x: np.abs(x - hm_data["best_p00"].iloc[0]))
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(pivot, aspect="auto", origin="lower", extent=[sub["n_cut"].min(), sub["n_cut"].max(), sub["n_modes"].min(), sub["n_modes"].max()])
    plt.colorbar(im, label="Abs Error vs Analytic")
    ax2.set_title(f"Error Distribution at beta={beta_max:.2f}")
    ax2.set_xlabel("n_cut")
    ax2.set_ylabel("n_modes")
    
    fig2.savefig(out_dir / f"{prefix}_heatmap.png", dpi=180)
    print(f"Saved: {prefix}_heatmap.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    args = parser.parse_args()
    plot_ultra(args)

if __name__ == "__main__":
    main()
