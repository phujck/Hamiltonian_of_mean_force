"""
make_crossover_involutive_figure.py
===================================
Generates a single-panel figure showing the involutive crossover geometry
between beta* and g*, marking the boundary between weak and strong coupling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys

# Add root to sys.path to import base functions
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulations.production.plot_bloch_branch_flip_test as base

# Paths
FIGURES_DIR = ROOT / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def _configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "figure.dpi": 300,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })

def make_figure():
    _configure_matplotlib()
    
    # Parameters matches plot_branch_bifurcation_involutive_manuscript.py
    theta = base.THETA_VAL
    beta_min = 0.05
    beta_max = 25.0
    n_beta = 150
    n_g = 120
    
    beta_grid = np.geomspace(beta_min, beta_max, n_beta)
    chi0_ref, _, _ = base.get_channels0(base.BETA_REF, theta=theta)
    gstar_ref = 1.0 / np.sqrt(chi0_ref)
    
    g_mult_grid = np.geomspace(0.05, 5.0, n_g)
    g_grid = g_mult_grid * gstar_ref
    
    # Precompute chi0(beta)
    chi0_beta = np.array([base.get_channels0(float(b), theta=theta)[0] for b in beta_grid])
    
    # 2D grid for y = log chi
    chi_grid = np.outer(g_grid**2, chi0_beta)
    y_grid = np.log(np.clip(chi_grid, 1e-12, None))
    
    # Involutive constructions
    gstar_beta = 1.0 / np.sqrt(np.clip(chi0_beta, 1e-20, None))
    
    # Solve beta_star(g) by interpolation
    # chi0(beta) is monotone, so we can interpolate beta as func of 1/g^2
    g_samples = np.geomspace(0.06, 4.8, 40) * gstar_ref
    target_chi0 = 1.0 / (g_samples**2)
    beta_star_samples = np.interp(target_chi0, chi0_beta, beta_grid, left=np.nan, right=np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    
    B, Gm = np.meshgrid(beta_grid, g_mult_grid)
    y_clip = np.clip(y_grid, -3.0, 4.0)
    
    # Heatmap
    im = ax.pcolormesh(B, Gm, y_clip, shading="auto", cmap="coolwarm", rasterized=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$y = \log \chi$", fontsize=9)
    
    # Contour at chi=1 (y=0)
    ax.contour(B, Gm, y_grid, levels=[0.0], colors=["#111111"], linewidths=1.8)
    
    # g_star(beta) curve
    ax.plot(beta_grid, gstar_beta / gstar_ref, color="#f4f4f4", lw=2.5, alpha=0.9)
    ax.plot(beta_grid, gstar_beta / gstar_ref, color="#4b4b4b", lw=1.2, label=r"$g_\star(\beta)$")
    
    # beta_star(g) samples
    valid = np.isfinite(beta_star_samples)
    ax.scatter(beta_star_samples[valid], g_samples[valid] / gstar_ref, 
               s=18, facecolors="none", edgecolors="#1d1d1d", linewidths=0.8, 
               label=r"$\beta_\star(g)$", zorder=6)
    
    # Reference chi=1 custom legend handle
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#111111', lw=1.8),
                    Line2D([0], [0], color='#4b4b4b', lw=1.2),
                    Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='#1d1d1d', markersize=4, lw=0)]
    
    ax.legend(custom_lines, [r"$\chi = 1$", r"$g_\star(\beta)$", r"$\beta_\star(g)$"], 
              loc="upper left", fontsize=7.5, framealpha=0.8)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(beta_min, beta_max)
    ax.set_ylim(0.05, 5.0)
    
    ax.set_xlabel(r"$\beta\omega_q$")
    ax.set_ylabel(r"$g/g_\star^{(\rm ref)}$")
    
    ax.grid(True, which="both", alpha=0.15, lw=0.4)
    
    plt.tight_layout()
    
    # Save
    fig_name = "hmf_crossover_involutive_v1"
    png_path = FIGURES_DIR / f"{fig_name}.png"
    pdf_path = FIGURES_DIR / f"{fig_name}.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

if __name__ == "__main__":
    make_figure()
