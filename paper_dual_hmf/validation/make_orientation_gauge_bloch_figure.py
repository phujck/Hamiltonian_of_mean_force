"""Generate an orientation-gauge geometry figure (Bloch sphere + affine reflection).

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_orientation_gauge_bloch.pdf
  paper_dual_hmf/manuscript/figures/hmf_orientation_gauge_bloch.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_dual_hmf" / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_sphere(ax, r: float = 1.0) -> None:
    u = np.linspace(0, 2 * np.pi, 70)
    v = np.linspace(0, np.pi, 45)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        alpha=0.13,
        color="#6fa8dc",
        linewidth=0,
        shade=True,
    )

    # Equator and z-axis for the reflection interpretation.
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r * np.cos(t), r * np.sin(t), 0 * t, color="0.25", lw=1.0, ls="--")
    ax.plot([0, 0], [0, 0], [-r, r], color="0.15", lw=1.0)


def main() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "figure.dpi": 200,
        }
    )

    beta = 2.0
    beta_eff = 1.3
    beta_eff_prime = 2.0 * beta - beta_eff

    # Representative Bloch vectors with identical radius and opposite oriented longitudinal component.
    m = np.array([0.48, 0.26, 0.64], dtype=float)
    mprime = np.array([m[0], m[1], -m[2]], dtype=float)
    radius = float(np.linalg.norm(m))

    fig = plt.figure(figsize=(7.1, 3.3), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    make_sphere(ax1, r=1.0)
    ax1.quiver(0, 0, 0, m[0], m[1], m[2], color="#cc0000", lw=2.0, arrow_length_ratio=0.09)
    ax1.quiver(
        0, 0, 0, mprime[0], mprime[1], mprime[2], color="#0b5394", lw=2.0, arrow_length_ratio=0.09
    )
    ax1.plot([m[0], mprime[0]], [m[1], mprime[1]], [m[2], mprime[2]], color="0.25", lw=1.0, ls="--")

    ax1.text(m[0] * 1.05, m[1] * 1.05, m[2] * 1.05, r"$m$", color="#cc0000", fontsize=8)
    ax1.text(
        mprime[0] * 1.04,
        mprime[1] * 1.04,
        mprime[2] * 1.04,
        r"$m'$",
        color="#0b5394",
        fontsize=8,
    )
    ax1.set_xlim([-1.05, 1.05])
    ax1.set_ylim([-1.05, 1.05])
    ax1.set_zlim([-1.05, 1.05])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("(a) Orientation-gauge representative pair")
    ax1.view_init(elev=22, azim=42)
    ax1.text2D(0.02, 0.95, rf"$|m|=|m'|={radius:.2f}$", transform=ax1.transAxes, fontsize=7)
    ax1.text2D(0.02, 0.08, r"$m'=(m_x,m_y,-m_z)$", transform=ax1.transAxes, fontsize=7)
    ax1.set_xticks([-1, 0, 1])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_zticks([-1, 0, 1])
    ax1.legend(
        handles=[
            Line2D([0], [0], color="#cc0000", lw=2.0, label=r"$m$"),
            Line2D([0], [0], color="#0b5394", lw=2.0, label=r"$m'$"),
        ],
        loc="upper right",
        framealpha=0.9,
    )

    # Affine reflection in beta_eff representatives.
    x = np.linspace(0.0, 2.0 * beta, 200)
    y = 2.0 * beta - x
    ax2.plot(x, y, color="0.2", lw=1.4, label=r"$\beta_{\mathrm{eff}}'=2\beta-\beta_{\mathrm{eff}}$")
    ax2.plot([beta_eff], [beta_eff_prime], "o", ms=4.5, color="#cc0000")
    ax2.plot([beta_eff_prime], [beta_eff], "o", ms=4.5, color="#0b5394")
    ax2.plot([beta, beta], [0, 2.0 * beta], color="0.4", lw=1.0, ls="--", label=r"symmetry line $\beta$")
    ax2.plot([0, 2.0 * beta], [beta, beta], color="0.4", lw=1.0, ls="--")
    ax2.plot([beta], [beta], marker="s", ms=4.0, color="0.1")
    ax2.annotate(
        "",
        xy=(beta_eff_prime, beta_eff),
        xytext=(beta_eff, beta_eff_prime),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0),
    )
    ax2.text(beta + 0.08, beta + 0.08, r"fixed point $(\beta,\beta)$", fontsize=7, color="0.2")
    ax2.annotate(
        r"$\beta_{\mathrm{eff}}$",
        xy=(beta_eff, beta_eff_prime),
        xytext=(beta_eff + 0.18, beta_eff_prime + 0.2),
        fontsize=7,
        color="#cc0000",
    )
    ax2.annotate(
        r"$\beta'_{\mathrm{eff}}$",
        xy=(beta_eff_prime, beta_eff),
        xytext=(beta_eff_prime - 0.95, beta_eff - 0.35),
        fontsize=7,
        color="#0b5394",
    )

    ax2.set_xlim(0.0, 2.0 * beta)
    ax2.set_ylim(0.0, 2.0 * beta)
    ax2.set_xlabel(r"$\beta_{\mathrm{eff}}$")
    ax2.set_ylabel(r"$\beta_{\mathrm{eff}}'$")
    ax2.set_title("(b) Affine representative reflection")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="lower left", framealpha=0.9)

    fig.suptitle("Orientation gauge: odd longitudinal flip with radial invariance", fontsize=10.1)

    out_pdf = OUT_DIR / "hmf_orientation_gauge_bloch.pdf"
    out_png = OUT_DIR / "hmf_orientation_gauge_bloch.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
