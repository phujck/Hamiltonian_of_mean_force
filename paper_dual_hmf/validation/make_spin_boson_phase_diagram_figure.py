"""Generate the exact canonical-channel spin-boson phase figure by spectral class.

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_phase_diagram.pdf
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_phase_diagram.png
"""

from __future__ import annotations

from math import gamma
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_dual_hmf" / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Illustration parameters in manuscript normalization.
OMEGA_Q = 1.0
OMEGA_C = 1.0
G_MAX = 1.4


def lambda_s(s: np.ndarray, omega_c: float = OMEGA_C) -> np.ndarray:
    """Exact low-temperature slope lambda_s = 2 * omega_c * Gamma(s)."""
    s_clipped = np.clip(np.asarray(s, dtype=float), 1e-6, None)
    gamma_vec = np.vectorize(gamma)
    return 2.0 * omega_c * gamma_vec(s_clipped)


def g_c_exact(s: np.ndarray, omega_q: float = OMEGA_Q, omega_c: float = OMEGA_C) -> np.ndarray:
    """Exact critical coupling: g_c^2 = omega_q / (4 * omega_c * Gamma(s))."""
    lam = lambda_s(s, omega_c=omega_c)
    return np.sqrt(omega_q / (2.0 * lam))


def m_beta(g: np.ndarray, s: float, beta: float, omega_q: float = OMEGA_Q, omega_c: float = OMEGA_C) -> np.ndarray:
    """Finite-beta order parameter from the exact canonical-channel closure law."""
    lam = float(lambda_s(np.array([s]), omega_c=omega_c)[0])
    arg = beta * (g * g * lam - 0.5 * omega_q)
    return np.tanh(arg)


def m_beta_reduced(x: np.ndarray, beta: float, omega_q: float = OMEGA_Q) -> np.ndarray:
    """Universal finite-beta sharpening in reduced coupling x = g / g_c."""
    return np.tanh(0.5 * beta * omega_q * (x * x - 1.0))


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

    s_grid = np.linspace(0.12, 2.2, 500)
    gc_grid = g_c_exact(s_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.25), constrained_layout=True)

    # Panel (a): exact spectral-class critical line in (s, g)
    ax1.axvspan(0.1, 1.0, color="#f7f7f7", alpha=0.65, lw=0)
    ax1.axvspan(1.0, 2.2, color="#eef6ff", alpha=0.65, lw=0)
    ax1.fill_between(
        s_grid,
        gc_grid,
        G_MAX,
        color="#f4cccc",
        alpha=0.88,
        label=r"$m_0=+1$ sector",
    )
    ax1.fill_between(s_grid, 0.0, gc_grid, color="#d9ead3", alpha=0.9, label=r"$m_0=-1$ sector")
    ax1.plot(s_grid, gc_grid, color="#990000", lw=1.7, label=r"exact $g_c(s)$")
    ax1.axvline(1.0, color="0.25", lw=1.0, ls="--")
    gc_ohm = float(g_c_exact(np.array([1.0]))[0])
    ax1.plot([1.0], [gc_ohm], marker="o", ms=4.6, color="#0b5394", label=rf"Ohmic point: $g_c={gc_ohm:.2f}$")
    ax1.text(0.26, 1.18, r"$m_0=+1$", color="#7f0000", fontsize=7)
    ax1.text(0.28, 0.12, r"$m_0=-1$", color="#274e13", fontsize=7)
    ax1.text(0.37, 1.33, "sub-Ohmic", fontsize=7, color="0.25")
    ax1.text(1.16, 1.33, "super-Ohmic", fontsize=7, color="0.25")
    ax1.text(1.02, 0.90, "Ohmic", fontsize=7, color="#0b5394", rotation=90, va="center")

    ax1.set_xlim(0.1, 2.2)
    ax1.set_ylim(0.0, G_MAX)
    ax1.set_xlabel(r"spectral exponent $s$")
    ax1.set_ylabel(r"coupling $g$")
    ax1.set_title("(a) Exact critical line")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.18)

    # Panel (b): universal sharpening in reduced coupling x = g/g_c.
    x_grid = np.linspace(0.25, 2.1, 420)
    beta_values = [2.0, 4.0, 8.0]
    colors = ["#0b5394", "#0086a8", "#6a329f"]
    for bval, col in zip(beta_values, colors):
        ax2.plot(
            x_grid,
            m_beta_reduced(x_grid, bval),
            color=col,
            lw=1.7,
            label=rf"$\beta\omega_q={bval:.0f}$",
        )

    ax2.axhline(0.0, color="0.25", lw=0.95, ls="--")
    ax2.axvline(1.0, color="0.25", lw=1.0, ls="--")
    ax2.text(1.03, 0.90, r"$g=g_c$", fontsize=7, color="0.2")
    ax2.text(0.40, -0.92, r"$m_0=-1$", fontsize=7, color="#274e13")
    ax2.text(1.55, 0.82, r"$m_0=+1$", fontsize=7, color="#7f0000")
    ax2.set_xlim(0.25, 2.1)
    ax2.set_ylim(-1.03, 1.03)
    ax2.set_xlabel(r"reduced coupling $g/g_c(s)$")
    ax2.set_ylabel(r"$m_z(\beta,g;s)$")
    ax2.set_title(r"(b) Universal finite-$\beta$ sharpening")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.18)

    fig.suptitle(
        r"Spin-boson canonical-channel criticality from exact mean-force closure ($\omega_q=\omega_c=1$)",
        fontsize=10.1,
    )

    out_pdf = OUT_DIR / "hmf_spin_boson_phase_diagram.pdf"
    out_png = OUT_DIR / "hmf_spin_boson_phase_diagram.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
