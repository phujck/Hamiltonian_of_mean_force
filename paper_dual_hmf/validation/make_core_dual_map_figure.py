"""Generate the core theory-led dual-map validation figure for paper_dual_hmf.

This script is self-contained by design. It avoids importing production figure modules
that may execute plotting side effects on import.

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_dual_map_core.pdf
  paper_dual_hmf/manuscript/figures/hmf_dual_map_core.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import scipy.integrate as quad


warnings.filterwarnings("ignore", category=quad.IntegrationWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_dual_hmf" / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Self-contained physical parameters
OMEGA_Q = 1.0
ALPHA = 1.0
OMEGA_C = 5.0
OMEGA_MIN = 0.04
OMEGA_MAX = 2.0
THETA = np.pi / 4.0

_CH_CACHE: dict[tuple[float, float], tuple[float, float, float]] = {}


def J_ohmic(w: float | np.ndarray, alpha: float = ALPHA, omega_c: float = OMEGA_C):
    arr = np.asarray(w, dtype=float)
    return alpha * arr * np.exp(-arr / omega_c)


def _F_z(Omega: float, beta: float, omega_q: float = OMEGA_Q) -> float:
    b = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a = b * oq / 2.0

    eps = abs(Om - oq)
    if eps < 1e-6 * max(oq, 1.0):
        bOm2 = b * oq / 2.0
        sh = np.sinh(np.clip(bOm2, 1e-14, 500))
        ch_a = np.cosh(np.clip(a, 0, 500))
        sh_a = np.sinh(np.clip(a, 0, 500))
        val = ch_a / sh * (
            b**2 / 4.0 * ch_a
            + b / (2.0 * oq) * sh_a
        ) - b * np.cosh(np.clip(bOm2, 0, 500)) / sh * ch_a / oq
        return float(val)

    Om_plus = Om + oq
    Om_minus = oq - Om

    bOm2 = b * Om / 2.0
    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    ch_a = np.cosh(np.clip(a, 0, 500))

    denom = oq**2 - Om**2
    term1 = -b * oq * ch_Om / denom
    term2 = (ch_a / sh_Om) * (
        np.sinh(np.clip(Om_plus * b / 2.0, 0, 500)) / Om_plus**2
        + np.sinh(np.clip(Om_minus * b / 2.0, 0, 500)) / Om_minus**2
    )
    return float(term1 + term2)


def _F_x(Omega: float, beta: float, omega_q: float = OMEGA_Q) -> float:
    b = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a = b * oq / 2.0

    eps = abs(Om - oq)
    bOm2 = b * Om / 2.0
    ch_a = np.cosh(np.clip(a, 0, 500))
    sh_a = np.sinh(np.clip(a, 0, 500))

    if abs(Om) < 1e-12:
        part1 = 2.0 * ch_a**2 * b
    else:
        sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
        part1 = 4.0 * ch_a**2 * sh_Om / Om

    if eps < 1e-6 * max(oq, 1.0):
        dOm = 1e-5 * max(oq, 1.0)
        return 0.5 * (_F_x(oq + dOm, beta, omega_q) + _F_x(oq - dOm, beta, omega_q))

    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    denom = Om**2 - oq**2
    part2 = 4.0 * ch_a * (Om * sh_Om * ch_a - oq * ch_Om * sh_a) / denom
    return float(part1 - part2)


def chi0_spectral(beta: float, theta: float = THETA):
    s = np.sin(theta)
    c = np.cos(theta)

    def integrand_z(Om: float) -> float:
        if Om < 1e-12:
            return 0.0
        return float(J_ohmic(Om) * _F_z(Om, beta))

    def integrand_x(Om: float) -> float:
        if Om < 1e-12:
            return 0.0
        return float(J_ohmic(Om) * _F_x(Om, beta))

    lo = max(OMEGA_MIN, 1e-10)
    mid = OMEGA_Q
    hi = OMEGA_MAX

    dz_lo, _ = quad.quad(integrand_z, lo, mid, limit=300, epsrel=1e-10)
    dz_hi, _ = quad.quad(integrand_z, mid, hi, limit=300, epsrel=1e-10)
    dz0 = (s**2 / np.pi) * (dz_lo + dz_hi)

    sx_lo, _ = quad.quad(integrand_x, lo, mid, limit=300, epsrel=1e-10)
    sx_hi, _ = quad.quad(integrand_x, mid, hi, limit=300, epsrel=1e-10)
    sx0 = (c * s / (np.pi * OMEGA_Q)) * (sx_lo + sx_hi)

    chi0 = float(np.sqrt(max(dz0 * dz0 + sx0 * sx0, 0.0)))
    return float(chi0), float(dz0), float(sx0)


def get_chi0(beta: float, theta: float = THETA):
    key = (float(beta), float(theta))
    if key not in _CH_CACHE:
        _CH_CACHE[key] = chi0_spectral(beta, theta=theta)
    return _CH_CACHE[key]


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

    theta = THETA
    beta_ref = 2.0
    beta0 = 4.0

    beta_grid = np.geomspace(0.08, 20.0, 100)
    chi0_grid = np.array([get_chi0(float(b), theta=theta)[0] for b in beta_grid])

    chi0_ref = get_chi0(beta_ref, theta=theta)[0]
    gstar_ref = 1.0 / np.sqrt(chi0_ref)

    g_mult_grid = np.geomspace(0.08, 8.0, 120)
    g_grid = g_mult_grid * gstar_ref

    chi_map = np.outer(g_grid**2, chi0_grid)
    y_map = np.log(np.clip(chi_map, 1e-14, None))

    chi0_b0 = get_chi0(beta0, theta=theta)[0]
    gstar_b0 = 1.0 / np.sqrt(chi0_b0)
    g_vals = np.geomspace(0.8 * gstar_b0, 8.0 * gstar_b0, 260)
    chi_vals = chi0_b0 * g_vals**2

    gamma_exact = np.tanh(chi_vals) / chi_vals
    gamma_strong = 1.0 / chi_vals

    g_dual = 1.0 / (chi0_b0 * g_vals)
    chi_dual = chi0_b0 * g_dual**2
    gamma_dual_weak_recon = chi_dual

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3), constrained_layout=True)

    B, Gm = np.meshgrid(beta_grid, g_mult_grid)
    y_clip = np.clip(y_map, -3.0, 3.0)
    pcm = ax1.pcolormesh(
        B,
        Gm,
        y_clip,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0),
    )
    cbar = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"$y=\log\chi$")
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    ax1.contour(B, Gm, y_map, levels=[0.0], colors=["black"], linewidths=1.4)
    gstar_beta = 1.0 / np.sqrt(np.clip(chi0_grid, 1e-20, None))
    gstar_rel = gstar_beta / gstar_ref
    gmin = g_mult_grid.min()
    gmax = g_mult_grid.max()
    gstar_plot = np.where((gstar_rel >= gmin) & (gstar_rel <= gmax), gstar_rel, np.nan)
    ax1.plot(beta_grid, gstar_plot, color="white", lw=2.0, alpha=0.9)
    ax1.plot(beta_grid, gstar_plot, color="0.2", lw=1.0, label=r"self-dual contour: $\chi=1$")

    g_rep = 3.0 * gstar_b0
    g_rep_dual = 1.0 / (chi0_b0 * g_rep)
    ax1.plot([beta0], [g_rep / gstar_ref], "o", color="#0b5394", ms=4, label=r"$g$ (strong)")
    ax1.plot([beta0], [g_rep_dual / gstar_ref], "o", color="#cc0000", ms=4, label=r"$g^\vee$ (weak)")
    ax1.annotate(
        "",
        xy=(beta0, g_rep_dual / gstar_ref),
        xytext=(beta0, g_rep / gstar_ref),
        arrowprops=dict(arrowstyle="<->", color="0.1", lw=1.0, linestyle="--"),
    )
    ax1.text(beta0 * 1.06, np.sqrt((g_rep / gstar_ref) * (g_rep_dual / gstar_ref)), r"$g\leftrightarrow g^\vee$", fontsize=7)
    ax1.text(0.11, 0.89, r"weak branch ($\chi<1$)", transform=ax1.transAxes, fontsize=7, color="#225ea8")
    ax1.text(0.60, 0.08, r"strong branch ($\chi>1$)", transform=ax1.transAxes, fontsize=7, color="#a50f15")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(gmin, gmax)
    ax1.set_xlabel(r"$\beta\,\omega_q$")
    ax1.set_ylabel(r"$g/g^\star_{\mathrm{ref}}$")
    ax1.set_title(r"(a) Representative map in $y=\log\chi$")
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.14)

    x = g_vals / gstar_b0
    ax2.axvspan(x.min(), 1.0, color="#dbe9ff", alpha=0.5, lw=0)
    ax2.axvspan(1.0, x.max(), color="#ffe3e3", alpha=0.45, lw=0)
    ax2.plot(x, gamma_exact, color="black", lw=1.5, label=r"exact $\gamma(\chi)$")
    mask_strong = x >= 1.0
    ax2.plot(x[mask_strong], gamma_strong[mask_strong], color="#0b5394", lw=1.4, ls="--", label=r"strong asymptotic $1/\chi$")
    ax2.plot(
        x[mask_strong],
        gamma_dual_weak_recon[mask_strong],
        color="#cc0000",
        lw=1.2,
        ls=":",
        label=r"dual weak reconstruction $\chi^\vee$",
    )
    ax2.axvline(1.0, color="0.2", lw=1.0, ls="-")
    ax2.text(1.05, 0.91, r"$\chi=1$", fontsize=7, color="0.2")

    ax2.set_xscale("log")
    ax2.set_xlim(0.8, 8.0)
    ax2.xaxis.set_major_locator(FixedLocator([1.0, 2.0, 4.0, 8.0]))
    ax2.xaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8"]))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(r"$g/g^\star(\beta_0)$, $\beta_0=4$")
    ax2.set_ylabel(r"observable value")
    ax2.set_ylim(0.0, 1.03)
    ax2.set_title(r"(b) Strong-direct vs dual-weak reconstruction")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, which="both", alpha=0.2)

    fig.suptitle("Core duality validation: branch exchange by representative transformation", fontsize=10.2)

    out_pdf = OUT_DIR / "hmf_dual_map_core.pdf"
    out_png = OUT_DIR / "hmf_dual_map_core.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
