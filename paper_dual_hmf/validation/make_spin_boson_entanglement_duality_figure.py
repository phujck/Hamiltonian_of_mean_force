"""Generate spin-boson entropy/duality illustration for the manuscript.

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_entropy_duality.pdf
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_entropy_duality.png
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as quad


warnings.filterwarnings("ignore", category=quad.IntegrationWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_dual_hmf" / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Same model constants as the core dual-map figure for consistency.
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
        val = ch_a / sh * (b**2 / 4.0 * ch_a + b / (2.0 * oq) * sh_a)
        val -= b * np.cosh(np.clip(bOm2, 0, 500)) / sh * ch_a / oq
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


def entropy_bits_from_chi(chi: np.ndarray) -> np.ndarray:
    """Qubit von Neumann entropy in bits from chi."""
    r = np.abs(np.tanh(chi))
    lp = 0.5 * (1.0 + r)
    lm = 0.5 * (1.0 - r)
    tiny = np.finfo(float).tiny
    lp = np.clip(lp, tiny, 1.0)
    lm = np.clip(lm, tiny, 1.0)
    return -(lp * np.log2(lp) + lm * np.log2(lm))


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
    g_mult_grid = np.geomspace(0.12, 8.0, 120)
    g_grid = g_mult_grid * gstar_ref

    chi_map = np.outer(g_grid**2, chi0_grid)
    S_map_bits = entropy_bits_from_chi(chi_map)

    chi0_b0 = get_chi0(beta0, theta=theta)[0]
    gstar_b0 = 1.0 / np.sqrt(chi0_b0)
    x = np.geomspace(0.4, 8.0, 280)  # x := g / g*(beta0)
    g_vals = x * gstar_b0

    chi_vals = chi0_b0 * g_vals**2
    S_direct = entropy_bits_from_chi(chi_vals)

    g_dual = 1.0 / (chi0_b0 * g_vals)
    chi_dual = chi0_b0 * g_dual**2  # = 1/chi
    S_dual_partner = entropy_bits_from_chi(chi_dual)  # entropy at g^vee (not equal to S_direct)

    # Strong-branch asymptotic written in the dual weak variable chi_dual -> 0.
    S_strong_from_dual = ((1.0 + 2.0 / chi_dual) * np.exp(-2.0 / chi_dual)) / np.log(2.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3), constrained_layout=True)

    B, Gm = np.meshgrid(beta_grid, g_mult_grid)
    pcm = ax1.pcolormesh(B, Gm, S_map_bits, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(pcm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"$S_Q$ (bits)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    chi_map_log = np.log(np.clip(chi_map, 1e-14, None))
    ax1.contour(B, Gm, chi_map_log, levels=[0.0], colors=["white"], linewidths=1.2)
    gstar_beta = 1.0 / np.sqrt(np.clip(chi0_grid, 1e-20, None))
    gstar_rel = gstar_beta / gstar_ref
    gmin = g_mult_grid.min()
    gmax = g_mult_grid.max()
    gstar_plot = np.where((gstar_rel >= gmin) & (gstar_rel <= gmax), gstar_rel, np.nan)
    ax1.plot(beta_grid, gstar_plot, color="0.1", lw=0.9, label=r"$\chi=1$")
    ax1.plot([beta0, beta0], [g_grid.min() / gstar_ref, g_grid.max() / gstar_ref], color="white", lw=1.0, ls="--")
    ax1.text(0.62, 0.15, r"$\beta=\beta_0$", color="white", fontsize=7, transform=ax1.transAxes)
    ax1.text(0.06, 0.90, r"weak branch", color="white", fontsize=7, transform=ax1.transAxes)
    ax1.text(0.06, 0.07, r"strong branch", color="white", fontsize=7, transform=ax1.transAxes)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(gmin, gmax)
    ax1.set_xlabel(r"$\beta\,\omega_q$")
    ax1.set_ylabel(r"$g/g^\star_{\mathrm{ref}}$")
    ax1.set_title(r"(a) Reduced-state entropy map")
    ax1.legend(loc="upper right", framealpha=0.85)

    ax2.axvspan(x.min(), 1.0, color="#dbe9ff", alpha=0.5, lw=0)
    ax2.axvspan(1.0, x.max(), color="#ffe3e3", alpha=0.45, lw=0)
    ax2.plot(x, S_direct, color="black", lw=1.7, label=r"direct $S_Q(g)$")
    ax2.plot(x, S_dual_partner, color="#cc0000", lw=1.3, ls="--", label=r"partner $S_Q(g^\vee)$")

    mask = x >= 1.2
    ax2.plot(
        x[mask],
        S_strong_from_dual[mask],
        color="#0b5394",
        lw=1.4,
        ls=":",
        label=r"strong asymptotic via $\chi^\vee$",
    )
    ax2.axvline(1.0, color="0.35", lw=1.0, ls="--")
    ax2.text(1.03, 0.92, r"self-dual $g=g^\star$", fontsize=7, color="0.2")

    ax2.set_xscale("log")
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel(r"$g/g^\star(\beta_0)$, $\beta_0=4$")
    ax2.set_ylabel(r"$S_Q$ (bits)")
    ax2.set_title(r"(b) Dual-entropy diagnostics on $\beta=\beta_0$")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Entropy geometry under strong-weak representative mapping", fontsize=10.2)

    out_pdf = OUT_DIR / "hmf_spin_boson_entropy_duality.pdf"
    out_png = OUT_DIR / "hmf_spin_boson_entropy_duality.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
