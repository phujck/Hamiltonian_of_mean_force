"""Generate angle-dependent closure criticality diagnostics for the spin-boson example.

This figure addresses how coupling angle changes the closure-level sign-inversion threshold.

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_angle_phase_map.pdf
  paper_dual_hmf/manuscript/figures/hmf_spin_boson_angle_phase_map.png
  paper_dual_hmf/manuscript_v2/figures/hmf_spin_boson_angle_phase_map.pdf
  paper_dual_hmf/manuscript_v2/figures/hmf_spin_boson_angle_phase_map.png
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIRS = [
    ROOT / "paper_dual_hmf" / "manuscript" / "figures",
    ROOT / "paper_dual_hmf" / "manuscript_v2" / "figures",
]
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)


def _load_core_module():
    script = ROOT / "paper_dual_hmf" / "validation" / "make_core_dual_map_figure.py"
    spec = importlib.util.spec_from_file_location("hmf_core_dual_map", str(script))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def critical_g_from_closure(beta: float, theta: float, core) -> tuple[float, float, float]:
    """Return (g_c, q, chi0) for closure criterion.

    Uses m_z = 0 condition:
      tanh(g^2 chi0) * q = tanh(beta*omega_q/2),
    with q := D0/chi0, D0 ≡ s^2 Sigma_z^(0).
    """
    chi0, dz0, _sx0 = core.get_chi0(float(beta), theta=float(theta))
    if not np.isfinite(chi0) or chi0 <= 0.0:
        return np.nan, np.nan, chi0
    q = float(dz0 / chi0)
    rhs = float(np.tanh(beta * core.OMEGA_Q / 2.0))
    if (not np.isfinite(q)) or (q <= 0.0) or (rhs >= q):
        return np.nan, q, chi0
    z = rhs / q
    if z >= 1.0:
        return np.nan, q, chi0
    g2 = float(np.arctanh(z) / chi0)
    if g2 <= 0.0:
        return np.nan, q, chi0
    return float(np.sqrt(g2)), q, chi0


def main() -> None:
    core = _load_core_module()

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "figure.dpi": 220,
        }
    )

    theta_grid = np.linspace(np.deg2rad(8.0), np.deg2rad(90.0), 110)
    theta_deg = np.rad2deg(theta_grid)
    beta_values = [1.5, 2.5, 4.0, 6.0]
    colors = ["#1f4e79", "#0b8c8c", "#8a3fa0", "#b30000"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3), constrained_layout=True)

    for b, col in zip(beta_values, colors):
        q_vals = []
        g_vals = []
        for th in theta_grid:
            g_c, q, _chi0 = critical_g_from_closure(b, float(th), core)
            q_vals.append(q)
            g_vals.append(g_c)
        q_vals = np.asarray(q_vals, dtype=float)
        g_vals = np.asarray(g_vals, dtype=float)
        rhs = np.tanh(b * core.OMEGA_Q / 2.0)

        ax1.plot(theta_deg, q_vals, color=col, lw=1.6, label=rf"$\beta\omega_q={b:.1f}$")
        ax1.hlines(rhs, theta_deg.min(), theta_deg.max(), color=col, lw=0.9, ls="--", alpha=0.8)

        mask = np.isfinite(g_vals) & (g_vals <= 4.0)
        ax2.plot(theta_deg[mask], g_vals[mask], color=col, lw=1.7, label=rf"$\beta\omega_q={b:.1f}$")
        invalid = ~np.isfinite(g_vals)
        if np.any(invalid):
            first_invalid = np.argmax(invalid)
            if first_invalid > 0:
                th_c = theta_deg[first_invalid - 1]
                ax2.plot([th_c], [g_vals[first_invalid - 1]], marker="o", ms=2.8, color=col)

    ax1.set_xlim(theta_deg.min(), theta_deg.max())
    ax1.set_ylim(0.0, 1.03)
    ax1.set_xlabel(r"coupling angle $\theta$ (deg)")
    ax1.set_ylabel(r"$Q_\theta:=D_0/\chi_0$")
    ax1.set_title(r"(a) Existence ratio vs angle")
    ax1.text(0.04, 0.09, r"dashed: $\tanh(\beta\omega_q/2)$", transform=ax1.transAxes, fontsize=7, color="0.3")
    ax1.axvline(90.0, color="0.25", lw=1.0, ls=":")
    ax1.text(0.67, 0.93, r"canonical $\theta=\pi/2$", transform=ax1.transAxes, fontsize=7, color="0.25")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="lower right", framealpha=0.9)

    ax2.set_xlim(theta_deg.min(), theta_deg.max())
    ax2.set_ylim(0.0, 4.05)
    ax2.set_xlabel(r"coupling angle $\theta$ (deg)")
    ax2.set_ylabel(r"closure threshold $g_c(\beta,\theta)$")
    ax2.set_title(r"(b) Finite-$\beta$ angle-dependent threshold")
    ax2.axvline(90.0, color="0.25", lw=1.0, ls=":")
    ax2.text(0.05, 0.92, "no finite solution region\n(not plotted)", transform=ax2.transAxes, fontsize=7, color="0.35")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Ohmic closure criticality becomes angle-selective away from the canonical channel", fontsize=10.1)

    for out_dir in OUT_DIRS:
        out_pdf = out_dir / "hmf_spin_boson_angle_phase_map.pdf"
        out_png = out_dir / "hmf_spin_boson_angle_phase_map.png"
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=240, bbox_inches="tight")
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
