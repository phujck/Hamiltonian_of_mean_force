"""
plot_gsweep_hub_branch_split_test.py
====================================
Diagnostic branch-switch test for the fixed-beta / g-sweep geometry (Fig. 7a style)
without the sqrt(mx) fold.

Purpose
-------
This is an interpretive branch-family test, not an exact equilibrium trajectory.
For three larger beta values we:
  1) keep the exact qubit radius r(g; beta),
  2) impose a hub crossing at phi = pi-theta before g = g_*,
  3) show two post-hub continuations:
       - groundward continuation
       - sign-flipped / alternate branch toward the theta sector

This directly tests the proposed "two-way continuation after the hub" picture in
the same visual language as Fig. 7(a), but on the unfolded Bloch circle
(signed transverse component vs m_z).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulations.production import plot_bloch_branch_flip_test as bd


FIGURES_DIR = ROOT / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 10,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "figure.dpi": 220,
            "lines.linewidth": 1.3,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    xx = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return xx * xx * (3.0 - 2.0 * xx)


def _polar_to_plane(r: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # alpha measured from -z toward +m_perp (same convention as Section 14 branch plots)
    return r * np.sin(alpha), -r * np.cos(alpha)


def _curve_arrow(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, frac: float, color: str, lw: float = 0.9) -> None:
    if len(x) < 3:
        return
    i1 = int(np.clip(round(frac * (len(x) - 1)), 1, len(x) - 2))
    i0 = max(0, i1 - 2)
    ax.annotate(
        "",
        xy=(float(x[i1]), float(y[i1])),
        xytext=(float(x[i0]), float(y[i0])),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=10,
    )


def _build_exact_radius_trace(beta: float, n_g: int, xi_max: float) -> dict[str, np.ndarray | float]:
    chi0, _, _ = bd.get_channels0(beta, theta=bd.THETA_VAL)
    g_star = 1.0 / np.sqrt(chi0)
    xi = np.linspace(0.0, xi_max, n_g)
    g_arr = xi * g_star

    pts = [bd.build_branch_point(float(g), float(x), float(beta)) for g, x in zip(g_arr, xi)]
    r_exact = np.array([p.r for p in pts], dtype=float)
    mx_exact = np.array([p.mx for p in pts], dtype=float)
    mz_exact = np.array([p.mz for p in pts], dtype=float)
    mperp_exact = np.array([p.m_perp_signed for p in pts], dtype=float)
    phi_wrap = np.array([p.phi_wrap for p in pts], dtype=float)
    phi_cont = np.unwrap(phi_wrap)

    return {
        "beta": float(beta),
        "g_star": float(g_star),
        "xi": xi,
        "g": g_arr,
        "r_exact": r_exact,
        "mx_exact": mx_exact,
        "mz_exact": mz_exact,
        "mperp_exact": mperp_exact,
        "phi_cont": phi_cont,
    }


def _build_branch_test_family(
    beta: float,
    xi_hub: float = 0.70,
    xi_max: float = 6.0,
    n_g: int = 1400,
) -> dict[str, np.ndarray | float]:
    data = _build_exact_radius_trace(beta=beta, n_g=n_g, xi_max=xi_max)

    theta = float(bd.THETA_VAL)
    alpha_theta = theta
    alpha_hub = float(np.pi - theta)
    alpha_2theta = float(2.0 * theta)

    xi = np.asarray(data["xi"], dtype=float)
    r_exact = np.asarray(data["r_exact"], dtype=float)
    mask_pre = xi <= xi_hub
    mask_post = xi >= xi_hub

    u_pre = np.zeros_like(xi)
    u_pre[mask_pre] = np.clip(xi[mask_pre] / max(xi_hub, 1e-12), 0.0, 1.0)
    t_post = np.zeros_like(xi)
    t_post[mask_post] = np.clip((xi[mask_post] - xi_hub) / max(xi_max - xi_hub, 1e-12), 0.0, 1.0)
    s_pre = _smoothstep(u_pre)
    s_post = _smoothstep(t_post)

    # Incoming branch (test construction): from the bare ground-state ray to the hub.
    alpha_in = np.full_like(xi, np.nan, dtype=float)
    alpha_in[mask_pre] = (1.0 - s_pre[mask_pre]) * 0.0 + s_pre[mask_pre] * alpha_hub

    # Two post-hub continuations:
    # 1) groundward branch returns to the bare ground-state ray,
    # 2) alternate branch starts at the local sign image (2theta) and precesses to theta.
    alpha_ground = np.full_like(xi, np.nan, dtype=float)
    alpha_ground[mask_post] = (1.0 - s_post[mask_post]) * alpha_hub + s_post[mask_post] * 0.0
    alpha_alt = np.full_like(xi, np.nan, dtype=float)
    alpha_alt[mask_post] = (1.0 - s_post[mask_post]) * alpha_2theta + s_post[mask_post] * alpha_theta

    x_in, z_in = _polar_to_plane(r_exact, alpha_in)
    x_g, z_g = _polar_to_plane(r_exact, alpha_ground)
    x_alt, z_alt = _polar_to_plane(r_exact, alpha_alt)

    # Hub / image points at xi = xi_hub using the exact radius at the split.
    idx_hub = int(np.argmin(np.abs(xi - xi_hub)))
    r_hub = float(r_exact[idx_hub])
    x_hub, z_hub = _polar_to_plane(np.array([r_hub]), np.array([alpha_hub]))
    x_img, z_img = _polar_to_plane(np.array([r_hub]), np.array([alpha_2theta]))

    data.update(
        {
            "theta": alpha_theta,
            "alpha_hub": alpha_hub,
            "alpha_2theta": alpha_2theta,
            "xi_hub": float(xi_hub),
            "mask_pre": mask_pre,
            "mask_post": mask_post,
            "alpha_in": alpha_in,
            "alpha_ground": alpha_ground,
            "alpha_alt": alpha_alt,
            "x_in": x_in,
            "z_in": z_in,
            "x_ground": x_g,
            "z_ground": z_g,
            "x_alt": x_alt,
            "z_alt": z_alt,
            "idx_hub": idx_hub,
            "x_hub": float(x_hub[0]),
            "z_hub": float(z_hub[0]),
            "x_img": float(x_img[0]),
            "z_img": float(z_img[0]),
        }
    )
    return data


def make_figure() -> tuple[Path, Path]:
    betas = [4.0, 5.0, 6.0]
    colors = ["#7a6aac", "#5b88c5", "#2b7bcb"]
    linestyles = ["-", "--", "-."]
    xi_hub = 0.70
    xi_max = 6.0

    traces = [_build_branch_test_family(beta=b, xi_hub=xi_hub, xi_max=xi_max) for b in betas]

    fig, ax = plt.subplots(figsize=(6.9, 4.35))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.14)

    # Bloch circle boundary and light fill
    t = np.linspace(0.0, 2.0 * np.pi, 700)
    ax.plot(np.cos(t), np.sin(t), color="#222222", lw=1.1, zorder=1)
    ax.fill(np.cos(t), np.sin(t), color="#f7f7f7", zorder=0)
    ax.axhline(0.0, color="k", ls=":", lw=0.45, alpha=0.25, zorder=1)
    ax.axvline(0.0, color="k", ls=":", lw=0.45, alpha=0.25, zorder=1)

    # Reference rays theta, pi-theta, and 2theta
    theta = float(bd.THETA_VAL)
    ray_specs = [
        (theta, "#8b2a2a", "-.", r"$\theta$"),
        (np.pi - theta, "#8b2a2a", "-", r"$\pi-\theta$"),
        (2.0 * theta, "#c17c2d", "--", r"$2\theta$"),
    ]
    rr = np.linspace(0.0, 1.03, 2)
    for ang, col, ls, _ in ray_specs:
        ax.plot(rr * np.sin(ang), -rr * np.cos(ang), color=col, ls=ls, lw=0.8, alpha=0.75, zorder=1)

    # Small angle labels placed once (to avoid clutter)
    ax.text(0.77 * np.sin(theta) + 0.01, -0.77 * np.cos(theta) - 0.03, r"$\theta$", color="#8b2a2a", fontsize=7.4)
    ax.text(0.80 * np.sin(np.pi - theta) + 0.01, -0.80 * np.cos(np.pi - theta) + 0.01, r"$\pi-\theta$", color="#8b2a2a", fontsize=7.4)
    ax.text(0.58 * np.sin(2.0 * theta) + 0.01, -0.58 * np.cos(2.0 * theta) + 0.01, r"$2\theta$", color="#9a5d14", fontsize=7.1)

    beta_handles = []
    for tr, color, ls, beta in zip(traces, colors, linestyles, betas):
        xi = np.asarray(tr["xi"])
        mask_pre = np.asarray(tr["mask_pre"])
        mask_post = np.asarray(tr["mask_post"])

        # Faint exact context (same beta g-sweep, unfolded, no sqrt fold)
        ax.plot(np.asarray(tr["mperp_exact"]), np.asarray(tr["mz_exact"]),
                color=color, lw=0.7, ls=":", alpha=0.25, zorder=2)

        # Branch-test family (incoming + two post-hub continuations)
        ax.plot(np.asarray(tr["x_in"])[mask_pre], np.asarray(tr["z_in"])[mask_pre],
                color=color, lw=1.6, ls=ls, zorder=4)
        ax.plot(np.asarray(tr["x_ground"])[mask_post], np.asarray(tr["z_ground"])[mask_post],
                color=color, lw=1.6, ls=ls, zorder=5)
        ax.plot(np.asarray(tr["x_alt"])[mask_post], np.asarray(tr["z_alt"])[mask_post],
                color=color, lw=1.6, ls=(0, (4, 2)), zorder=5)

        # Sign-image link at the hub (same radius, alternate sign/image branch)
        ax.plot([float(tr["x_hub"]), float(tr["x_img"])], [float(tr["z_hub"]), float(tr["z_img"])],
                color=color, lw=0.8, ls="--", alpha=0.75, zorder=4)

        # Markers: g=0, g_hub, g=g*, and asymptotic endpoints of the two continuations
        idx0 = 0
        idx_star = int(np.argmin(np.abs(xi - 1.0)))
        ax.plot(float(tr["x_in"][idx0]), float(tr["z_in"][idx0]), "o", ms=4.8, color=color,
                mec="white", mew=0.6, zorder=7)
        ax.plot(float(tr["x_hub"]), float(tr["z_hub"]), marker="D", ms=4.7, color=color,
                mec="white", mew=0.55, zorder=7)

        # Two possible branch locations at g=g* after the pre-g* hub split.
        if float(tr["xi_hub"]) < 1.0:
            ax.plot(float(tr["x_ground"][idx_star]), float(tr["z_ground"][idx_star]), marker="*", ms=8.0,
                    color=color, mec="white", mew=0.55, zorder=7)
            ax.plot(float(tr["x_alt"][idx_star]), float(tr["z_alt"][idx_star]), marker="*", ms=8.0,
                    color=color, mec="white", mew=0.55, mfc="none", zorder=7)

        # Asymptotic endpoints (shared across beta for this fixed-theta branch family)
        xg_inf, zg_inf = _polar_to_plane(np.array([1.0]), np.array([0.0]))
        xf_inf, zf_inf = _polar_to_plane(np.array([1.0]), np.array([theta]))
        ax.plot(float(xg_inf[0]), float(zg_inf[0]), "s", ms=4.0, color=color, mec="white", mew=0.45, zorder=7)
        ax.plot(float(xf_inf[0]), float(zf_inf[0]), "s", ms=4.0, color=color, mec="white", mew=0.45, mfc="none", zorder=7)

        # Direction arrows
        _curve_arrow(ax, np.asarray(tr["x_in"])[mask_pre], np.asarray(tr["z_in"])[mask_pre], 0.78, color)
        _curve_arrow(ax, np.asarray(tr["x_ground"])[mask_post], np.asarray(tr["z_ground"])[mask_post], 0.50, color)
        _curve_arrow(ax, np.asarray(tr["x_alt"])[mask_post], np.asarray(tr["z_alt"])[mask_post], 0.50, color)

        beta_handles.append(mpl.lines.Line2D([], [], color=color, ls=ls, lw=1.6, label=rf"$\beta\omega_q={beta:g}$"))

    # Legends: beta values + branch/marker semantics
    leg1 = ax.legend(handles=beta_handles, loc="upper left", bbox_to_anchor=(0.01, 0.995),
                     framealpha=0.92, ncol=1, handlelength=2.1)
    ax.add_artist(leg1)

    semantic_handles = [
        mpl.lines.Line2D([], [], color="#444444", ls=":", lw=0.8, alpha=0.55, label=r"exact context"),
        mpl.lines.Line2D([], [], color="#444444", lw=1.6, label=r"hub branch family"),
        mpl.lines.Line2D([], [], color="#444444", lw=1.6, ls=(0, (4, 2)), label=r"alternate continuation"),
        mpl.lines.Line2D([], [], marker="o", color="k", ls="none", ms=4.5, mec="white", mew=0.5, label=r"$g=0$"),
        mpl.lines.Line2D([], [], marker="D", color="k", ls="none", ms=4.3, mec="white", mew=0.5, label=rf"$g_{{\rm hub}}={xi_hub:.1f}\,g_\star$"),
        mpl.lines.Line2D([], [], marker="*", color="k", ls="none", ms=7.5, mec="white", mew=0.5, label=r"$g=g_\star$ (filled/open = two branches)"),
    ]
    ax.legend(handles=semantic_handles, loc="lower left", bbox_to_anchor=(0.01, 0.01),
              framealpha=0.92, fontsize=6.2, handlelength=2.2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$m_\perp^{(\mathrm{signed})}$")
    ax.set_ylabel(r"$m_z$")
    ax.grid(True, alpha=0.10, lw=0.35)
    ax.text(0.05, 0.94, r"(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10)

    png = FIGURES_DIR / "hmf_gsweep_hub_branch_split_test.png"
    pdf = FIGURES_DIR / "hmf_gsweep_hub_branch_split_test.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")
    return png, pdf


def main() -> None:
    _configure_matplotlib()
    make_figure()


if __name__ == "__main__":
    main()
