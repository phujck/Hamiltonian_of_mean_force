"""
plot_tilt_sign_mfg_branch_family.py
===================================
Publication-facing figure for the hub-centered branch/sign multiplicity story in
`14_tilt_sign_branch_mfg_alignment.tex`.

This is an analytic/interpretive construction (not an exact equilibrium trace):
- radius is guided by the Cresser-Anders MFG form
- angular branches are a hub-centered branch-family ansatz around the axis-angle
  crossing at phi_ax = pi - theta
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulations.production.plot_bloch_branch_flip_test as base

FIGURES_DIR = ROOT / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 10,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "figure.dpi": 220,
            "lines.linewidth": 1.35,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
        }
    )


def _panel_label(ax: mpl.axes.Axes, lab: str) -> None:
    ax.text(0.04, 0.96, lab, transform=ax.transAxes, ha="left", va="top", fontsize=9)


def _polar_to_plane(r: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # plotting convention: alpha measured from -z toward +m_perp
    return r * np.sin(alpha), -r * np.cos(alpha)


def _curve_arrow(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, frac: float, color: str, lw: float = 0.8) -> None:
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


def build_hub_centered_mfg_branch_family(
    theta: float = np.pi / 6.0,
    beta_min: float = base.BETA_MIN,
    beta_max: float = base.BETA_MAX,
    n_beta: int = 480,
    beta_hub: float = 1.25,
    visual_mirror_z: bool = True,
) -> dict[str, np.ndarray | float]:
    beta = np.geomspace(beta_min, beta_max, n_beta)
    a = 0.5 * beta * base.OMEGA_Q
    c = float(np.cos(theta))

    # MFG-guided Bloch radius target and density-matrix purity target.
    r_mfg = np.tanh(a * c)
    P_mfg = 0.5 * (1.0 + r_mfg * r_mfg)

    # Semantic (manuscript) rays.
    alpha_theta = float(theta)
    alpha_hub = float(np.pi - theta)
    alpha_2theta = float(2.0 * theta)

    beta = np.asarray(beta, dtype=float)
    mask_pre = beta <= beta_hub
    mask_post = beta >= beta_hub

    # Use log-time coordinates to make the branch geometry smooth across decades.
    # pre:  beta_min -> beta_hub  maps to u in [0,1]
    # post: beta_hub -> beta_max maps to t in [0,1]
    u_pre = np.zeros_like(beta)
    if np.any(mask_pre):
        lp0 = np.log(beta_min)
        lp1 = np.log(beta_hub)
        u_pre[mask_pre] = (np.log(beta[mask_pre]) - lp0) / max(lp1 - lp0, 1e-12)
        u_pre[mask_pre] = np.clip(u_pre[mask_pre], 0.0, 1.0)

    t_post = np.zeros_like(beta)
    if np.any(mask_post):
        lq0 = np.log(beta_hub)
        lq1 = np.log(beta_max)
        t_post[mask_post] = (np.log(beta[mask_post]) - lq0) / max(lq1 - lq0, 1e-12)
        t_post[mask_post] = np.clip(t_post[mask_post], 0.0, 1.0)

    # Smoothstep easing for publication-friendly curves.
    def smoothstep(x: np.ndarray, p: float = 1.0) -> np.ndarray:
        xx = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        if p != 1.0:
            xx = xx**p
        return xx * xx * (3.0 - 2.0 * xx)

    s_pre = smoothstep(u_pre, p=0.90)
    s_post = smoothstep(t_post, p=0.95)

    # Apparent (axis-chart) incoming branch approaching the hub from the high-T side.
    # We keep it in [0,pi) and intentionally short in geometric extent.
    alpha_incoming = np.full_like(beta, np.nan, dtype=float)
    alpha_start = float(np.pi - 0.08)  # close to the top of the Bloch circle
    alpha_incoming[mask_pre] = (1.0 - s_pre[mask_pre]) * alpha_start + s_pre[mask_pre] * alpha_hub

    # Post-hub continuations on a lifted vector-angle chart:
    # 1) "groundward" branch reproducing the original apparent branch behavior.
    alpha_ground = np.full_like(beta, np.nan, dtype=float)
    alpha_ground[mask_post] = (1.0 - s_post[mask_post]) * alpha_hub + s_post[mask_post] * 0.0

    # 2) sign-flipped / alternate arctan branch, starting at the image point 2theta
    # and precessing back to theta (MFG-consistent direction).
    alpha_mfg = np.full_like(beta, np.nan, dtype=float)
    alpha_mfg[mask_post] = (1.0 - s_post[mask_post]) * alpha_2theta + s_post[mask_post] * alpha_theta

    # Optional display-convention mirror: alpha -> pi - alpha, equivalent to z -> -z.
    plot_alpha_theta = alpha_theta
    plot_alpha_hub = alpha_hub
    plot_alpha_2theta = alpha_2theta
    plot_alpha_incoming = alpha_incoming.copy()
    plot_alpha_ground = alpha_ground.copy()
    plot_alpha_mfg = alpha_mfg.copy()
    if visual_mirror_z:
        plot_alpha_theta = float(np.pi - alpha_theta)
        plot_alpha_hub = float(np.pi - alpha_hub)
        plot_alpha_2theta = float(np.pi - alpha_2theta)
        plot_alpha_incoming = np.where(np.isfinite(alpha_incoming), np.pi - alpha_incoming, np.nan)
        plot_alpha_ground = np.where(np.isfinite(alpha_ground), np.pi - alpha_ground, np.nan)
        plot_alpha_mfg = np.where(np.isfinite(alpha_mfg), np.pi - alpha_mfg, np.nan)

    # "Apparent" axis branch (what one sees in the folded/original picture) used in panel (b):
    # pre-hub it follows the incoming axis chart, post-hub it follows the groundward continuation.
    phi_ax_app = np.full_like(beta, np.nan, dtype=float)
    phi_ax_app[mask_pre] = np.mod(plot_alpha_incoming[mask_pre], np.pi)
    phi_ax_app[mask_post] = np.mod(plot_alpha_ground[mask_post], np.pi)

    # Coordinates on the Bloch circle plane for the branch-family ansatz.
    x_in, z_in = _polar_to_plane(r_mfg, plot_alpha_incoming)
    x_g, z_g = _polar_to_plane(r_mfg, plot_alpha_ground)
    x_f, z_f = _polar_to_plane(r_mfg, plot_alpha_mfg)

    # Hub and sign-flipped image points at beta=beta_hub.
    r_h = float(np.tanh(0.5 * beta_hub * base.OMEGA_Q * c))
    x_h, z_h = _polar_to_plane(np.array([r_h]), np.array([plot_alpha_hub]))
    x_flip, z_flip = _polar_to_plane(np.array([r_h]), np.array([plot_alpha_2theta]))

    return {
        "beta": beta,
        "a": a,
        "theta": float(theta),
        "alpha_theta": alpha_theta,
        "alpha_hub": alpha_hub,
        "alpha_2theta": alpha_2theta,
        "plot_alpha_theta": plot_alpha_theta,
        "plot_alpha_hub": plot_alpha_hub,
        "plot_alpha_2theta": plot_alpha_2theta,
        "beta_hub": float(beta_hub),
        "r_mfg": r_mfg,
        "P_mfg": P_mfg,
        "alpha_incoming": alpha_incoming,
        "alpha_ground": alpha_ground,
        "alpha_mfg": alpha_mfg,
        "plot_alpha_incoming": plot_alpha_incoming,
        "plot_alpha_ground": plot_alpha_ground,
        "plot_alpha_mfg": plot_alpha_mfg,
        "phi_ax_app": phi_ax_app,
        "x_in": x_in,
        "z_in": z_in,
        "x_g": x_g,
        "z_g": z_g,
        "x_f": x_f,
        "z_f": z_f,
        "x_hub": float(x_h[0]),
        "z_hub": float(z_h[0]),
        "x_flip": float(x_flip[0]),
        "z_flip": float(z_flip[0]),
        "mask_pre": mask_pre,
        "mask_post": mask_post,
        "visual_mirror_z": bool(visual_mirror_z),
    }


def make_figure() -> tuple[Path, Path]:
    d = build_hub_centered_mfg_branch_family()

    beta = d["beta"]
    mask_pre = d["mask_pre"]
    mask_post = d["mask_post"]

    fig = plt.figure(figsize=(7.05, 3.95))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.22, 1.02],
        height_ratios=[1.0, 0.98],
        wspace=0.34,
        hspace=0.23,
        left=0.07,
        right=0.985,
        bottom=0.14,
        top=0.975,
    )
    axA = fig.add_subplot(gs[:, 0])   # large Bloch-circle panel
    axB = fig.add_subplot(gs[0, 1])   # angle branches
    axC = fig.add_subplot(gs[1, 1])   # radius + purity target

    # ------------------------------------------------------------------
    # (a) Bloch circle + hub-centered branch family
    # ------------------------------------------------------------------
    t = np.linspace(0.0, 2.0 * np.pi, 700)
    axA.plot(np.cos(t), np.sin(t), color="#222222", lw=1.15)
    axA.axhline(0.0, color="k", ls=":", lw=0.45, alpha=0.25)
    axA.axvline(0.0, color="k", ls=":", lw=0.45, alpha=0.25)

    # Reference rays: theta, pi-theta, and the image point 2theta.
    rr = np.linspace(0.0, 1.0, 2)
    ray_specs = [
        (d["plot_alpha_theta"], "#8b2a2a", "-."),
        (d["plot_alpha_hub"], "#8b2a2a", "-"),
        (d["plot_alpha_2theta"], "#c17c2d", "--"),
    ]
    for ang, col, ls in ray_specs:
        axA.plot(rr * np.sin(ang), -rr * np.cos(ang), color=col, ls=ls, lw=0.8, alpha=0.75, zorder=1)

    # Incoming apparent branch (high T -> hub)
    incoming_color = "#2b7bcb"
    ground_color = "#1f4e99"
    alt_color = "#239b56"
    sign_color = "#c17c2d"
    hub_color = "#c62828"

    axA.plot(
        np.asarray(d["x_in"])[mask_pre],
        np.asarray(d["z_in"])[mask_pre],
        color=incoming_color,
        lw=1.85,
        zorder=4,
    )
    # Post-hub continuations
    axA.plot(
        np.asarray(d["x_g"])[mask_post],
        np.asarray(d["z_g"])[mask_post],
        color=ground_color,
        lw=1.9,
        zorder=5,
    )
    axA.plot(
        np.asarray(d["x_f"])[mask_post],
        np.asarray(d["z_f"])[mask_post],
        color=alt_color,
        lw=1.9,
        ls="--",
        zorder=5,
    )

    # Hub and sign-flipped image points
    x_h = float(d["x_hub"])
    z_h = float(d["z_hub"])
    x_flip = float(d["x_flip"])
    z_flip = float(d["z_flip"])
    axA.plot(x_h, z_h, marker="*", ms=8.8, color=hub_color, mec="white", mew=0.6, zorder=7)
    axA.plot(x_flip, z_flip, marker="o", ms=4.6, color=sign_color, mec="white", mew=0.5, zorder=7)

    # Sign-flip image jump at the hub (chart branch / sign choice, not exact dynamics).
    axA.plot([x_h, x_flip], [z_h, z_flip], color=sign_color, lw=0.8, ls="--", alpha=0.8, zorder=6)

    # Local right-angle cue between theta and pi-theta for theta=pi/4.
    ang_arc = np.linspace(float(d["plot_alpha_theta"]), float(d["plot_alpha_hub"]), 60)
    r_arc = 0.17
    axA.plot(r_arc * np.sin(ang_arc), -r_arc * np.cos(ang_arc), color="#8b2a2a", lw=0.7, alpha=0.65)

    # Labels and direction arrows (increasing beta along the displayed parameterization)
    _curve_arrow(axA, np.asarray(d["x_in"])[mask_pre], np.asarray(d["z_in"])[mask_pre], 0.82, incoming_color)
    _curve_arrow(axA, np.asarray(d["x_g"])[mask_post], np.asarray(d["z_g"])[mask_post], 0.50, ground_color)
    _curve_arrow(axA, np.asarray(d["x_f"])[mask_post], np.asarray(d["z_f"])[mask_post], 0.52, alt_color)

    # Minimal geometric labels only.
    theta_lab_r = 0.84
    hub_lab_r = 0.84
    twotheta_lab_r = 0.62
    axA.text(
        theta_lab_r * np.sin(float(d["plot_alpha_theta"])) + 0.02,
        -theta_lab_r * np.cos(float(d["plot_alpha_theta"])) - 0.03,
        r"$\theta$",
        fontsize=7.4,
        color="#8b2a2a",
    )
    axA.text(
        hub_lab_r * np.sin(float(d["plot_alpha_hub"])) + 0.02,
        -hub_lab_r * np.cos(float(d["plot_alpha_hub"])) + 0.01,
        r"$\pi-\theta$",
        fontsize=7.4,
        color="#8b2a2a",
    )
    axA.text(
        twotheta_lab_r * np.sin(float(d["plot_alpha_2theta"])) + 0.02,
        -twotheta_lab_r * np.cos(float(d["plot_alpha_2theta"])) + 0.01,
        r"$2\theta$",
        fontsize=7.1,
        color="#9a5d14",
    )

    # Compact legend rather than long in-panel prose.
    legend_handles = [
        mpl.lines.Line2D([], [], color=incoming_color, lw=1.85, label=r"incoming branch"),
        mpl.lines.Line2D([], [], color=ground_color, lw=1.9, label=r"groundward continuation"),
        mpl.lines.Line2D([], [], color=alt_color, lw=1.9, ls="--", label=r"MFG-directed continuation"),
        mpl.lines.Line2D([], [], color=hub_color, marker="*", ls="none", ms=8, label=r"hub"),
    ]
    axA.legend(handles=legend_handles, loc="lower left", framealpha=0.92, fontsize=6.3, handlelength=2.2)

    axA.set_aspect("equal", adjustable="box")
    axA.set_xlim(-1.05, 1.05)
    axA.set_ylim(-1.05, 1.05)
    axA.set_xlabel(r"$m_\perp^{(\mathrm{signed})}$")
    axA.set_ylabel(r"$m_z$")
    axA.grid(True, alpha=0.10, lw=0.35)
    _panel_label(axA, "(a)")

    # ------------------------------------------------------------------
    # (b) Angle-sheet picture: apparent axis angle and two lifted continuations
    # ------------------------------------------------------------------
    axB.plot(beta[mask_pre], np.asarray(d["plot_alpha_incoming"])[mask_pre], color=incoming_color, lw=1.5)
    axB.plot(beta[mask_post], np.asarray(d["plot_alpha_ground"])[mask_post], color=ground_color, lw=1.55)
    axB.plot(beta[mask_post], np.asarray(d["plot_alpha_mfg"])[mask_post], color=alt_color, lw=1.65, ls="--")
    axB.axvline(float(d["beta_hub"]), color="#c62828", ls=":", lw=0.9, alpha=0.85)
    axB.axhline(float(d["plot_alpha_hub"]), color="#8b2a2a", ls="-", lw=0.75, alpha=0.55)
    axB.axhline(float(d["plot_alpha_theta"]), color="#8b2a2a", ls="-.", lw=0.75, alpha=0.55)
    axB.axhline(float(d["plot_alpha_2theta"]), color="#c17c2d", ls="--", lw=0.75, alpha=0.55)
    axB.axhline(0.0, color="#666666", ls=":", lw=0.7, alpha=0.5)
    angle_handles = [
        mpl.lines.Line2D([], [], color=incoming_color, lw=1.5, label=r"incoming"),
        mpl.lines.Line2D([], [], color=ground_color, lw=1.55, label=r"groundward"),
        mpl.lines.Line2D([], [], color=alt_color, lw=1.65, ls="--", label=r"MFG-directed"),
        mpl.lines.Line2D([], [], color="#c62828", lw=0.9, ls=":", label=r"$\beta_{\rm hub}$"),
    ]
    y_all = np.concatenate(
        [
            np.asarray(d["phi_ax_app"])[np.isfinite(np.asarray(d["phi_ax_app"]))],
            np.asarray(d["plot_alpha_mfg"])[np.isfinite(np.asarray(d["plot_alpha_mfg"]))],
            np.asarray(d["plot_alpha_incoming"])[np.isfinite(np.asarray(d["plot_alpha_incoming"]))],
            np.asarray(d["plot_alpha_ground"])[np.isfinite(np.asarray(d["plot_alpha_ground"]))],
        ]
    )
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    axB.set_xscale("log")
    axB.set_xlabel(r"$\beta\omega_q$")
    axB.set_ylabel(r"Branch angle (rad)")
    axB.set_ylim(min(-0.10, y_min - 0.10), y_max + 0.12)
    tick_pairs = [
        (0.0, r"$0$"),
        (float(d["plot_alpha_theta"]), r"$\theta$"),
        (float(d["plot_alpha_2theta"]), r"$2\theta$"),
        (float(d["plot_alpha_hub"]), r"$\pi-\theta$"),
    ]
    tick_pairs.sort(key=lambda t: t[0])
    axB.set_yticks([p[0] for p in tick_pairs])
    axB.set_yticklabels([p[1] for p in tick_pairs])
    axB.grid(True, which="both", alpha=0.12, lw=0.35)
    axB.legend(handles=angle_handles, loc="lower left", framealpha=0.92, fontsize=6.3, handlelength=2.2)
    _panel_label(axB, "(b)")

    # ------------------------------------------------------------------
    # (c) MFG-guided Bloch radius and density-matrix purity target
    # ------------------------------------------------------------------
    axC.plot(beta, d["r_mfg"], color="#239b56", lw=1.75)
    axC.plot(beta, d["P_mfg"], color="#c17c2d", lw=1.45, ls="--")
    axC.axvline(float(d["beta_hub"]), color="#c62828", ls=":", lw=0.9, alpha=0.85)
    purity_handles = [
        mpl.lines.Line2D([], [], color="#239b56", lw=1.75, label=r"$r_{\rm MFG}(\beta)$"),
        mpl.lines.Line2D([], [], color="#c17c2d", lw=1.45, ls="--", label=r"$\mathcal P_{\rm MFG}(\beta)$"),
        mpl.lines.Line2D([], [], color="#c62828", lw=0.9, ls=":", label=r"$\beta_{\rm hub}$"),
    ]
    axC.set_xscale("log")
    axC.set_xlabel(r"$\beta\omega_q$")
    axC.set_ylabel(r"$r,\ \mathcal P$")
    axC.set_ylim(-0.02, 1.02)
    axC.grid(True, which="both", alpha=0.12, lw=0.35)
    axC.legend(handles=purity_handles, loc="upper left", framealpha=0.92, fontsize=6.3, handlelength=2.2)
    _panel_label(axC, "(c)")

    png = FIGURES_DIR / "hmf_tilt_sign_mfg_branch_family.png"
    pdf = FIGURES_DIR / "hmf_tilt_sign_mfg_branch_family.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")
    return png, pdf


def main() -> None:
    _configure_matplotlib()
    print("Building hub-centered MFG-guided branch-family figure...")
    make_figure()


if __name__ == "__main__":
    main()
