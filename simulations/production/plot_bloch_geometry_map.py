"""
plot_bloch_geometry_map.py  --  Hybrid Bloch geometry map for Results section
==============================================================================

Publication-quality 2x2 figure:
  (a) Bloch-sphere hero panel (3D overview)
  (b) Coupling-plane decomposition + bath-channel weighting (2D)
  (c) Symmetrised influence map (2D)
  (d) Thermal recombination + temperature-memory attractor inset (2D + inset)

The figure is generated from the exact analytic qubit formulas via the helper
exact_qubit_geometry_point() in fig1_chi_theory.py, and is aligned with the
notation used in the rewritten Results section.

Output:
  manuscript/figures/hmf_bloch_geometry_map.pdf
  manuscript/figures/hmf_bloch_geometry_map.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import (  # noqa: E402
    THETA,
    OMEGA_Q,
    exact_qubit_geometry_point,
    get_chi0,
)


FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "figure.dpi": 220,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
})


# -----------------------------------------------------------------------------
# Fixed worked point and inset parameters (locked by plan)
# -----------------------------------------------------------------------------
BETA_WORK = 4.0
THETA_WORK = THETA

BETAS_INSET = [0.5, 1.5, 3.0, 6.0]
COLORS_INSET = ["#d73027", "#f4a582", "#92c5de", "#2166ac"]  # same palette as disk portrait
LABELS_INSET = [
    r"$\beta\omega_q=0.5$",
    r"$\beta\omega_q=1.5$",
    r"$\beta\omega_q=3$",
    r"$\beta\omega_q=6$",
]

COL = {
    "sys": "#444444",
    "coupling": "#2c9a42",
    "bath": "#e08214",
    "infl": "#1f77b4",
    "state": "#d62728",
    "u": "#b2182b",
    "thermal": "#7b3294",
    "grid": "#999999",
}

HALO = [pe.withStroke(linewidth=2.8, foreground="white", alpha=0.9)]
SHOW_PLOT_TITLES = False
SHOW_PLOT_SUBTITLES = False


def _txt(ax, x, y, s, **kwargs):
    kwargs.setdefault("fontsize", 7.2)
    kwargs.setdefault("zorder", 20)
    t = ax.text(x, y, s, **kwargs)
    try:
        t.set_path_effects(HALO)
    except Exception:
        pass
    return t


def _panel_tag(ax, tag):
    if not tag:
        return
    if hasattr(ax, "text2D"):  # 3D axes
        ax.text2D(0.02, 0.98, tag, transform=ax.transAxes, va="top",
                  ha="left", fontweight="bold", fontsize=9)
    else:
        ax.text(0.02, 0.98, tag, transform=ax.transAxes, va="top",
                ha="left", fontweight="bold", fontsize=9)


def _draw_unit_disk(ax, radius=1.0, show_labels=True):
    disk = patches.Circle((0, 0), radius=radius, facecolor="#f7f7f7",
                          edgecolor="black", linewidth=0.85, zorder=0)
    ax.add_patch(disk)
    ax.plot([-2.0, 2.0], [0, 0], ls=":", lw=0.6, color=COL["grid"], alpha=0.45, zorder=0)
    ax.plot([0, 0], [-2.0, 2.0], ls=":", lw=0.6, color=COL["grid"], alpha=0.45, zorder=0)
    if show_labels:
        _txt(ax, 1.04, -0.03, r"$\hat{\mathbf r}_\perp$", ha="left", va="top", color=COL["sys"])
        _txt(ax, 0.04, 1.04, r"$\mathbf n_s$", ha="left", va="bottom", color=COL["sys"])
        _txt(ax, 0.04, -1.05, r"$-\mathbf n_s$", ha="left", va="top", color=COL["sys"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_vec2(ax, vec, color, label=None, ls="-", lw=1.8, alpha=1.0,
               zorder=4, head_scale=7, text_shift=None, text_color=None):
    vec = np.asarray(vec, dtype=float)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-12:
        return
    ax.annotate(
        "",
        xy=(vec[0], vec[1]),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=head_scale,
            linewidth=lw,
            linestyle=ls,
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )
    if label:
        if text_shift is None:
            text_shift = (0.08 * vec[0] / max(nrm, 1e-8), 0.08 * vec[1] / max(nrm, 1e-8))
        tx = vec[0] + text_shift[0]
        ty = vec[1] + text_shift[1]
        _txt(ax, tx, ty, label, color=(text_color or color), ha="center", va="center")


def _draw_component_guides(ax, vec, color, label_x=None, label_z=None):
    x, z = float(vec[0]), float(vec[1])
    ax.plot([x, x], [0, z], ls=":", lw=0.9, color=color, alpha=0.45, zorder=2)
    ax.plot([0, x], [z, z], ls=":", lw=0.9, color=color, alpha=0.45, zorder=2)
    if label_x:
        _txt(ax, x / 2, -0.11 if z >= 0 else 0.10, label_x, color=color, ha="center", va="center")
    if label_z:
        _txt(ax, -0.10 if x >= 0 else 0.10, z / 2, label_z, color=color, ha="center", va="center",
             rotation=90)


def _plane_limits(ax, lim=1.65):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def _add_axis_caption(ax, title, subtitle=None, subtitle_y=0.88):
    if SHOW_PLOT_TITLES:
        ax.set_title(title, loc="left", pad=4)
    if SHOW_PLOT_SUBTITLES and subtitle:
        ax.text(0.02, subtitle_y, subtitle, transform=ax.transAxes, fontsize=7.0,
                color="#444444", ha="left", va="top")


def _build_worked_point():
    chi0, _, _ = get_chi0(BETA_WORK)
    gstar = 1.0 / np.sqrt(chi0)
    g_work = gstar  # g/g_star = 1
    geom = exact_qubit_geometry_point(g_work, BETA_WORK, THETA_WORK)
    return g_work, geom


def _validate_point(geom):
    # Helper consistency checks required by the plan.
    chi = geom["chi"]
    h_eff = np.asarray(geom["h_eff"])
    u = np.asarray(geom["u"])
    gam = float(geom["gamma"])
    if abs(np.linalg.norm(u) - np.tanh(chi)) > 1e-9:
        raise RuntimeError("|u| != tanh(chi) consistency check failed.")
    if np.linalg.norm(u - gam * h_eff) > 1e-11:
        raise RuntimeError("u = gamma * h_eff consistency check failed.")

    # Thermal recombination identity (recompute independently from u).
    a = float(geom["a"])
    u_par = float(u[1])
    denom = np.cosh(a) - u_par * np.sinh(a)
    v_check = np.array([u[0] / denom, (u_par * np.cosh(a) - np.sinh(a)) / denom], dtype=float)
    if np.linalg.norm(v_check - geom["v"]) > 1e-11:
        raise RuntimeError("Bloch recombination consistency check failed.")


def _draw_3d_panel(ax, geom, tag=None):
    _panel_tag(ax, tag)
    if SHOW_PLOT_TITLES:
        ax.set_title("Bloch-sphere overview", loc="left", pad=2)
    if SHOW_PLOT_SUBTITLES:
        ax.text2D(
            0.02, 0.91,
            rf"Worked point: $\theta=\pi/4$, $\beta\omega_q={BETA_WORK:g}$, $g/g_\star=1$",
            transform=ax.transAxes, fontsize=6.7, color="#444444"
        )

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    # Sphere shell + guide circles
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, rstride=6, cstride=6, color="#bdbdbd",
                      linewidth=0.4, alpha=0.35, zorder=0)

    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), 0*t, color="#bdbdbd", lw=0.8, alpha=0.4)  # equator
    ax.plot(np.cos(t), 0*t, np.sin(t), color="#bdbdbd", lw=0.8, alpha=0.55)  # coupling plane

    # Subtle axis guides
    ax.plot([0, 0], [0, 0], [-1.05, 1.05], color=COL["sys"], lw=0.6, ls=":", alpha=0.4)
    ax.plot([-1.05, 1.05], [0, 0], [0, 0], color=COL["sys"], lw=0.5, ls=":", alpha=0.25)

    # Vectors
    vecs = [
        (geom["n_s_vec"], COL["sys"], r"$\mathbf n_s$", "-", 1.5, 0.95),
        (geom["r_vec_3d"], COL["coupling"], r"$\mathbf r$", "-", 1.8, 0.98),
        (geom["h_eff_3d"], COL["infl"], None, "-", 2.0, 1.02),
        (geom["v_3d"], COL["state"], None, "-", 2.2, 1.02),
    ]
    for vec, color, label, ls, lw, txt_scale in vecs:
        vec = np.asarray(vec, dtype=float)
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                  color=color, linewidth=lw, linestyle=ls,
                  arrow_length_ratio=0.10, normalize=False)
        tip = vec * txt_scale
        if label:
            ax.text(tip[0], tip[1], tip[2], label, fontsize=7,
                    color=color, zorder=10,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # Legend (external to the sphere body)
    handles = [
        Line2D([0], [0], color=COL["sys"], lw=1.4, label=r"system axis $\mathbf n_s$"),
        Line2D([0], [0], color=COL["coupling"], lw=1.6, label=r"coupling $\mathbf r$"),
        Line2D([0], [0], color=COL["infl"], lw=1.8, label=r"influence $\mathbf h_{\mathrm{eff}}$"),
        Line2D([0], [0], color=COL["state"], lw=1.8, label=r"state $\mathbf v$"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.00, -0.02),
              framealpha=0.92, borderpad=0.35, handlelength=1.4, labelspacing=0.22,
              fontsize=7.2)

    ax.view_init(elev=23, azim=-57)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_alpha(0.0)
            axis.line.set_alpha(0.0)
        except Exception:
            pass


def _draw_panel_b(ax, geom, tag=None):
    _panel_tag(ax, tag)
    _draw_unit_disk(ax)
    _plane_limits(ax, lim=1.95)
    _add_axis_caption(ax, "Decompose + bath-channel weighting",
                      r"$\mathbf a_b=(b_{\mathrm r}r_\perp)\hat{\mathbf r}_\perp + (b_{\mathrm{nr}}r_\parallel)\mathbf n_s$",
                      subtitle_y=0.92)

    r = np.asarray(geom["r_vec_plane"])
    a_b = np.asarray(geom["a_b"])

    # Show bare coupling and its decomposition
    _draw_vec2(ax, r, COL["coupling"], r"$\mathbf r$", lw=1.9)
    _draw_component_guides(ax, r, COL["coupling"], r"$r_\perp$", r"$r_\parallel$")
    _draw_vec2(ax, np.array([r[0], 0.0]), COL["coupling"], None, lw=1.3, alpha=0.35)
    _draw_vec2(ax, np.array([0.0, r[1]]), COL["coupling"], None, lw=1.3, alpha=0.35)

    # Bath-weighted pre-influence vector
    _draw_vec2(ax, a_b, COL["bath"], r"$\mathbf a_b$", lw=2.0,
               text_shift=(0.12*np.sign(a_b[0] if a_b[0] != 0 else 1), 0.10))
    _draw_component_guides(ax, a_b, COL["bath"],
                           r"$b_{\mathrm r}r_\perp$", r"$b_{\mathrm{nr}}r_\parallel$")

    # Small channel-space inset (make b explicit as non-spatial)
    ax_in = ax.inset_axes([0.60, 0.10, 0.32, 0.32])
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.95)
    ax_in.axhline(0, color="#999999", lw=0.5, ls=":")
    ax_in.axvline(0, color="#999999", lw=0.5, ls=":")
    b_vec = np.array([geom["b_nr"], geom["b_r"]], dtype=float)
    b_lim = max(1.0, 1.15 * np.max(np.abs(b_vec)))
    ax_in.set_xlim(-b_lim, b_lim)
    ax_in.set_ylim(-b_lim, b_lim)
    ax_in.annotate("", xy=(b_vec[0], b_vec[1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color=COL["bath"], lw=1.3,
                                   mutation_scale=6))
    ax_in.plot(b_vec[0], b_vec[1], "o", color=COL["bath"], ms=3)
    ax_in.text(0.94, 0.06, r"$b_{\mathrm{nr}}$", transform=ax_in.transAxes,
               ha="right", va="bottom", fontsize=5.7)
    ax_in.text(0.05, 0.95, r"$b_{\mathrm r}$", transform=ax_in.transAxes,
               ha="left", va="top", fontsize=5.7)
    ax_in.set_xticks([])
    ax_in.set_yticks([])


def _draw_panel_c(ax, geom, tag=None):
    _panel_tag(ax, tag)
    _draw_unit_disk(ax)
    _plane_limits(ax, lim=1.95)
    chi = geom["chi"]
    gam = geom["gamma"]
    _add_axis_caption(
        ax,
        "Symmetrised influence map",
        rf"$\tilde{{\mathbf h}}=\mathbf N\times \mathbf a_b,\ \mathbf u=\gamma(\chi)\mathbf h_{{\mathrm{{eff}}}},\ \chi={chi:.2f}$",
        subtitle_y=0.92,
    )

    a_b = np.asarray(geom["a_b"])
    h_tilde = np.asarray(geom["h_tilde"])
    h_eff = np.asarray(geom["h_eff"])
    u = np.asarray(geom["u"])

    # Mapping chain with visual hierarchy
    _draw_vec2(ax, a_b, COL["bath"], r"$\mathbf a_b$", lw=1.6, alpha=0.55,
               text_shift=(0.10, 0.10))
    _draw_vec2(ax, h_tilde, COL["infl"], None, lw=1.6, ls=":",
               alpha=0.85, text_shift=(0.12, -0.12))
    _draw_vec2(ax, h_eff, COL["infl"], r"$\mathbf h_{\mathrm{eff}}$", lw=2.2,
               text_shift=(-0.26, -0.16))
    _draw_vec2(ax, u, COL["u"], r"$\mathbf u$", lw=1.8, ls="--",
               text_shift=(0.18, 0.14))
    _txt(ax, h_tilde[0] + 0.16, h_tilde[1] + 0.02, r"$\tilde{\mathbf h}$",
         color=COL["infl"], fontsize=6.8)

    # 90° rotation arc (a_b -> h_tilde)
    th1 = np.arctan2(a_b[1], a_b[0])
    th2 = np.arctan2(h_tilde[1], h_tilde[0])
    if th2 < th1:
        th2 += 2*np.pi
    r_arc = 0.42
    tt = np.linspace(th1, th2, 80)
    ax.plot(r_arc*np.cos(tt), r_arc*np.sin(tt), color="#444444", lw=0.9, alpha=0.7)
    ax.annotate("", xy=(r_arc*np.cos(tt[-1]), r_arc*np.sin(tt[-1])),
                xytext=(r_arc*np.cos(tt[-5]), r_arc*np.sin(tt[-5])),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#444444",
                                mutation_scale=6))
    _txt(ax, r_arc*np.cos((th1+th2)/2)+0.12, r_arc*np.sin((th1+th2)/2)+0.10,
         r"$\mathbf N\times$", color="#444444", fontsize=6.8)

    # Scale/saturation callouts
    # Keep panel interior focused on vectors; move scalar annotations to caption.


def _draw_panel_d(ax, geom, tag=None):
    _panel_tag(ax, tag)
    _draw_unit_disk(ax)
    _plane_limits(ax, lim=1.25)
    _add_axis_caption(
        ax,
        "Thermal recombination",
        r"$\mathbf v_{\rm th}=-\tanh(a)\mathbf n_s,\ \ \mathbf v = \mathcal{R}_\Pi(\mathbf u)$",
        subtitle_y=0.92,
    )

    u = np.asarray(geom["u"])
    v = np.asarray(geom["v"])
    v_th = np.asarray(geom["v_th"])

    # Auxiliary and final state vectors
    _draw_vec2(ax, v_th, COL["thermal"], r"$\mathbf v_{\mathrm{th}}$", ls=":", lw=1.6,
               text_shift=(0.20, 0.00))
    _draw_vec2(ax, u, COL["u"], None, ls="--", lw=1.8)
    _draw_vec2(ax, v, COL["state"], r"$\mathbf v$", lw=2.3,
               text_shift=(0.14, -0.12))
    ax.plot([v_th[0], v[0]], [v_th[1], v[1]], color="#666666", lw=0.9, alpha=0.35)

    # Inset: temperature-dependent large-g attractor directions
    ax_in = ax.inset_axes([0.54, 0.50, 0.42, 0.42])
    disk = patches.Circle((0, 0), radius=1.0, facecolor="white", edgecolor="#777777",
                          linewidth=0.65, zorder=0)
    ax_in.add_patch(disk)
    ax_in.axhline(0, color="#aaaaaa", lw=0.5, ls=":", alpha=0.6)
    ax_in.axvline(0, color="#aaaaaa", lw=0.5, ls=":", alpha=0.6)

    r_ref = np.asarray(geom["r_vec_plane"])
    r_ref = r_ref / max(np.linalg.norm(r_ref), 1e-12)
    ax_in.annotate("", xy=(0.85*r_ref[0], 0.85*r_ref[1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.8,
                                   ls="--", mutation_scale=5))
    ax_in.text(0.87*r_ref[0], 0.87*r_ref[1], r"$\hat{\mathbf r}$", fontsize=5.8,
               color="#666666", ha="left" if r_ref[0] >= 0 else "right", va="bottom")

    for beta, col, lab in zip(BETAS_INSET, COLORS_INSET, LABELS_INSET):
        chi0, _, _ = get_chi0(beta)
        gstar = 1.0 / np.sqrt(chi0)
        geom_inf = exact_qubit_geometry_point(20.0 * gstar, beta, THETA_WORK)
        v_inf = np.asarray(geom_inf["v"])
        # Use large-g exact state directly (already near the boundary) to show temperature memory.
        ax_in.plot(v_inf[0], v_inf[1], marker="s", ms=4.1, color=col,
                   mec="white", mew=0.45, zorder=4)
        ax_in.annotate("", xy=(0.96*v_inf[0], 0.96*v_inf[1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="-", lw=1.0, color=col, alpha=0.65))

    # Inset is described in the caption to avoid on-figure text crowding.
    ax_in.set_xlim(-1.05, 1.05)
    ax_in.set_ylim(-1.05, 1.05)
    ax_in.set_aspect("equal")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=col, marker="s", markersize=4, lw=1.0, label=lab)
        for col, lab in zip(COLORS_INSET, LABELS_INSET)
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.00, -0.02),
              framealpha=0.93, borderpad=0.30, handlelength=1.0, labelspacing=0.20,
              fontsize=6.4)


def _save_single_panel(draw_fn, geom, out_stem, figsize=(3.35, 3.10), projection3d=False):
    fig = plt.figure(figsize=figsize)
    if projection3d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    draw_fn(ax, geom, tag=None)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.96)
    out_png = FIGURES / f"{out_stem}.png"
    out_pdf = FIGURES / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")


def main():
    g_work, geom = _build_worked_point()
    _validate_point(geom)

    # Diagnostic printout (documents the exact checks requested in the plan).
    print("Worked point diagnostics")
    print(f"  theta = {geom['theta']:.6f} rad")
    print(f"  beta*omega_q = {BETA_WORK:.6f}")
    print(f"  g/g_star = {g_work / geom['gstar']:.6f}")
    print(f"  chi = {geom['chi']:.12f}, gamma = {geom['gamma']:.12f}")
    print(f"  b_nr = {geom['b_nr']:.12f}, b_r = {geom['b_r']:.12f}")
    print(f"  mx = {geom['mx']:.12f}, mz = {geom['mz']:.12f}  [matches bloch_ohmic]")

    # Standalone panels for readable manuscript placement.
    _save_single_panel(
        _draw_3d_panel, geom,
        out_stem="hmf_bloch_overview",
        figsize=(3.45, 3.30), projection3d=True,
    )
    _save_single_panel(
        _draw_panel_b, geom,
        out_stem="hmf_bloch_decompose_weighting",
        figsize=(3.45, 3.25),
    )
    _save_single_panel(
        _draw_panel_c, geom,
        out_stem="hmf_bloch_symmetrised_map",
        figsize=(3.45, 3.25),
    )
    _save_single_panel(
        _draw_panel_d, geom,
        out_stem="hmf_bloch_thermal_recombination",
        figsize=(3.45, 3.25),
    )


if __name__ == "__main__":
    main()
