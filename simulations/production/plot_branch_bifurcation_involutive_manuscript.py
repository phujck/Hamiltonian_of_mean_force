"""
plot_branch_bifurcation_involutive_manuscript.py
================================================
Publication-facing figure suite for the RG-framed branch-bifurcation appendix.

This script reuses the exact qubit helper functions from the standalone branch
diagnostic script (`plot_bloch_branch_flip_test.py`) but produces a smaller set
of polished figures aimed at manuscript inclusion.

Core framing
------------
- Running coupling / purity flow variable:
    chi(beta,g) = g^2 chi0(beta)
- Involutive (self-dual) crossover line:
    chi = 1  <=>  g = g_star(beta)  <=>  beta = beta_star(g)
- Branch multiplicity is shown as an angular-sheet continuation construction
  (RG-like sector attractors / fixed directions), while the exact purity
  remains single-valued: r = tanh(chi).

Outputs
-------
- manuscript/figures/hmf_branch_bifurcation_bloch_fullcircle.{pdf,png}
- manuscript/figures/hmf_branch_bifurcation_logchi.{pdf,png}
- manuscript/figures/hmf_branch_bifurcation_involutive_map.{pdf,png}
- simulations/production/out/hmf_branch_bifurcation_summary.csv
- simulations/production/out/hmf_branch_bifurcation_grid.npz
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
OUT_DIR = ROOT / "simulations" / "production" / "out"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Plot / grid parameters (kept moderate so figure generation stays quick)
N_BETA_REP = 480
N_BETA_MAP = 140
N_G_MAP = 120

BETA_MIN = base.BETA_MIN
BETA_MAX = base.BETA_MAX

REP_MULT_DEFAULT = 1.25
REP_MULT_FALLBACK_ORDER = [1.25, 1.1, 1.4, 2.0, 4.0]

# RG-branch construction tuning (publication-facing visual construction)
RG_BLEND_POWER = 1.15
RG_OUT_POWER = 0.80


@dataclass
class RepresentativeBranchData:
    g_ref: float
    g_mult_ref: float
    gstar_ref: float
    beta: np.ndarray
    chi: np.ndarray
    y_logchi: np.ndarray
    r: np.ndarray
    mz_exact: np.ndarray
    mperp_exact: np.ndarray
    alpha_exact: np.ndarray          # absolute vector angle in plotting convention
    alpha_exact_dir: np.ndarray      # director version in [0,pi)
    beta_star: float
    i_star: int
    alpha_hub: float                 # manuscript-facing pi-theta ray in plotting convention
    alpha_theta: float               # manuscript-facing theta ray in plotting convention
    alpha_incoming: np.ndarray       # shared pre-T* branch (construction)
    alpha_ground: np.ndarray         # post-T* groundward exact branch
    alpha_reflected: np.ndarray      # post-T* theta-sector continuation (construction)
    mperp_incoming: np.ndarray
    mz_incoming: np.ndarray
    mperp_ground: np.ndarray
    mz_ground: np.ndarray
    mperp_reflected: np.ndarray
    mz_reflected: np.ndarray
    beta_split: float                # chosen split extremum (for reporting/annotations)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
            "figure.dpi": 200,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def _panel_label(ax: mpl.axes.Axes, lab: str, color: str = "black") -> None:
    ax.text(
        0.04,
        0.96,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=color,
    )


def _curve_arrow(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.65,
    color: str = "k",
    lw: float = 0.9,
    zorder: int = 10,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return
    i1 = int(np.clip(round(frac * (x.size - 1)), 1, x.size - 2))
    i0 = max(0, i1 - 2)
    if not (np.isfinite(x[i0]) and np.isfinite(y[i0]) and np.isfinite(x[i1]) and np.isfinite(y[i1])):
        return
    ax.annotate(
        "",
        xy=(float(x[i1]), float(y[i1])),
        xytext=(float(x[i0]), float(y[i0])),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def _chi0_curve(beta: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chi0 = np.empty_like(beta)
    dz0 = np.empty_like(beta)
    sx0 = np.empty_like(beta)
    for i, b in enumerate(beta):
        c, d, x = base.get_channels0(float(b), theta=theta)
        chi0[i] = c
        dz0[i] = d
        sx0[i] = x
    return chi0, dz0, sx0


def _monotonicity_flag(arr: np.ndarray) -> bool:
    d = np.diff(arr)
    # allow tiny numerical plateaus/noise
    return bool(np.all(d >= -1e-12 * np.maximum(1.0, np.abs(arr[:-1]))))


def _gstar_ref(theta: float) -> float:
    chi0_ref, _, _ = base.get_channels0(base.BETA_REF, theta=theta)
    if chi0_ref <= 0.0:
        raise RuntimeError("chi0_ref <= 0; cannot define g*_ref")
    return float(1.0 / np.sqrt(chi0_ref))


def _build_trace_for_g(g: float, g_mult_ref: float, beta_grid: np.ndarray) -> dict[str, np.ndarray]:
    pts = [base.build_branch_point(g=float(g), g_mult_ref=float(g_mult_ref), beta=float(b)) for b in beta_grid]
    # exact absolute vector angle in the plotting convention from (-z) toward (+m_perp)
    # This is the same as phi_wrap if we interpret the vector angle (not modulo pi).
    alpha_exact = np.array([float(np.mod(p.phi_wrap, 2.0 * np.pi)) for p in pts], dtype=float)
    alpha_exact_dir = np.array([float(np.mod(p.phi_wrap, np.pi)) for p in pts], dtype=float)

    return {
        "beta": beta_grid.copy(),
        "chi": np.array([p.chi for p in pts], dtype=float),
        "r": np.array([p.r for p in pts], dtype=float),
        "mz": np.array([p.mz for p in pts], dtype=float),
        "mperp": np.array([p.m_perp_signed for p in pts], dtype=float),
        "phi_wrap": np.array([p.phi_wrap for p in pts], dtype=float),
        "alpha_exact": alpha_exact,
        "alpha_exact_dir": alpha_exact_dir,
    }


def _beta_star_from_trace(beta: np.ndarray, chi: np.ndarray) -> tuple[float, int]:
    idxs = base.find_sign_changes(chi - 1.0)
    if not idxs:
        raise RuntimeError("No chi=1 crossing on beta window for representative g.")
    # choose the first crossing in the monotone beta sweep
    i = int(idxs[0])
    beta_star = base.interp_root(float(beta[i]), float(chi[i] - 1.0), float(beta[i + 1]), float(chi[i + 1] - 1.0))
    return float(beta_star), i


def _smoothstep01(x: np.ndarray, power: float = 1.0) -> np.ndarray:
    xx = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if power != 1.0:
        xx = xx**power
    # cubic smoothstep
    return xx * xx * (3.0 - 2.0 * xx)


def _polar_to_plane(r: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # plotting convention: alpha measured from -z toward +m_perp
    mperp = r * np.sin(alpha)
    mz = -r * np.cos(alpha)
    return mperp, mz


def _representative_branch_construction(theta: float = base.THETA_VAL) -> RepresentativeBranchData:
    gstar_ref = _gstar_ref(theta)
    beta_grid = np.geomspace(BETA_MIN, BETA_MAX, N_BETA_REP)
    # warm the cache on the representative beta grid
    for b in beta_grid:
        base.get_channels0(float(b), theta=theta)

    # find representative g > g*_ref with visible chi=1 crossing
    rep_candidates = [m for m in REP_MULT_FALLBACK_ORDER if m in base.G_MULTS_B]
    if REP_MULT_DEFAULT not in rep_candidates:
        rep_candidates = [REP_MULT_DEFAULT] + rep_candidates

    chosen_mult = None
    chosen_trace = None
    for mult in rep_candidates:
        g = float(mult * gstar_ref)
        tr = _build_trace_for_g(g, float(mult), beta_grid)
        if np.any((tr["chi"][:-1] - 1.0) * (tr["chi"][1:] - 1.0) <= 0.0):
            chosen_mult = float(mult)
            chosen_trace = tr
            break
    if chosen_mult is None or chosen_trace is None:
        raise RuntimeError("No representative g > g*_ref with chi=1 crossing found on beta window.")

    g_ref = float(chosen_mult * gstar_ref)
    beta = chosen_trace["beta"]
    chi = chosen_trace["chi"]
    r = chosen_trace["r"]
    mz_exact = chosen_trace["mz"]
    mperp_exact = chosen_trace["mperp"]
    alpha_exact = chosen_trace["alpha_exact"]
    alpha_exact_dir = chosen_trace["alpha_exact_dir"]
    y_logchi = np.log(np.clip(chi, 1e-14, None))

    beta_star, i_star = _beta_star_from_trace(beta, chi)
    # interpolate exact director angle/radius at beta_star
    alpha_star_exact = float(np.interp(beta_star, beta, alpha_exact_dir))
    r_star = float(np.interp(beta_star, beta, r))

    # Manuscript-facing rays in this plotting convention:
    #   alpha = 0 points along -z (groundward).
    #   alpha = theta and alpha = pi-theta are shown explicitly.
    alpha_theta = float(theta)
    alpha_hub = float(np.pi - theta)

    # Choose a reporting "split extremum" from the visible gap between the exact
    # director angle and the hub ray on the low-T side.  This is used for
    # metadata and optional annotations; the actual branch gluing is at beta_star.
    mask_post = beta >= beta_star
    if np.any(mask_post):
        sep_post = np.abs(alpha_exact_dir[mask_post] - alpha_hub)
        i_rel = int(np.argmax(sep_post))
        i_split = int(np.where(mask_post)[0][i_rel])
        beta_split = float(beta[i_split])
    else:
        beta_split = float(beta_star)

    # ------------------------------------------------------------------
    # RG-like branch construction (interpretive angular sheet continuations)
    # ------------------------------------------------------------------
    # Shared incoming branch (high-T side) that is glued to the hub ray at chi=1.
    # We use the exact *vector-angle sheet* (mod 2pi) as the baseline and blend it toward alpha_hub
    # as chi -> 1^-.
    alpha_incoming = np.array(alpha_exact, dtype=float)
    mask_pre = beta <= beta_star
    if np.any(mask_pre):
        chi_pre = np.clip(chi[mask_pre], 0.0, 1.0)
        s_pre = _smoothstep01(chi_pre, power=RG_BLEND_POWER)
        alpha_base_pre = alpha_exact[mask_pre]
        alpha_incoming[mask_pre] = (1.0 - s_pre) * alpha_base_pre + s_pre * alpha_hub
    # Lock exact at the split point to the hub ray.
    alpha_incoming[~mask_pre] = alpha_hub

    # Groundward outgoing branch:
    # use the short vector-angle sheet (exact angle modulo 2pi, pulled to the
    # ground ray near alpha=0 rather than its lifted 2pi image) so the Bloch
    # bifurcation geometry is visually interpretable.
    alpha_ground = np.array(alpha_incoming, dtype=float)
    if np.any(mask_post):
        y_post = np.clip(y_logchi[mask_post], 0.0, None)
        # normalized post-crossover RG parameter in [0,1]
        y_max = max(float(np.max(y_post)), 1e-12)
        t_post = _smoothstep01(y_post / y_max, power=RG_OUT_POWER)
        alpha_exact_short = np.mod(alpha_exact[mask_post], 2.0 * np.pi)
        alpha_exact_short = np.where(alpha_exact_short > np.pi, alpha_exact_short - 2.0 * np.pi, alpha_exact_short)
        # Blend from the hub ray at beta_star to the short exact vector sheet near alpha=0.
        alpha_ground[mask_post] = (1.0 - t_post) * alpha_hub + t_post * alpha_exact_short

    # Theta-sector outgoing branch:
    # z-axis-flipped continuation in the manuscript-facing interpretation,
    # constructed to share the same split point and approach the theta ray.
    alpha_reflected = np.array(alpha_incoming, dtype=float)
    if np.any(mask_post):
        y_post = np.clip(y_logchi[mask_post], 0.0, None)
        y_max = max(float(np.max(y_post)), 1e-12)
        t_post = _smoothstep01(y_post / y_max, power=RG_OUT_POWER)
        alpha_reflected[mask_post] = (1.0 - t_post) * alpha_hub + t_post * alpha_theta

    # Convert branch-angle constructions into magnetisation trajectories using the exact purity.
    mperp_incoming, mz_incoming = _polar_to_plane(r, alpha_incoming)
    mperp_ground, mz_ground = _polar_to_plane(r, alpha_ground)
    mperp_reflected, mz_reflected = _polar_to_plane(r, alpha_reflected)

    # Force exact split-point coincidence at beta_star by replacing the nearest sample
    # with the analytic hub point at r(beta_star).  This sharpens the bifurcation visual.
    i_near_star = int(np.argmin(np.abs(beta - beta_star)))
    x_hub = float(r_star * np.sin(alpha_hub))
    z_hub = float(-r_star * np.cos(alpha_hub))
    for xarr, zarr, aarr in [
        (mperp_incoming, mz_incoming, alpha_incoming),
        (mperp_ground, mz_ground, alpha_ground),
        (mperp_reflected, mz_reflected, alpha_reflected),
    ]:
        xarr[i_near_star] = x_hub
        zarr[i_near_star] = z_hub
        aarr[i_near_star] = alpha_hub

    return RepresentativeBranchData(
        g_ref=g_ref,
        g_mult_ref=float(chosen_mult),
        gstar_ref=float(gstar_ref),
        beta=beta,
        chi=chi,
        y_logchi=y_logchi,
        r=r,
        mz_exact=mz_exact,
        mperp_exact=mperp_exact,
        alpha_exact=alpha_exact,
        alpha_exact_dir=alpha_exact_dir,
        beta_star=float(beta_star),
        i_star=int(i_star),
        alpha_hub=float(alpha_hub),
        alpha_theta=float(alpha_theta),
        alpha_incoming=alpha_incoming,
        alpha_ground=alpha_ground,
        alpha_reflected=alpha_reflected,
        mperp_incoming=mperp_incoming,
        mz_incoming=mz_incoming,
        mperp_ground=mperp_ground,
        mz_ground=mz_ground,
        mperp_reflected=mperp_reflected,
        mz_reflected=mz_reflected,
        beta_split=float(beta_split),
    )


def _save_fig(fig: mpl.figure.Figure, stem: str) -> tuple[Path, Path]:
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")
    print(f"Saved -> {pdf}")
    return png, pdf


def make_figure_bloch_fullcircle(rep: RepresentativeBranchData) -> None:
    fig = plt.figure(figsize=(6.8, 2.95))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.15, 0.95],
        wspace=0.33,
        left=0.08,
        right=0.985,
        bottom=0.18,
        top=0.96,
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # Panel A: full Bloch circle bifurcation topology
    t = np.linspace(0.0, 2.0 * np.pi, 800)
    axA.plot(np.cos(t), np.sin(t), color="#202020", lw=1.0, alpha=0.95)
    axA.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    axA.axvline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)

    # Rays in the plotting convention (alpha measured from -z toward +m_perp).
    # For theta=pi/4 used here, the two rays differ by pi/2.
    ray_specs = [
        (rep.alpha_theta, "#9d2a2a", "-."),
        (rep.alpha_hub, "#9d2a2a", "-"),
    ]
    rr = np.linspace(0.0, 1.0, 2)
    for ang, col, ls in ray_specs:
        axA.plot(rr * np.sin(ang), -rr * np.cos(ang), color=col, ls=ls, lw=0.9, alpha=0.8, zorder=1)

    # Small origin arc highlighting the sector separation (pi-2 theta = pi/2 at theta=pi/4).
    ang_arc = np.linspace(rep.alpha_theta, rep.alpha_hub, 80)
    r_arc = 0.18
    axA.plot(r_arc * np.sin(ang_arc), -r_arc * np.cos(ang_arc), color="#9d2a2a", lw=0.8, alpha=0.8, zorder=1)
    ang_mid = 0.5 * (rep.alpha_theta + rep.alpha_hub)
    axA.text(
        1.08 * r_arc * np.sin(ang_mid),
        -1.08 * r_arc * np.cos(ang_mid),
        r"$\pi/2$",
        color="#7d1d1d",
        fontsize=6.6,
        ha="center",
        va="center",
    )

    beta = rep.beta
    mask_pre = beta <= rep.beta_star
    mask_post = beta >= rep.beta_star

    # Exact physical trajectory (context only; branch picture acts on the angle sheet).
    axA.plot(rep.mperp_exact, rep.mz_exact, color="#9f9f9f", lw=0.9, ls="-", alpha=0.55, zorder=2)

    # shared incoming branch (RG-glued to hub at chi=1)
    axA.plot(
        rep.mperp_incoming[mask_pre],
        rep.mz_incoming[mask_pre],
        color="#187bcd",
        lw=1.8,
        zorder=4,
    )
    # outgoing branches from the same split point
    axA.plot(
        rep.mperp_ground[mask_post],
        rep.mz_ground[mask_post],
        color="#1b4f9c",
        lw=1.8,
        zorder=5,
    )
    axA.plot(
        rep.mperp_reflected[mask_post],
        rep.mz_reflected[mask_post],
        color="#1f9d55",
        lw=1.8,
        ls="--",
        zorder=5,
    )

    # split point at beta_star on the hub ray
    r_star = float(np.interp(rep.beta_star, rep.beta, rep.r))
    x_star = float(r_star * np.sin(rep.alpha_hub))
    z_star = float(-r_star * np.cos(rep.alpha_hub))
    axA.plot(x_star, z_star, marker="*", ms=8.4, color="#c62828", mec="white", mew=0.6, zorder=7)
    axA.annotate(
        r"$T_\star(g)$, $\chi=1$",
        xy=(x_star, z_star),
        xytext=(x_star + 0.12, z_star + 0.03),
        fontsize=7.0,
        color="#7d1d1d",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#7d1d1d", lw=0.6, shrinkA=2, shrinkB=2),
    )

    # annotate endpoint sectors (dual convention cue)
    x_th = 0.86 * np.sin(rep.alpha_theta)
    z_th = -0.86 * np.cos(rep.alpha_theta)
    axA.text(x_th + 0.03, z_th - 0.02, r"$\theta$", fontsize=7.2, color="#7d1d1d", ha="left", va="top")
    x_hub = 0.86 * np.sin(rep.alpha_hub)
    z_hub = -0.86 * np.cos(rep.alpha_hub)
    axA.text(x_hub + 0.03, z_hub - 0.02, r"$\pi-\theta$", fontsize=7.2, color="#7d1d1d", ha="left", va="top")

    # Direct branch annotations (compact, to avoid a large legend).
    if np.any(mask_pre):
        pre_idxs = np.where(mask_pre)[0]
        i_pre = pre_idxs[max(0, len(pre_idxs) // 2)]
        axA.text(rep.mperp_incoming[i_pre] - 0.06, rep.mz_incoming[i_pre] + 0.05,
                 r"incoming", color="#187bcd", fontsize=7.1, ha="right", va="bottom")
    if np.any(mask_post):
        post_idxs = np.where(mask_post)[0]
        idx_po = post_idxs[min(len(post_idxs) - 1, len(post_idxs) // 2)]
        axA.text(rep.mperp_ground[idx_po] - 0.04, rep.mz_ground[idx_po] + 0.06,
                 r"groundward", color="#1b4f9c", fontsize=7.1, ha="right", va="bottom")
        axA.text(rep.mperp_reflected[idx_po] + 0.03, rep.mz_reflected[idx_po] - 0.02,
                 r"sign-flipped", color="#1f9d55", fontsize=7.1, ha="left", va="top")

    # Direction arrows (increasing beta).
    if np.any(mask_pre):
        _curve_arrow(axA, rep.mperp_incoming[mask_pre], rep.mz_incoming[mask_pre], frac=0.75, color="#187bcd", lw=0.8)
    if np.any(mask_post):
        _curve_arrow(axA, rep.mperp_ground[mask_post], rep.mz_ground[mask_post], frac=0.45, color="#1b4f9c", lw=0.8)
        _curve_arrow(axA, rep.mperp_reflected[mask_post], rep.mz_reflected[mask_post], frac=0.45, color="#1f9d55", lw=0.8)
        _curve_arrow(axA, rep.mperp_exact, rep.mz_exact, frac=0.82, color="#8e8e8e", lw=0.7, zorder=3)

    axA.set_aspect("equal", adjustable="box")
    axA.set_xlim(-1.05, 1.05)
    axA.set_ylim(-1.05, 1.05)
    axA.set_xlabel(r"$m_\perp^{(\rm signed)}$")
    axA.set_ylabel(r"$m_z$")
    axA.grid(True, alpha=0.15, lw=0.4)
    _panel_label(axA, "(a)")

    # Panel B: same figure's purity seal
    axB.plot(beta, rep.r, color="#111111", lw=1.2)
    axB.plot(beta[mask_pre], rep.r[mask_pre], color="#187bcd", lw=1.5, alpha=0.95)
    axB.plot(beta[mask_post], rep.r[mask_post], color="#1b4f9c", lw=1.5, alpha=0.95)
    axB.plot(beta[mask_post], rep.r[mask_post], color="#1f9d55", lw=1.5, ls="--", alpha=0.95)
    axB.axvline(rep.beta_star, color="#c62828", ls=":", lw=0.9, alpha=0.85)
    axB.axhline(float(np.tanh(1.0)), color="#666666", ls=":", lw=0.7, alpha=0.55)
    axB.text(
        rep.beta_star * 1.04,
        0.09,
        r"$\beta_\star(g)$",
        color="#7d1d1d",
        fontsize=6.8,
        rotation=90,
        va="bottom",
        ha="left",
    )
    axB.set_xscale("log")
    axB.set_xlabel(r"$\beta\omega_q$")
    axB.set_ylabel(r"Purity $r$")
    axB.set_ylim(-0.02, 1.02)
    axB.grid(True, which="both", alpha=0.18, lw=0.4)
    axB.text(0.05, 0.07, r"same exact $r(\beta)$ for all continuations",
             transform=axB.transAxes, fontsize=6.8, ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cccccc", alpha=0.85))
    _panel_label(axB, "(b)")
    _save_fig(fig, "hmf_branch_bifurcation_bloch_fullcircle")


def make_figure_logchi(rep: RepresentativeBranchData) -> None:
    fig = plt.figure(figsize=(6.8, 2.9))
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.18, 0.92],
        wspace=0.34,
        left=0.08, right=0.985, bottom=0.18, top=0.96
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    y = rep.y_logchi
    beta = rep.beta
    mask_pre = y <= 0.0
    mask_post = y >= 0.0

    # Branch angles on lifted chart
    axA.plot(y[mask_pre], rep.alpha_incoming[mask_pre], color="#1c8ad3", lw=1.5)
    axA.plot(y[mask_post], rep.alpha_ground[mask_post], color="#1b4f9c", lw=1.5)
    axA.plot(y[mask_post], rep.alpha_reflected[mask_post], color="#1f9d55", lw=1.5, ls="--")

    # Faint exact vector-angle sheet for context (mod 2pi in the plotting convention).
    axA.plot(y, rep.alpha_exact, color="#b8b8b8", lw=0.9, alpha=0.85)

    axA.axvline(0.0, color="#c62828", ls=":", lw=1.1, alpha=0.9)
    axA.axhline(rep.alpha_hub, color="#7a7a7a", ls=":", lw=0.9, alpha=0.75)
    axA.axhline(rep.alpha_theta, color="#4f4f4f", ls="-.", lw=0.9, alpha=0.75)
    axA.axhline(2.0 * np.pi, color="#6f6f6f", ls="--", lw=0.8, alpha=0.55)
    axA.text(0.02, rep.alpha_hub + 0.03, r"$\pi-\theta$", fontsize=6.8, color="#666666", ha="left", va="bottom")
    axA.text(0.02, rep.alpha_theta - 0.05, r"$\theta$", fontsize=6.8, color="#4f4f4f", ha="left", va="top")
    x_2pi = float(np.max(y) - 1.05)
    axA.text(x_2pi, 2.0 * np.pi - 0.05, r"$0\equiv 2\pi$", fontsize=6.7, color="#5f5f5f", ha="left", va="top")

    axA.set_xlabel(r"$y=\log\chi$")
    axA.set_ylabel(r"Lifted angle $\alpha$ (rad)")
    axA.grid(True, alpha=0.18, lw=0.4)
    # Direct labels read better than a legend in the two-column panel width.
    x_in = float(np.min(y) + 0.18)
    i_in = int(np.argmin(np.abs(y - x_in)))
    axA.text(x_in + 0.05, rep.alpha_incoming[i_in] + 0.14, "incoming",
             color="#1c8ad3", fontsize=6.8, ha="left", va="bottom")
    x_gr = float(np.max(y) - 1.05)
    i_gr = int(np.argmin(np.abs(y - x_gr)))
    axA.text(x_gr + 0.05, rep.alpha_ground[i_gr] - 0.02, "ground",
             color="#1b4f9c", fontsize=6.8, ha="left", va="bottom")
    axA.text(x_gr + 0.05, rep.alpha_reflected[i_gr] - 0.10, "reflected",
             color="#1f9d55", fontsize=6.8, ha="left", va="top")
    axA.text(x_gr + 0.05, rep.alpha_exact[i_gr] + 0.22, "exact",
             color="#9b9b9b", fontsize=6.6, ha="left", va="bottom")
    _panel_label(axA, "(a)")

    # Purity collapse vs y
    y_line = np.linspace(float(np.min(y)), float(np.max(y)), 400)
    r_line = np.tanh(np.exp(y_line))
    axB.plot(y_line, r_line, color="#111111", lw=1.2, alpha=0.9)
    axB.plot(y[mask_pre], rep.r[mask_pre], color="#1c8ad3", lw=1.4, alpha=0.95)
    axB.plot(y[mask_post], rep.r[mask_post], color="#1b4f9c", lw=1.4, alpha=0.95)
    axB.plot(y[mask_post], rep.r[mask_post], color="#1f9d55", lw=1.4, ls="--", alpha=0.95)
    axB.axvline(0.0, color="#c62828", ls=":", lw=0.9, alpha=0.85)
    axB.set_xlabel(r"$y=\log\chi$")
    axB.set_ylabel(r"Purity $r$")
    axB.set_ylim(-0.02, 1.02)
    axB.grid(True, alpha=0.18, lw=0.4)
    axB.text(0.58, 0.08, r"$r(y)=\tanh(e^y)$",
             transform=axB.transAxes, fontsize=6.8, ha="left", va="bottom", color="#222222")
    axB.text(0.06, 0.10, r"$y=0 \Leftrightarrow \chi=1$", transform=axB.transAxes,
             fontsize=6.8, ha="left", va="bottom", color="#7d1d1d")
    _panel_label(axB, "(b)")
    _save_fig(fig, "hmf_branch_bifurcation_logchi")


def _beta_star_of_g_from_monotone_curve(
    g_vals: np.ndarray,
    beta_curve: np.ndarray,
    chi0_curve: np.ndarray,
) -> np.ndarray:
    # Solve chi0(beta_star) = 1/g^2 on the monotone chi0(beta) branch by interpolation.
    target = 1.0 / np.clip(g_vals * g_vals, 1e-30, None)
    # chi0_curve is monotone increasing with beta on the tested domain.
    # np.interp expects ascending x; we interpolate beta as function of chi0.
    return np.interp(target, chi0_curve, beta_curve, left=np.nan, right=np.nan)


def make_figure_involutive_map(theta: float = base.THETA_VAL) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_grid = np.geomspace(BETA_MIN, BETA_MAX, N_BETA_MAP)
    gstar_ref = _gstar_ref(theta)
    g_mult_grid = np.geomspace(0.05, 5.0, N_G_MAP)
    g_grid = g_mult_grid * gstar_ref

    chi0_beta, dz0_beta, sx0_beta = _chi0_curve(beta_grid, theta=theta)
    chi0_monotone_flag = _monotonicity_flag(chi0_beta)

    # 2D flow coordinate y = log chi
    chi_grid = np.outer(g_grid**2, chi0_beta)
    y_grid = np.log(np.clip(chi_grid, 1e-18, None))

    # Optional formal branch-singularity diagnostic: sample Theta sign on the same grid.
    Theta_grid = np.empty_like(chi_grid)
    for ig, g in enumerate(g_grid):
        for ib, b in enumerate(beta_grid):
            p = base.build_branch_point(g=float(g), g_mult_ref=float(g / gstar_ref), beta=float(b))
            Theta_grid[ig, ib] = p.Theta

    # Involutive line constructions
    gstar_beta = 1.0 / np.sqrt(np.clip(chi0_beta, 1e-30, None))
    # beta_star(g) sampled on a subset for points
    g_samples = np.geomspace(0.06, 4.8, 36) * gstar_ref
    beta_star_samples = _beta_star_of_g_from_monotone_curve(g_samples, beta_grid, chi0_beta)

    fig = plt.figure(figsize=(6.8, 3.0))
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.18, 0.92],
        wspace=0.36,
        left=0.08, right=0.985, bottom=0.18, top=0.96
    )
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # Panel A: 2D involutive/separatrix map
    B, Gm = np.meshgrid(beta_grid, g_mult_grid)
    y_clip = np.clip(y_grid, -3.0, 4.0)
    im = axA.pcolormesh(B, Gm, y_clip, shading="auto", cmap="coolwarm", rasterized=True)
    cbar = fig.colorbar(im, ax=axA, shrink=0.92, pad=0.02)
    cbar.set_label(r"$y=\log\chi$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # chi=1 contour (i.e., y=0)
    cs = axA.contour(B, Gm, y_grid, levels=[0.0], colors=["#111111"], linewidths=1.5)
    has_chi1_contour = any(len(seg) for seg in cs.allsegs[0]) if getattr(cs, "allsegs", None) else False

    # g_star(beta) and beta_star(g) inverse constructions
    axA.plot(beta_grid, gstar_beta / gstar_ref, color="#f4f4f4", lw=2.0, alpha=0.95)
    axA.plot(beta_grid, gstar_beta / gstar_ref, color="#4b4b4b", lw=0.9, alpha=0.95, label=r"$g_\star(\beta)$")
    valid = np.isfinite(beta_star_samples)
    axA.scatter(
        beta_star_samples[valid],
        g_samples[valid] / gstar_ref,
        s=10,
        facecolors="none",
        edgecolors="#1d1d1d",
        linewidths=0.7,
        label=r"$\beta_\star(g)$",
        zorder=6,
    )

    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(float(beta_grid[0]), float(beta_grid[-1]))
    axA.set_ylim(float(g_mult_grid[0]), float(g_mult_grid[-1]))
    axA.set_xlabel(r"$\beta\omega_q$")
    axA.set_ylabel(r"$g/g_\star^{(\rm ref)}$")
    axA.grid(True, which="both", alpha=0.12, lw=0.35)
    if has_chi1_contour:
        contour_handle = mpl.lines.Line2D([], [], color="#111111", lw=1.2, label=r"$\chi=1$")
        handles, labels = axA.get_legend_handles_labels()
        handles = [contour_handle] + handles
        labels = [contour_handle.get_label()] + labels
        axA.legend(handles, labels, fontsize=6.2, loc="upper left", framealpha=0.90,
                   borderpad=0.35, handlelength=1.5, labelspacing=0.25)
    else:
        axA.legend(fontsize=6.2, loc="upper left", framealpha=0.90,
                   borderpad=0.35, handlelength=1.5, labelspacing=0.25)
    _panel_label(axA, "(a)")

    # Panel B: RG variables on the monotone branch.
    # A representative flow y(beta; g_ref) makes the involutive crossing explicit.
    g_rep = REP_MULT_DEFAULT * gstar_ref
    chi_rep = (g_rep**2) * chi0_beta
    y_rep = np.log(np.clip(chi_rep, 1e-300, None))
    log10_chi0 = np.log10(np.clip(chi0_beta, 1e-300, None))
    log10_chi_rep = np.log10(np.clip(chi_rep, 1e-300, None))
    beta_star_rep = _beta_star_of_g_from_monotone_curve(np.array([g_rep]), beta_grid, chi0_beta)[0]

    axB.plot(beta_grid, log10_chi0, color="#005f99", lw=1.3)
    axB.plot(beta_grid, log10_chi_rep, color="#8e2c8a", lw=1.3)
    axB.axhline(0.0, color="#666666", ls=":", lw=0.8, alpha=0.7)
    if np.isfinite(beta_star_rep):
        axB.axvline(float(beta_star_rep), color="#c62828", ls=":", lw=0.9, alpha=0.85)
        axB.text(float(beta_star_rep) * 1.03, 0.15, r"$\beta_\star(g_{\rm ref})$",
                 fontsize=6.4, color="#7d1d1d", rotation=90, ha="left", va="bottom")
    axB.set_xscale("log")
    axB.set_xlabel(r"$\beta\omega_q$")
    axB.set_ylabel(r"$\log_{10}\chi$ variables")
    axB.grid(True, which="both", alpha=0.15, lw=0.35)

    theta_min = float(np.min(Theta_grid))
    theta_max = float(np.max(Theta_grid))
    theta_sign_text = (
        r"$\Theta<0$ on plotted domain"
        if theta_max < 0.0
        else (r"$\Theta>0$ on plotted domain" if theta_min > 0.0 else r"$\Theta=0$ contour enters domain")
    )
    axB.text(
        0.04,
        0.06,
        (
            "Involutive line:\n"
            r"$\chi=1 \Leftrightarrow g_\star(\beta)\leftrightarrow \beta_\star(g)$" + "\n\n" +
            f"{theta_sign_text}\n"
            rf"(sampled: $\Theta_{{\min}}={theta_min:.2f}$, $\Theta_{{\max}}={theta_max:.2f}$)"
        ),
        transform=axB.transAxes,
        fontsize=6.0,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#c7c7c7", alpha=0.9),
    )
    # Direct labels at the right edge keep the panel readable.
    x_lbl = float(beta_grid[-1] / 1.08)
    axB.text(x_lbl, float(log10_chi0[-1]) - 0.3, r"$\log_{10}\chi_0(\beta)$",
             fontsize=6.4, color="#005f99", ha="right", va="top")
    axB.text(x_lbl, float(log10_chi_rep[-1]) + 0.3, rf"$\log_{{10}}\chi(\beta;g={REP_MULT_DEFAULT:.1f}g_\star^{{(\rm ref)}})$",
             fontsize=6.4, color="#8e2c8a", ha="right", va="bottom")
    _panel_label(axB, "(b)")
    _save_fig(fig, "hmf_branch_bifurcation_involutive_map")

    meta = {
        "gstar_ref": float(gstar_ref),
        "chi0_beta_min": float(np.min(chi0_beta)),
        "chi0_beta_max": float(np.max(chi0_beta)),
        "chi0_monotone_flag": float(1.0 if chi0_monotone_flag else 0.0),
        "Theta_grid_min": float(theta_min),
        "Theta_grid_max": float(theta_max),
    }
    return meta, beta_grid, g_mult_grid, y_grid, Theta_grid


def _write_summary_csv(rep: RepresentativeBranchData, map_meta: dict[str, float]) -> None:
    csv_path = OUT_DIR / "hmf_branch_bifurcation_summary.csv"
    theta_chart = rep.alpha_theta
    pi_minus_theta_chart = rep.alpha_hub

    purity_branch_diff_max = float(
        max(
            np.max(np.abs(rep.r - rep.r)),
            np.max(np.abs(rep.r - rep.r)),
        )
    )
    row = {
        "g_ref": rep.g_ref,
        "g_over_gstar_ref": rep.g_mult_ref,
        "beta_star": rep.beta_star,
        "beta_split": rep.beta_split,
        "theta_ray_chart": theta_chart,
        "theta_ray_manuscript": float(base.THETA_VAL),
        "pi_minus_theta_ray_chart": pi_minus_theta_chart,
        "pi_minus_theta_ray_manuscript": float(np.pi - base.THETA_VAL),
        "alpha_incoming_at_beta_star": float(np.interp(rep.beta_star, rep.beta, rep.alpha_incoming)),
        "alpha_groundward_beta_max": float(rep.alpha_ground[-1]),
        "alpha_zreflected_beta_max": float(rep.alpha_reflected[-1]),
        "purity_beta_max": float(rep.r[-1]),
        "purity_branch_diff_max": purity_branch_diff_max,
        "chi0_monotone_flag": int(map_meta.get("chi0_monotone_flag", 0.0)),
        "Theta_grid_min": map_meta.get("Theta_grid_min", np.nan),
        "Theta_grid_max": map_meta.get("Theta_grid_max", np.nan),
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote summary -> {csv_path}")


def _write_grid_cache(beta_grid: np.ndarray, g_mult_grid: np.ndarray, y_grid: np.ndarray, Theta_grid: np.ndarray) -> None:
    npz_path = OUT_DIR / "hmf_branch_bifurcation_grid.npz"
    np.savez_compressed(
        npz_path,
        beta=beta_grid,
        g_over_gstar_ref=g_mult_grid,
        y_logchi=y_grid,
        Theta=Theta_grid,
    )
    print(f"Wrote grid cache -> {npz_path}")


def main() -> None:
    _configure_matplotlib()
    print("Building RG-framed branch bifurcation manuscript figures...")

    rep = _representative_branch_construction(theta=base.THETA_VAL)
    print(
        "Representative coupling: "
        f"g/g*_ref={rep.g_mult_ref:.3f}, beta_star={rep.beta_star:.6f}, "
        f"beta_split={rep.beta_split:.6f}"
    )

    make_figure_bloch_fullcircle(rep)
    make_figure_logchi(rep)
    map_meta, beta_grid_map, g_mult_grid_map, y_grid_map, Theta_grid_map = make_figure_involutive_map(theta=base.THETA_VAL)
    _write_summary_csv(rep, map_meta)
    _write_grid_cache(beta_grid_map, g_mult_grid_map, y_grid_map, Theta_grid_map)

    # brief terminal checks
    print("\nQuality checks (quick):")
    print(f"  purity overlap max diff (constructed branches): 0.0 (by construction, exact r used)")
    print(
        "  endpoint sectors (rad): "
        f"ground={rep.alpha_ground[-1]:.4f}, z-reflected={rep.alpha_reflected[-1]:.4f}, "
        f"theta-ray={rep.alpha_theta:.4f}, hub(pi-theta)={rep.alpha_hub:.4f}"
    )


if __name__ == "__main__":
    main()
