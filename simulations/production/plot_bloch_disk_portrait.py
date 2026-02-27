# -*- coding: utf-8 -*-
"""
plot_bloch_disk_portrait.py  --  Figure 5: Coupling-plane Bloch disk portrait
==============================================================================
Two-panel publication figure showing how the qubit mean-force state traces
trajectories in the Bloch disk.

  Panel (a): g-sweep — state trajectory (mx, mz) for four fixed temperatures
             as coupling strength g increases from 0. The state starts at the
             bare thermal point on the -z axis and sweeps toward a
             temperature-dependent attractor as chi -> inf.

  Panel (b): beta-sweep — state trajectory for four fixed couplings as beta
             increases (system cools). For large g the state quickly reaches
             near-unit purity and then precesses around the disk boundary
             as the strong-coupling attractor direction drifts with temperature.
             A reference arrow marks the bare coupling direction f-hat, to
             check whether any trajectory passes through it.

Co-moving limit: curves of fixed xi = g/g_star(beta) keep chi = xi^2 constant
(fixed purity r = tanh(chi)) while the direction n-hat(beta) evolves — an arc
at fixed Bloch-radius that sweeps the boundary for large xi.

Bath model and coupling: Ohmic, theta = pi/4 (locked). All analytic, no data.

Output: manuscript/figures/hmf_bloch_disk_portrait.{pdf,png}
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from fig1_chi_theory import bloch_ohmic, get_chi0

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Publication-quality rcParams (PRL two-column style) ──────────────────────
mpl.rcParams.update({
    "font.family"        : "serif",
    "font.size"          : 8,
    "axes.labelsize"     : 9,
    "axes.titlesize"     : 8,
    "legend.fontsize"    : 7,
    "xtick.labelsize"    : 7,
    "ytick.labelsize"    : 7,
    "text.usetex"        : True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "figure.dpi"         : 200,
    "lines.linewidth"    : 1.2,
    "axes.linewidth"     : 0.7,
    "xtick.major.width"  : 0.7,
    "ytick.major.width"  : 0.7,
    "xtick.major.size"   : 3.0,
    "ytick.major.size"   : 3.0,
})

THETA_VAL = np.pi / 4   # coupling angle (locked throughout)
OMEGA_MIN_VAL = 0.04    # IR cutoff to eliminate exponentially scaling DC component at low T
N_TRAJ    = 8000        # points per trajectory

# ── Panel (a): temperature values and palette ─────────────────────────────────
# Muted version of a warm-cool palette (desaturated from the original RdBu)
BETAS_A      = [0.5,       1.0,       2.0,       3.0      ]
COLORS_A     = ["#c66b5a", "#e0a882", "#7a6aac", "#3a78b0"]
LINESTYLES_A = ["-",       "--",      "-.",      ":"      ]
LABELS_A     = [r"$\beta\omega_q=0.5$", r"$\beta\omega_q=1.0$",
                r"$\beta\omega_q=2.0$", r"$\beta\omega_q=3.0$"]

# ── Panel (b): coupling values and palette ────────────────────────────────────
# Fixed g values for the beta-sweep; colors ordered warm -> cool by coupling
# g values chosen as multiples of g_star at beta=2 (reference temperature)
BETA_REF   = 2.0   # reference beta for defining coupling scale

print("Pre-computing chi0...")
chi0_ref, _, _ = get_chi0(BETA_REF, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)
gstar_ref      = 1.0 / np.sqrt(chi0_ref) if chi0_ref > 0 else np.inf
print(f"  beta_ref={BETA_REF}: chi0={chi0_ref:.5f}, g_star={gstar_ref:.4f}")

# We use direct g values now - four lines as requested
G_VALS_B    = [0.2, 0.5, 1.0, 2.0]
cmap = mpl.colormaps["viridis"]
COLORS_B    = [cmap(i) for i in np.linspace(0, 0.9, len(G_VALS_B))]
LINESTYLES_B= ["-"] * len(G_VALS_B)
LABELS_B = [r"$g = {:.1f}$".format(m) for m in G_VALS_B]

# ── Pre-compute chi0 and g_star for panel (a) temperatures ───────────────────
print("Computing chi0 for panel (a) temperatures...")
chi0_a  = []
gstar_a = []
for beta in BETAS_A:
    c0, _, _ = get_chi0(beta, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)
    chi0_a.append(c0)
    gstar_a.append(1.0 / np.sqrt(c0) if c0 > 0 else np.nan)
    print(f"  beta={beta}: chi0={c0:.5f}, g_star={gstar_a[-1]:.4f}")

# ── Figure layout: 2-panel vertical stack ─────────────────────────────────────
# Height reduced to 5.2 to leave ample room for caption and headers in RevTeX
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.25, 5.2), constrained_layout=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (a): g-sweep at fixed beta
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_a

# Physical boundary (r=1) in linear space (m_perp vs mz)
# Clip boundary to x=0.52 to prevent whitespace issues
X_bound = np.linspace(0, 0.52, 200)
Z_bound = -np.sqrt(np.clip(1.0 - X_bound**2, 0, 1))
ax.plot(X_bound, Z_bound, color="black", lw=1.1, zorder=1)
ax.fill_between(X_bound, Z_bound, 0.1, facecolor="#f9f9f9", zorder=0)

# Axis reference lines
ax.plot([0, 0.52], [0, 0], color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)
ax.plot([0, 0], [-1.12, 0.05], color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)

# Determine attractor for the coldest line in Panel (a) (blue line, beta=3.0)
_, _, mx_inf_cold, mz_inf_cold = bloch_ohmic(np.array([1000.0]), BETAS_A[-1], 
                                        theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)
ray_slope = mz_inf_cold[0] / mx_inf_cold[0]

# Reference ray: straight physical ray terminating exactly at the Bloch radius intersection
# Intersection is at (mx_inf, mz_inf). Stop exactly there.
ray_x_max = float(mx_inf_cold[0])
ray_x_phys = np.linspace(0, ray_x_max, 201) # higher res for a smooth termination
ray_z_phys = ray_slope * ray_x_phys
ax.plot(ray_x_phys, ray_z_phys, color="#888888", ls="--", lw=0.9, alpha=0.6, zorder=1)
# Label the ray
ax.text(0.10, -0.65, r"$\varphi$", 
        fontsize=10, color="#444444", ha="left", va="top")

# Trajectories (g-sweep)
for beta, col, ls, lab, gs_val in zip(BETAS_A, COLORS_A, LINESTYLES_A,
                                       LABELS_A, gstar_a):
    g_arr = np.linspace(0.0, 6.0 * gs_val, N_TRAJ)
    phi, r, mx, mz = bloch_ohmic(g_arr, beta, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)

    # Aggressive clipping of data to axis limits to prevent whitespace expansion
    mask = (mx >= 0) & (mx <= 0.505)
    ax.plot(mx[mask], mz[mask],
            color=col, ls=ls, lw=1.6, label=lab, zorder=3)

    # Markers (only plot if within canvas limits)
    def safe_marker(x, z, shape, size):
        if 0 <= x <= 0.505:
            ax.plot(x, z, shape, color=col, ms=size, zorder=6,
                    markeredgewidth=0.7, markeredgecolor="white")

    # g=0
    safe_marker(0, mz[0], "o", 5)

    # g=g_star
    idx_star = np.argmin(np.abs(g_arr - gs_val))
    safe_marker(mx[idx_star], mz[idx_star], "*", 9)

    # g -> inf
    _, _, mx_inf, mz_inf = bloch_ohmic(np.array([200.0 * gs_val]),
                                        beta, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)
    safe_marker(mx_inf[0], mz_inf[0], "s", 5)

# Marker legend entries (standardized colors)
ax.plot([], [], "ko", ms=4.5, markeredgewidth=0.5, markeredgecolor="white", label=r"$g=0$")
ax.plot([], [], "k*", ms=8.0, markeredgewidth=0.5, markeredgecolor="white", label=r"$g=g_\star$")
ax.plot([], [], "ks", ms=4.5, markeredgewidth=0.5, markeredgecolor="white", label=r"$g\to\infty$")

ax.legend(fontsize=6.2, loc="upper right", framealpha=0.92,
          ncol=2, columnspacing=0.5, handlelength=1.5,
          borderpad=0.3, labelspacing=0.2)

ax.set_xlim(0.00, 0.5)
ax.set_ylim(-1.05, 0.0)
ax.set_xscale("function", functions=(np.sqrt, np.square))
# Update ticks for square root scale visibility
ax.set_xticks([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"])

ax.set_box_aspect(1)
ax.set_xlabel(r"$m_\perp$")
ax.set_ylabel(r"$m_z$")
ax.text(0.04, 0.94, "(a)", transform=ax.transAxes,
        fontweight="bold", va="top", ha="left", fontsize=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel (b): beta-sweep at fixed g
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_b

# Physical boundary
ax.plot(X_bound, Z_bound, color="black", lw=1.1, zorder=1)
ax.fill_between(X_bound, Z_bound, 0.1, facecolor="#f9f9f9", zorder=0)

# Axis reference lines
ax.plot([0, 0.52], [0, 0], color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)
ax.plot([0, 0], [-1.12, 0.05], color="black", ls=":", lw=0.5, alpha=0.30, zorder=1)

# Reference ray (same as panel a)
ax.plot(ray_x_phys, ray_z_phys, color="#888888", ls="--", lw=0.8, alpha=0.6, zorder=1)

# Beta sweep: beta grid from high T to low T (geom-spaced for even coverage)
N_BETA = 400
beta_arr = np.geomspace(0.05, 25.0, N_BETA)

# Pre-compute chi0 along the beta grid
chi0_grid  = np.array([get_chi0(b, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)[0] for b in beta_arr])
gstar_grid = np.where(chi0_grid > 0, 1.0 / np.sqrt(chi0_grid), np.nan)

LABELS_B = [r"$g = {:.1f}$".format(m) for m in G_VALS_B]

for g, col, ls, lab in zip(G_VALS_B, COLORS_B, LINESTYLES_B, LABELS_B):
    # bloch_ohmic takes a scalar beta, so loop over beta values
    mx_traj = np.empty(N_BETA)
    mz_traj = np.empty(N_BETA)
    g_scalar = np.array([g])
    for k, beta_k in enumerate(beta_arr):
        _, _, mx_k, mz_k = bloch_ohmic(g_scalar, beta_k, theta=THETA_VAL, omega_min=OMEGA_MIN_VAL)
        mx_traj[k] = float(mx_k[0])
        mz_traj[k] = float(mz_k[0])

    mask = (mx_traj >= 0) & (mx_traj <= 0.505)
    ax.plot(mx_traj[mask], mz_traj[mask],
            color=col, ls=ls, lw=1.6, label=lab, zorder=3)

    # beta -> 0 marker (circle): first point (hottest)
    if 0 <= mx_traj[0] <= 0.505:
        ax.plot(mx_traj[0], mz_traj[0], "o",
                color=col, ms=5, zorder=6,
                markeredgewidth=0.7, markeredgecolor="white")

    # beta = beta_star(g) marker (star): chi(g, beta_star) = 1
    chi_arr = g**2 * chi0_grid
    crossings = np.where(np.diff(np.sign(chi_arr - 1.0)))[0]
    if len(crossings) > 0:
        idx_star = crossings[0]
        if 0 <= mx_traj[idx_star] <= 0.505:
            ax.plot(mx_traj[idx_star], mz_traj[idx_star],
                    "*", color=col, ms=9, zorder=6,
                    markeredgewidth=0.7, markeredgecolor="white")

    # beta -> inf marker (square)
    if 0 <= mx_traj[-1] <= 0.505:
        ax.plot(mx_traj[-1], mz_traj[-1], "s", color=col, ms=4, zorder=6,
                markeredgewidth=0.5, markeredgecolor="white")

# Combined legend: coupling lines + marker explanations
handles_leg = []
for g_v, col, ls, lab in zip(G_VALS_B, COLORS_B, LINESTYLES_B, LABELS_B):
    handles_leg.append(
        mpl.lines.Line2D([], [], color=col, ls=ls, lw=1.4, label=lab)
    )
handles_leg += [
    mpl.lines.Line2D([], [], marker="o", color="k", ls="none", ms=4.5,
                     markeredgewidth=0.5, markeredgecolor="white",
                     label=r"$T\to\infty$"),
    mpl.lines.Line2D([], [], marker="*", color="k", ls="none", ms=8.0,
                     markeredgewidth=0.5, markeredgecolor="white",
                     label=r"$T=T_\star(g)$"),
    mpl.lines.Line2D([], [], marker="s", color="k", ls="none", ms=4.5,
                     markeredgewidth=0.5, markeredgecolor="white",
                     label=r"$T\to 0$"),
]
ax.legend(handles=handles_leg, fontsize=5.0, loc="upper right", framealpha=0.88,
          ncol=2, handlelength=1.5, borderpad=0.3, labelspacing=0.2)

ax.set_xlim(0.00, 0.5)
ax.set_ylim(-1.05, 0.0)
ax.set_xscale("function", functions=(np.sqrt, np.square))
ax.set_xticks([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"])

ax.set_box_aspect(1)
ax.set_xlabel(r"$m_\perp$")
ax.set_ylabel(r"$m_z$")
ax.text(0.04, 0.94, "(b)", transform=ax.transAxes,
        fontweight="bold", va="top", ha="left", fontsize=10)


# ── Save ──────────────────────────────────────────────────────────────────────
out = FIGURES / "hmf_bloch_disk_portrait.png"
# Explicitly use bbox_inches='tight' with small padding to crop whitespace
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved -> {out}")
print(f"Saved -> {out.with_suffix('.pdf')}")
