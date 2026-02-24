# -*- coding: utf-8 -*-
"""
fig5_convergence.py  –  Figure 5: Representability bottleneck
=============================================================
Three-panel figure:
  (a) p_00(ED) vs n_cut at 4 representative (g, beta) points,
      with disc. analytic as horizontal reference line
  (b) |p_00(ED, n_cut) - p_00(disc. analytic)| vs n_cut (semilog)
      — shows convergence RATE toward the correct analytic target
  (c) 2D heatmap of |p_00(ED, n_cut=60) - p_00(disc. analytic)| in
      (g, beta) space with chi=1 contour: where does ED fail even
      at maximum n_cut?

Data required: data/pi4_ncut_convergence.csv
Output: ../../manuscript/figures/hmf_fig5_convergence.pdf + .png
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

DATA    = Path(__file__).parent / "data" / "pi4_ncut_convergence.csv"
FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": False, "figure.dpi": 200, "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
})

# Fixed physics parameters
OMEGA_Q    = 2.0
Q_STRENGTH = 10.0
TAU_C      = 1.0
THETA      = np.pi / 4
OMEGA_MIN  = 0.5
OMEGA_MAX  = 8.0
N_MODES    = 2    # n_modes used throughout


# =============================================================================
# ANALYTIC HMF FORMULA  (same implementation as fig4_2d_landscape.py)
# =============================================================================

def _channels(beta, omega_q, theta, omegas, g2_k, n_grid=601):
    """
    Compute g-independent channel amplitudes (sp0, sm0, dz0) for a
    given (beta, mode set).  Returns (sp0, sm0, dz0).
    """
    u  = np.linspace(0.0, beta, n_grid)
    k0 = np.zeros(n_grid)
    for wk, g2k in zip(omegas, g2_k):
        den = np.sinh(0.5 * beta * wk)
        if abs(den) > 1e-14:
            k0 += g2k * np.cosh(wk * (0.5 * beta - u)) / den
        else:
            k0 += g2k * 2.0 / max(beta * wk, 1e-14)

    def lap(ws):
        return float(np.trapezoid(k0 * np.exp(ws * u), u))
    def res(ws):
        return float(np.trapezoid((beta - u) * k0 * np.exp(ws * u), u))

    c, s = np.cos(theta), np.sin(theta)
    k00, k0p, k0m = lap(0.0), lap(+omega_q), lap(-omega_q)
    r0p, r0m      = res(+omega_q), res(-omega_q)

    sp0 = (c * s / omega_q) * ((1.0 + np.exp(+beta * omega_q)) * k00 - 2.0 * k0p)
    sm0 = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * k00 - 2.0 * k0m)
    dz0 = s**2 * 0.5 * (r0p - r0m)
    return float(sp0), float(sm0), float(dz0)


def _p00(g, sp0, sm0, dz0, beta, omega_q):
    """Compute p_00 from pre-computed (g-independent) channel amplitudes."""
    g2 = g * g
    sp, sm, dz = g2 * sp0, g2 * sm0, g2 * dz0
    chi   = float(np.sqrt(max(dz**2 + sp * sm, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-12 else 1.0
    a     = 0.5 * beta * omega_q
    t     = float(np.tanh(a))
    u_v   = gamma * dz
    den   = 1.0 - u_v * t
    if abs(den) < 1e-14:
        den = 1e-14 * (1.0 if den >= 0 else -1.0)
    mz    = (u_v - t) / den
    return float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))


def compute_disc_analytic_grid(g_vals, beta_vals, omega_q, theta,
                               omega_min, omega_max, n_modes, q_strength, tau_c):
    """Return p_00(disc. analytic) array of shape (n_beta, n_g)."""
    if n_modes == 1:
        omegas  = np.array([0.5 * (omega_min + omega_max)])
        d_omega = omega_max - omega_min
    else:
        omegas  = np.linspace(omega_min, omega_max, n_modes)
        d_omega = float(omegas[1] - omegas[0])
    j_vals = q_strength * tau_c * omegas * np.exp(-tau_c * omegas)
    g2_k   = np.maximum(j_vals, 0.0) * d_omega

    grid = np.zeros((len(beta_vals), len(g_vals)))
    for i, beta in enumerate(beta_vals):
        sp0, sm0, dz0 = _channels(beta, omega_q, theta, omegas, g2_k)
        for j, g in enumerate(g_vals):
            grid[i, j] = _p00(g, sp0, sm0, dz0, beta, omega_q)
    return grid


# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading pi4_ncut_convergence.csv ...")
df_all = pd.read_csv(DATA)
df2    = df_all[df_all.n_modes == N_MODES].copy()
ncut_vals = sorted(df2.n_cut.unique())
N_CUT_MAX = ncut_vals[-1]   # 60
N_CUT_MIN = ncut_vals[0]    # 6

g_vals    = sorted(df2.g.unique())
beta_vals = sorted(df2.beta.unique())
# Convert to dimensionless βω_q for all axes and overlays
beta_plot = [b * OMEGA_Q for b in beta_vals]

print(f"  n_modes={N_MODES}: {len(df2)} rows, n_cut={ncut_vals}")

# Helper: nearest grid value
def nearest(arr, val):
    arr = np.asarray(arr)
    return arr[np.argmin(np.abs(arr - val))]

# ---------- chi map from n_cut=max slice (discrete, used for chi_max lookup) ---
df_max    = df2[df2.n_cut == N_CUT_MAX].copy()
chi_pivot = df_max.pivot(index="beta", columns="g", values="chi")
chi_map   = chi_pivot.values
G, B      = np.meshgrid(g_vals, beta_plot)

# ---------- continuum chi_0(beta) and g*(beta) from 800-mode bath -------------
# Use a dense approximation to the physical Ohmic continuum (same as fig4).
print("Computing continuum chi_0(beta) for panel (c) overlays ...")
omegas_cont, g2k_cont = (lambda nm=800, omin=0.02, omax=20.0:
    (np.linspace(omin, omax, nm),
     np.maximum(Q_STRENGTH * TAU_C * np.linspace(omin, omax, nm) *
                np.exp(-TAU_C * np.linspace(omin, omax, nm)), 0.0) *
     float(np.linspace(omin, omax, nm)[1] - np.linspace(omin, omax, nm)[0])))()
chi0_cont_arr = np.zeros(len(beta_vals))
for i, beta in enumerate(beta_vals):
    u  = np.linspace(0.0, beta, 601)
    k0 = np.zeros(601)
    for wk, g2k in zip(omegas_cont, g2k_cont):
        den = np.sinh(0.5 * beta * wk)
        k0 += g2k * (np.cosh(wk * (0.5 * beta - u)) / den
                     if abs(den) > 1e-14 else 2.0 / max(beta * wk, 1e-14))
    c, s = np.cos(THETA), np.sin(THETA)
    k00  = float(np.trapezoid(k0, u))
    k0p  = float(np.trapezoid(k0 * np.exp(+OMEGA_Q * u), u))
    k0m  = float(np.trapezoid(k0 * np.exp(-OMEGA_Q * u), u))
    r0p  = float(np.trapezoid((beta - u) * k0 * np.exp(+OMEGA_Q * u), u))
    r0m  = float(np.trapezoid((beta - u) * k0 * np.exp(-OMEGA_Q * u), u))
    sp0  = (c * s / OMEGA_Q) * ((1.0 + np.exp(+beta * OMEGA_Q)) * k00 - 2.0 * k0p)
    sm0  = (c * s / OMEGA_Q) * ((1.0 + np.exp(-beta * OMEGA_Q)) * k00 - 2.0 * k0m)
    dz0  = s**2 * 0.5 * (r0p - r0m)
    chi0_cont_arr[i] = float(np.sqrt(max(dz0**2 + sp0 * sm0, 0.0)))

g_star_arr    = np.where(chi0_cont_arr > 0,
                         1.0 / np.sqrt(chi0_cont_arr), np.nan)
chi_cont_map  = G**2 * chi0_cont_arr[:, np.newaxis]   # (n_beta, n_g) in scaled coords

# ---------- disc. analytic grid ------------------------------------------
print("Computing disc. analytic p_00 (n_modes=2) ...")
disc_grid = compute_disc_analytic_grid(g_vals, beta_vals, OMEGA_Q, THETA,
                                       OMEGA_MIN, OMEGA_MAX,
                                       N_MODES, Q_STRENGTH, TAU_C)
print(f"  disc analytic range: [{disc_grid.min():.4f}, {disc_grid.max():.4f}]")

# disc analytic at n_cut_max (for error map)
disc_at_beta = {b: disc_grid[i, :] for i, b in enumerate(beta_vals)}

# ED at n_cut=max (for error map in panel c)
ed_max_pivot = df_max.pivot(index="beta", columns="g", values="ed_p00")
ed_max_map   = ed_max_pivot.values

# Residual map |p_00(ED,60) - p_00(disc.analytic)|
resid_map = np.abs(ed_max_map - disc_grid)
print(f"  max residual (n_cut={N_CUT_MAX} vs disc. analytic): {resid_map.max():.4f}")


# =============================================================================
# REPRESENTATIVE POINTS  (chosen for diverse convergence behaviour)
# Convergence is driven by the mode displacement alpha ~ g * c_1 / omega_1
# c_1 = sqrt(J(omega_1)*d_omega), omega_1=0.5, c_1 ~ 4.76
# => alpha ~ 9.52 * g, n_cut needed ~ alpha^2 ~ 90 * g^2
# =============================================================================

# Target (g, beta, label, color)
REPR_TARGETS = [
    (0.10, 0.50,  r"Weak  ($g{=}0.10,\,\beta\omega_q{=}1.0$)",    "#2166ac"),
    (0.35, 2.00,  r"Moderate ($g{=}0.35,\,\beta\omega_q{=}4.0$)",  "#74c476"),
    (0.70, 4.00,  r"Strong ($g{=}0.70,\,\beta\omega_q{=}8.0$)",    "#fd8d3c"),
    (1.36, 8.00,  r"Ultrastrong ($g{=}1.36,\,\beta\omega_q{=}16$)","#a50026"),
]

repr_points = []
for g_t, b_t, label, color in REPR_TARGETS:
    g_n = nearest(g_vals, g_t)
    b_n = nearest(beta_vals, b_t)
    sub = df2[(np.isclose(df2.g, g_n, atol=1e-6)) &
              (np.isclose(df2.beta, b_n, atol=1e-4))].sort_values("n_cut")

    i_b = np.argmin(np.abs(np.array(beta_vals) - b_n))
    j_g = np.argmin(np.abs(np.array(g_vals)    - g_n))
    p00_disc = float(disc_grid[i_b, j_g])
    chi_max  = float(chi_map[i_b, j_g])

    repr_points.append((g_n, b_n, label, color, sub, p00_disc, chi_max))
    alpha_est = g_n * np.sqrt(Q_STRENGTH * TAU_C * 0.5 * np.exp(-TAU_C * 0.5) *
                               (OMEGA_MAX - OMEGA_MIN)) / 0.5
    print(f"  g={g_n:.3f}, b={b_n:.3f}, chi={chi_max:.3f}, p00_disc={p00_disc:.4f}, "
          f"alpha_est={alpha_est:.2f}")


# =============================================================================
# FIGURE ASSEMBLY
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
fig.subplots_adjust(wspace=0.52)

# ── Panel (a): p_00(ED, n_cut) vs n_cut ──────────────────────────────────────
ax = axes[0]
for g_n, b_n, label, color, sub, p00_disc, chi_max in repr_points:
    nc = sub["n_cut"].values
    pp = sub["ed_p00"].values
    ax.plot(nc, pp, color=color, lw=1.4, marker="o", ms=3.0, label=label)
    ax.axhline(p00_disc, color=color, lw=0.8, ls="--", alpha=0.6)

ax.set_xlabel(r"Fock cutoff $n_{\rm cut}$")
ax.set_ylabel(r"$p_{00}$ (ED)")
ax.set_xscale("log")
ax.set_xlim(ncut_vals[0] * 0.9, N_CUT_MAX * 1.15)
ax.legend(fontsize=5.5, framealpha=0.85, loc="best")
ax.text(0.05, 0.97, "(a)", transform=ax.transAxes,
        fontweight="bold", fontsize=9, va="top")
ax.set_title(r"$p_{00}$ vs $n_{\rm cut}$" + "\n" + "(dashed = analytic target)",
             fontsize=7.5, pad=3)

# ── Panel (b): convergence error |p_00(ED) - disc.analytic| vs n_cut ─────────
ax = axes[1]
for g_n, b_n, label, color, sub, p00_disc, chi_max in repr_points:
    nc  = sub["n_cut"].values
    err = np.abs(sub["ed_p00"].values - p00_disc)
    # Avoid log(0): floor at 1e-5
    err = np.maximum(err, 1e-5)
    ax.semilogy(nc, err, color=color, lw=1.4, marker="o", ms=3.0)

ax.set_xlabel(r"Fock cutoff $n_{\rm cut}$")
ax.set_ylabel(r"$|p_{00}^{\rm ED} - p_{00}^{\rm disc.}|$")
ax.set_xscale("log")
ax.set_xlim(ncut_vals[0] * 0.9, N_CUT_MAX * 1.15)
ax.axhline(0.005, color="k", lw=0.8, ls=":", alpha=0.6,
           label=r"$\varepsilon=0.005$")
ax.legend(fontsize=6.5, loc="best", framealpha=0.85)
ax.text(0.05, 0.97, "(b)", transform=ax.transAxes,
        fontweight="bold", fontsize=9, va="top")
ax.set_title(r"Convergence error vs $n_{\rm cut}$", fontsize=7.5, pad=3)

# ── Panel (c): 2D residual map |p_00(ED,60) - p_00(disc.analytic)| ───────────
ax = axes[2]

# Log scale for the colormap to reveal small errors
log_resid = np.log10(np.maximum(resid_map, 1e-5))

im = ax.pcolormesh(G, B, log_resid,
                   cmap="YlOrRd", vmin=-5, vmax=np.log10(resid_map.max()),
                   shading="nearest", rasterized=True)
cb = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
cb.set_label(r"$\log_{10}|p_{00}^{\rm ED} - p_{00}^{\rm disc.}|$", fontsize=7)
cb.ax.tick_params(labelsize=6)

# χ_cont = 1 contour (continuum bath — consistent with fig4)
cs = ax.contour(G, B, chi_cont_map, levels=[1.0], colors="white",
                linewidths=1.2, linestyles="--")
ax.clabel(cs, fmt=r"$\chi{=}1$", fontsize=5.5, inline=True, inline_spacing=2)

# Mark representative points (β converted to βω_q)
for g_n, b_n, _, color, _, _, _ in repr_points:
    ax.plot(g_n, b_n * OMEGA_Q, marker="*", ms=8, color=color,
            markeredgecolor="white", markeredgewidth=0.5, zorder=5)

ax.set_xlabel(r"coupling $g/\omega_q$")
ax.set_ylabel(r"$\beta\omega_q$")
ax.set_yscale("log")
ax.set_ylim(beta_plot[0], beta_plot[-1])
ax.set_xlim(g_vals[0], g_vals[-1])
ax.set_title(r"Residual map ($n_{\rm cut}=60$)", fontsize=7.5, pad=3)
ax.text(0.04, 0.97, "(c)", transform=ax.transAxes,
        fontweight="bold", color="white", fontsize=9, va="top")

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig5_convergence{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")

plt.close(fig)
print("Done.")
