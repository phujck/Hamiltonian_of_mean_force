# -*- coding: utf-8 -*-
"""
fig4_2d_landscape.py  –  Figure 4: 2D qubit-state landscape (heatmaps)
=======================================================================
2×2 heatmap figure showing p_00(ED), p_00(discrete analytic), and signed
discrepancies across the full (g, β) plane at θ=π/4.

Data required: data/pi4_ncut_convergence.csv  (n_modes=2, n_cut=60)
Output: ../../manuscript/figures/hmf_fig4_2d_landscape.pdf + .png
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
    "text.usetex": False, "figure.dpi": 200,
    "axes.linewidth": 0.8,
})

# Fixed physics parameters (from CSV headers)
OMEGA_Q    = 2.0
Q_STRENGTH = 10.0
TAU_C      = 1.0
THETA      = np.pi / 4
OMEGA_MIN  = 0.5
OMEGA_MAX  = 8.0

# We use n_modes=2 ED data (highest n_cut = 60 for this mode count)
N_MODES_ED = 2
N_CUT_MAX  = 60


# =============================================================================
# 1. ANALYTIC HMF COMPUTATION
# =============================================================================

def _make_mode_set(n_modes, omega_min, omega_max, q_strength, tau_c):
    """Return (omegas, g2_k) arrays for the discrete bath."""
    if n_modes == 1:
        omegas  = np.array([0.5 * (omega_min + omega_max)])
        d_omega = omega_max - omega_min
    else:
        omegas  = np.linspace(omega_min, omega_max, n_modes)
        d_omega = float(omegas[1] - omegas[0])
    j_vals = q_strength * tau_c * omegas * np.exp(-tau_c * omegas)
    g2_k   = np.maximum(j_vals, 0.0) * d_omega
    return omegas, g2_k


def _kernel_laplace_res(beta, omegas, g2_k, omega_q, n_grid=601):
    """
    Compute the five kernel integrals for given (beta, mode set):
      K0(0), K0(+wq), K0(-wq), R0(+wq), R0(-wq)
    These are g-independent.  Returns (k00, k0p, k0m, r0p, r0m).
    """
    u = np.linspace(0.0, beta, n_grid)

    # Build k0(u) vectorised over modes
    k0 = np.zeros(n_grid)
    for wk, g2k in zip(omegas, g2_k):
        den = np.sinh(0.5 * beta * wk)
        if abs(den) > 1e-14:
            k0 += g2k * np.cosh(wk * (0.5 * beta - u)) / den
        else:
            k0 += g2k * 2.0 / max(beta * wk, 1e-14)

    # Laplace / resonant transforms via trapezoidal rule
    def lap(ws):
        return float(np.trapezoid(k0 * np.exp(ws * u), u))

    def res(ws):
        return float(np.trapezoid((beta - u) * k0 * np.exp(ws * u), u))

    return lap(0.0), lap(+omega_q), lap(-omega_q), res(+omega_q), res(-omega_q)


def _channels_from_integrals(k00, k0p, k0m, r0p, r0m, beta, omega_q, theta):
    """Convert kernel integrals to channel amplitudes (sp0, sm0, dz0)."""
    c, s = np.cos(theta), np.sin(theta)
    sp0 = (c * s / omega_q) * ((1.0 + np.exp(+beta * omega_q)) * k00 - 2.0 * k0p)
    sm0 = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * k00 - 2.0 * k0m)
    dz0 = s**2 * 0.5 * (r0p - r0m)
    return float(sp0), float(sm0), float(dz0)


def _p00_from_channels(g, sp0, sm0, dz0, beta, omega_q):
    """Compute p_00 from g-independent channel amplitudes."""
    g2 = g * g
    sp, sm, dz = g2 * sp0, g2 * sm0, g2 * dz0

    chi = float(np.sqrt(max(dz**2 + sp * sm, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-12 else 1.0

    a   = 0.5 * beta * omega_q
    t   = float(np.tanh(a))
    u_v = gamma * dz
    den = 1.0 - u_v * t
    if abs(den) < 1e-14:
        den = 1e-14 * (1.0 if den >= 0 else -1.0)

    mz  = (u_v - t) / den
    p00 = float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))
    return p00


def compute_analytic_grid(g_vals, beta_vals, omega_q, theta, omega_min, omega_max,
                           n_modes, q_strength, tau_c):
    """
    Compute p_00(analytic) for every (g, β) pair.  Returns a 2D array
    of shape (len(beta_vals), len(g_vals)).
    """
    omegas, g2_k = _make_mode_set(n_modes, omega_min, omega_max, q_strength, tau_c)
    p00_grid = np.zeros((len(beta_vals), len(g_vals)))

    for i, beta in enumerate(beta_vals):
        k00, k0p, k0m, r0p, r0m = _kernel_laplace_res(beta, omegas, g2_k, omega_q)
        sp0, sm0, dz0 = _channels_from_integrals(k00, k0p, k0m, r0p, r0m,
                                                  beta, omega_q, theta)
        for j, g in enumerate(g_vals):
            p00_grid[i, j] = _p00_from_channels(g, sp0, sm0, dz0, beta, omega_q)

    return p00_grid


# =============================================================================
# 2. LOAD DATA AND BUILD MAPS
# =============================================================================

print("Loading pi4_ncut_convergence.csv …")
df = pd.read_csv(DATA)
df_ed = df[(df.n_modes == N_MODES_ED) & (df.n_cut == N_CUT_MAX)].copy()
print(f"  ED reference: {len(df_ed)} rows  (n_modes={N_MODES_ED}, n_cut={N_CUT_MAX})")

g_vals    = sorted(df_ed.g.unique())
beta_vals = sorted(df_ed.beta.unique())
# Convert to dimensionless βω_q for all axes and overlays
beta_plot = [b * OMEGA_Q for b in beta_vals]

# 2D meshgrid in (g, βω_q) coordinates
G, B = np.meshgrid(g_vals, beta_plot)

# ED population heatmap
ed_pivot  = df_ed.pivot(index="beta", columns="g", values="ed_p00")
ed_map    = ed_pivot.values                          # shape (n_beta, n_g)

# ---------- CONTINUUM chi₀(β) and g★(β) ─────────────────────────────────────
# The manuscript χ-crossover theory refers to the continuum Ohmic bath.
# Compute χ₀(β) using a 800-mode dense approximation to the continuum.
print("Computing continuum chi_0(beta) for physical g*(beta) overlays ...")
omegas_cont, g2k_cont = _make_mode_set(800, 0.02, 20.0, Q_STRENGTH, TAU_C)
chi0_cont_arr = np.zeros(len(beta_vals))
for i, beta in enumerate(beta_vals):
    k00c, k0pc, k0mc, r0pc, r0mc = _kernel_laplace_res(beta, omegas_cont, g2k_cont,
                                                         OMEGA_Q)
    sp0c, sm0c, dz0c = _channels_from_integrals(k00c, k0pc, k0mc, r0pc, r0mc,
                                                 beta, OMEGA_Q, THETA)
    chi0_cont_arr[i] = float(np.sqrt(max(dz0c**2 + sp0c * sm0c, 0.0)))

# g★(β) = χ₀_cont(β)^{-1/2}  (continuum crossover coupling)
g_star_arr   = np.where(chi0_cont_arr > 0,
                        1.0 / np.sqrt(chi0_cont_arr), np.nan)

# χ_cont(g, β) = g² * χ₀_cont(β) as a 2D map for contour plotting
chi_cont_map = G**2 * chi0_cont_arr[:, np.newaxis]  # shape (n_beta, n_g)

print(f"  g*(beta) range: {np.nanmin(g_star_arr):.3f} - {np.nanmax(g_star_arr):.3f}")
print(f"  chi_cont at (g=0.5, beta=2): "
      f"{float(0.5**2 * chi0_cont_arr[np.argmin(np.abs(np.array(beta_vals)-2))]):.3f}")

print("Computing discrete analytic p_00 (n_modes=2) …")
disc_map = compute_analytic_grid(g_vals, beta_vals, OMEGA_Q, THETA,
                                 OMEGA_MIN, OMEGA_MAX,
                                 N_MODES_ED, Q_STRENGTH, TAU_C)

print("Computing continuous analytic p_00 (n_modes=800, wide window) …")
cont_map = compute_analytic_grid(g_vals, beta_vals, OMEGA_Q, THETA,
                                 0.02, 25.0,
                                 800, Q_STRENGTH, TAU_C)

disc_err = disc_map - ed_map   # signed discrepancy, disc − ED
cont_err = cont_map - ed_map   # signed discrepancy, cont − ED

print(f"  max |disc_err| = {np.abs(disc_err).max():.4f}")
print(f"  max |cont_err| = {np.abs(cont_err).max():.4f}")


# =============================================================================
# 3. PLOT
# =============================================================================

CMAP_POP = "YlOrRd_r"    # for raw populations
CMAP_ERR = "RdBu_r"      # diverging for signed errors

fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
fig.subplots_adjust(hspace=0.42, wspace=0.48)

panel_data = [
    (axes[0, 0], ed_map,   CMAP_POP, r"$p_{00}$ (ED, $n_{\rm cut}=60$)", "(a)"),
    (axes[0, 1], disc_map, CMAP_POP, r"$p_{00}$ (discrete analytic)",     "(b)"),
    (axes[1, 0], disc_err, CMAP_ERR, r"$\Delta p_{00}$ disc $-$ ED",       "(c)"),
    (axes[1, 1], cont_err, CMAP_ERR, r"$\Delta p_{00}$ cont $-$ ED",       "(d)"),
]

for ax, data, cmap, title, tag in panel_data:
    if cmap == CMAP_POP:
        vmin, vmax = 0.0, 1.0
    else:
        amax = max(abs(data.min()), abs(data.max())) + 1e-6
        vmin, vmax = -amax, amax

    im = ax.pcolormesh(G, B, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest", rasterized=True)
    cb = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
    cb.ax.tick_params(labelsize=6)

    # χ = 1 contour (continuum bath)
    cs = ax.contour(G, B, chi_cont_map, levels=[1.0], colors="white",
                    linewidths=1.2, linestyles="--")
    ax.clabel(cs, fmt=r"$\chi{=}1$", fontsize=5.5, inline=True, inline_spacing=2)

    ax.set_xlabel(r"coupling $g / \omega_q$")
    ax.set_ylabel(r"$\beta\omega_q$")
    ax.set_yscale("log")
    ax.set_ylim(beta_plot[0], beta_plot[-1])
    ax.set_xlim(g_vals[0], g_vals[-1])
    ax.set_title(title, fontsize=8, pad=4)
    ax.text(0.04, 0.94, tag, transform=ax.transAxes,
            fontweight="bold", color="white", fontsize=9,
            va="top")

# Shared legend: single χ=1 / g★ boundary line
from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0], [0], color="white", lw=1.2, ls="--",
           label=r"$\chi_{\rm cont}=1$ boundary ($g_\star(\beta)$)"),
]
fig.legend(handles=legend_lines, loc="lower center", ncol=1, fontsize=7,
           framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

for ext in (".pdf", ".png"):
    out = FIGURES / f"hmf_fig4_2d_landscape{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out.name}")

plt.close(fig)
print("Done.")
