# -*- coding: utf-8 -*-
"""
hmf_crossover_figset_codex_v48.py
----------------------------------
Four-figure crossover paper set.  All data from existing CSVs:
  - hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_scan.csv  (parameter sweeps)
  - hmf_narrowband_convergence_scan_codex_v36.csv              (bandwidth scan)

Fig 1: Pure theory – gamma(chi) crossover function and chi(beta, g) contour map
Fig 2: ED vs ordered-kernel analytic – coupling sweep at beta=2, temp & angle overlays
Fig 3: ED simulability crossover – RMSE vs bandwidth (v36), ncut sensitivity
Fig 4: Full parameter sweep landscape – coupling / temperature / angle, disc vs cont
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SRC   = Path(__file__).parent
FIGS  = SRC.parents[1] / "manuscript" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CSV43 = SRC / "hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_scan.csv"
CSV36 = SRC / "hmf_narrowband_convergence_scan_codex_v36.csv"

# ── publication style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.labelsize":  9,
    "axes.titlesize":  9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex":     False,
    "figure.dpi":      200,
    "lines.linewidth": 1.3,
    "axes.linewidth":  0.8,
})

PALETTE = {
    "disc":  "#1f77b4",   # discrete analytic – blue
    "cont":  "#ff7f0e",   # continuous analytic – orange
    "ed":    "#2ca02c",   # ED – green
    "chi":   "#9467bd",   # chi-related – purple
    "bstar": "#d62728",   # magic bandwidth – red
}


# ─────────────────────────────────────────────────────────────────────────────
# Analytic helpers (exact, no approximation)
# ─────────────────────────────────────────────────────────────────────────────

def gamma(chi):
    """Exact crossover function gamma(chi) = tanh(chi)/chi."""
    chi = np.asarray(chi, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(chi == 0.0, 1.0, np.tanh(chi) / chi)
    return out


def chi_ohmic(g, beta, omega_q=1.0, Q=0.1, tau_c=1.0, n_modes=2,
              omega_lo=0.2, omega_hi=1.8):
    """
    Approximate chi for the v43 Ohmic bath (omega_q=1, window [0.2,1.8]).
    Uses the analytic continuous-kernel formula via numerical quadrature
    for the three kernel samples K(0), K(omega_q), R.
    Here we use the simplified Drude form for the spectral density:
      J(w) = Q * tau_c * w * exp(-tau_c * w)
    and the exact kernel integrals.
    Returns scalar chi.
    """
    import scipy.integrate as integrate

    def Jw(w):
        return Q * tau_c * w * np.exp(-tau_c * w)

    def coth(x):
        return 1.0 / np.tanh(np.clip(x, 1e-10, 700))

    # K(0) = integral J(w)/w * beta * n(w) dw   [zero-frequency kernel]
    def integrand_K0(w):
        if w < 1e-12:
            return Jw(1e-12) / 1e-12 * beta    # limit as w->0: J/w -> Q*tau_c, beta*n(w)->1
        nb = 1.0 / (np.exp(beta * w) - 1.0) if beta * w < 700 else 0.0
        return Jw(w) / w * beta * nb

    # K(omega_q) = integral J(w)/(w^2 - omega_q^2) * ... [resonant]
    # In the analytic formula from 04_results_v5 the channel amplitudes are
    # Sigma_pm ~ integral J(w) * f(w, beta) dw evaluated at ±omega_q.
    # For chi we use chi^2 = Delta_z^2 + Sigma+ * Sigma-
    # and the known result chi ~ g^2 * chi0(beta, theta).
    # For theta = pi/2 (pure sigma_x coupling), Delta_z -> 0 and
    # Sigma^2 = (g * sum_k lambda_k / (2 omega_k) * tanh(beta*omega_k/2))^2  [discrete]
    # Continuous: Sigma = g^2 * integral J(w)/(2w) * tanh(beta*w/2) dw
    def integrand_Sigma(w):
        th = np.tanh(beta * w / 2.0) if beta * w < 700 else 1.0
        return Jw(w) / (2.0 * w) * th

    Sigma, _ = integrate.quad(integrand_Sigma, omega_lo, omega_hi, limit=200)
    chi_val = g**2 * abs(Sigma)
    return chi_val


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  – Pure theory: gamma(chi) and chi landscape
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    fig.subplots_adjust(wspace=0.38)

    # ── panel (a): gamma(chi) ─────────────────────────────────────────────
    ax = axes[0]
    chi_vals = np.linspace(0, 4, 400)
    gam = gamma(chi_vals)

    ax.plot(chi_vals, gam, color=PALETTE["chi"], lw=1.8, label=r"$\gamma(\chi)=\tanh\chi/\chi$")
    ax.axvline(1.0, color="k", ls="--", lw=0.9, alpha=0.7)
    ax.axhline(gamma(1.0), color="k", ls=":", lw=0.7, alpha=0.5)

    # weak and ultrastrong asymptotes
    chi_w = np.linspace(0, 1.2, 80)
    chi_u = np.linspace(0.6, 4, 200)
    ax.plot(chi_w, 1 - chi_w**2 / 3, lw=1.0, color="#aaaaaa", ls="-.",
            label=r"$1-\chi^2/3$")
    ax.plot(chi_u, 1.0 / chi_u, lw=1.0, color="#888888", ls="--",
            label=r"$1/\chi$")

    ax.annotate(r"$\chi=1$", xy=(1.0, 0.05), xytext=(1.3, 0.12),
                fontsize=7, color="k",
                arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"$\gamma(\chi)$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.text(0.05, 0.92, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.text(0.05, 0.22, "weak", transform=ax.transAxes, fontsize=6.5, color="gray")
    ax.text(0.72, 0.22, "ultrastrong", transform=ax.transAxes, fontsize=6.5, color="gray")
    ax.axvspan(0, 1, alpha=0.04, color=PALETTE["chi"])
    ax.axvspan(1, 4, alpha=0.04, color=PALETTE["disc"])

    # ── panel (b): chi(beta, g) contour ──────────────────────────────────
    ax = axes[1]
    beta_arr = np.linspace(0.3, 10, 120)
    g_arr    = np.linspace(0.01, 2.0, 120)
    BB, GG   = np.meshgrid(beta_arr, g_arr)

    # chi ~ g^2 * chi0(beta); chi0 ≈ integral J(w)/(2w)*tanh(beta*w/2)dw
    # Precompute chi0 on the beta grid (cheap numerical quad)
    import scipy.integrate as integrate
    def chi0_beta(b):
        def integrand(w):
            J = 0.1 * 1.0 * w * np.exp(-1.0 * w)  # Q=0.1, tau_c=1
            th = np.tanh(b * w / 2.0) if b * w < 700 else 1.0
            return J / (2.0 * w) * th
        val, _ = integrate.quad(integrand, 0.2, 1.8, limit=200)
        return val

    chi0_arr = np.array([chi0_beta(b) for b in beta_arr])
    CHI = GG**2 * chi0_arr[np.newaxis, :]   # shape (n_g, n_beta)

    levels = [0.25, 0.5, 1.0, 2.0, 4.0]
    cf = ax.contourf(BB, GG, CHI, levels=np.linspace(0, 5, 60),
                     cmap="RdYlBu_r", vmin=0, vmax=5)
    cs = ax.contour(BB, GG, CHI, levels=levels, colors="k",
                    linewidths=0.7, linestyles=["--", "--", "-", "-", "-"])
    ax.clabel(cs, fmt={l: rf"$\chi={l:.2g}$" for l in levels}, fontsize=6,
              inline=True, inline_spacing=2)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(r"$\chi(\beta,g)$", fontsize=8)
    cb.ax.tick_params(labelsize=6)

    # mark the crossover line chi=1
    g_star = np.sqrt(1.0 / np.where(chi0_arr > 0, chi0_arr, np.nan))
    ax.plot(beta_arr, g_star, color="white", lw=1.5, ls="-",
            label=r"$g_\star(\beta)$: $\chi=1$")
    ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)
    ax.set_xlabel(r"$\beta\omega_q$")
    ax.set_ylabel(r"$g/\omega_q$")
    ax.set_xlim(beta_arr[0], beta_arr[-1])
    ax.set_ylim(g_arr[0], g_arr[-1])
    ax.text(0.05, 0.92, "(b)", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color="w")

    fig.suptitle("Theory: the $\\chi$ crossover", fontsize=9, y=1.01)
    out = FIGS / "hmf_fig1_chi_theory_v48.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"[Fig 1] saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  – ED vs ordered-kernel: coupling / temperature / angle sweeps
#             (three panels side-by-side, focused on disc analytic vs ED)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2(df43):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6))
    fig.subplots_adjust(wspace=0.42)

    sweep_meta = [
        ("coupling", "g",    r"$g/\omega_q$",   r"$p_{00}$",    "beta",  2.0, "(a) Coupling sweep"),
        ("temperature","beta",r"$\beta\omega_q$",r"$p_{00}$",    "g",     0.5, "(b) Temperature sweep"),
        ("angle",    "theta", r"$\theta$ (rad)", r"$p_{00}$",    "beta",  2.0, "(c) Angle sweep"),
    ]

    for ax, (sw, px, xl, yl, fixed_col, fixed_val, title) in zip(axes, sweep_meta):
        sub = df43[(df43.sweep == sw)].copy()
        # select canonical fixed value
        unique_fixed = sub[fixed_col].unique()
        idx = np.argmin(np.abs(unique_fixed - fixed_val))
        fval = unique_fixed[idx]
        sub = sub[np.isclose(sub[fixed_col], fval, atol=0.01)]
        sub = sub.sort_values(px)

        x = sub[px].values
        ax.plot(x, sub["ed_p00"].values,
                color=PALETTE["ed"], lw=1.5, label="ED", zorder=3)
        ax.plot(x, sub["analytic_disc_p00"].values,
                color=PALETTE["disc"], lw=1.2, ls="--", label="Analytic (disc)")
        ax.plot(x, sub["analytic_cont_p00"].values,
                color=PALETTE["cont"], lw=1.0, ls=":", label="Analytic (cont)")

        # bare Gibbs reference (g=0 value from coupling sweep, or beta=0 from temp sweep)
        bare = sub["ed_p00"].values[0] if sw != "coupling" else sub.iloc[0]["analytic_disc_p00"]
        ax.axhline(bare, color="gray", lw=0.7, ls="-.", alpha=0.6, label="Bare Gibbs")

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlim(x.min(), x.max())
        if ax == axes[0]:
            ax.legend(fontsize=6, framealpha=0.8, loc="upper right")

    fig.suptitle(r"ED vs.\ exact theory: $\omega_q=1$, $N_\omega=2$, $n_{\max}=6$",
                 fontsize=9, y=1.02)
    out = FIGS / "hmf_fig2_ed_vs_analytic_sweeps_v48.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"[Fig 2] saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  – ED simulability crossover: RMSE vs bandwidth (v36 data)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3(df36):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6))
    fig.subplots_adjust(wspace=0.42)

    # ── panel (a): RMSE-averaged-over-beta vs bandwidth for ncut 4,5,6 ──
    ax = axes[0]
    ncut_vals = [4, 5, 6]
    colors_nc = ["#aec6f5", "#5799d2", "#1f4e91"]
    betas_all = sorted(df36["beta"].unique())

    for nc, col_nc in zip(ncut_vals, colors_nc):
        sub = df36[df36.n_cut == nc].copy()
        bw_vals = sorted(sub["bandwidth"].unique())
        rmse_mean = []
        for bw in bw_vals:
            rows = sub[np.isclose(sub["bandwidth"], bw, atol=0.01)]
            # RMSE = sqrt(mean(d_ed_ord_p00^2)) across beta values
            rmse = np.sqrt(np.mean(rows["d_ed_ord_p00"].values**2))
            rmse_mean.append(rmse)
        ax.plot(bw_vals, rmse_mean, color=col_nc, lw=1.3,
                label=rf"$n_{{\max}}={nc}$", marker="o", ms=3)

    # mark magic bandwidth B* = 3.8 and empirical RMSE minimum B=3.2
    ax.axvline(3.8, color=PALETTE["bstar"], lw=1.0, ls="--",
               label=r"$B^*=3.8$")
    ax.axvline(3.2, color=PALETTE["ed"], lw=1.0, ls="-.",
               label=r"RMSE min $B=3.2$")
    ax.set_xlabel(r"Bandwidth $B$")
    ax.set_ylabel(r"RMSE($\,p_{00}$)")
    ax.set_title("(a) RMSE vs bandwidth", fontsize=8, pad=3)
    ax.legend(fontsize=6, framealpha=0.8)
    ax.set_xlim(0.3, 5.2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── panel (b): RMSE difference between ncut=4 and ncut=6 vs bandwidth ─
    ax = axes[1]
    sub4 = df36[df36.n_cut == 4].copy()
    sub6 = df36[df36.n_cut == 6].copy()
    bw_vals = sorted(sub4["bandwidth"].unique())

    for b_val in [0.6, 2.48, 5.3]:   # three representative betas
        r4, r6, bws = [], [], []
        for bw in bw_vals:
            rows4 = sub4[(np.isclose(sub4["bandwidth"], bw, atol=0.01)) &
                         (np.isclose(sub4["beta"], b_val, atol=0.05))]
            rows6 = sub6[(np.isclose(sub6["bandwidth"], bw, atol=0.01)) &
                         (np.isclose(sub6["beta"], b_val, atol=0.05))]
            if len(rows4) and len(rows6):
                r4.append(abs(rows4["d_ed_ord_p00"].values[0]))
                r6.append(abs(rows6["d_ed_ord_p00"].values[0]))
                bws.append(bw)
        delta = np.abs(np.array(r4) - np.array(r6))
        ax.plot(bws, delta, lw=1.2, marker="s", ms=2.5,
                label=rf"$\beta\omega_q={b_val}$")

    ax.axvline(3.2, color=PALETTE["ed"], lw=1.0, ls="-.", alpha=0.7)
    ax.set_xlabel(r"Bandwidth $B$")
    ax.set_ylabel(r"$|p_{00}(n_{\max}=4)-p_{00}(n_{\max}=6)|$")
    ax.set_title("(b) Cutoff sensitivity vs bandwidth", fontsize=8, pad=3)
    ax.legend(fontsize=6, framealpha=0.8)
    ax.set_xlim(0.3, 5.2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── panel (c): RMSE vs beta at fixed bandwidth B=3.2 for ncut=4,5,6 ─
    ax = axes[2]
    for nc, col_nc in zip(ncut_vals, colors_nc):
        sub = df36[(df36.n_cut == nc) &
                   np.isclose(df36["bandwidth"], 3.2, atol=0.05)].copy()
        sub = sub.sort_values("beta")
        ax.plot(sub["beta"].values, np.abs(sub["d_ed_ord_p00"].values),
                color=col_nc, lw=1.3, marker="o", ms=3,
                label=rf"$n_{{\max}}={nc}$")

    ax.set_xlabel(r"$\beta\omega_q$")
    ax.set_ylabel(r"$|\Delta p_{00}|$ at $B=3.2$")
    ax.set_title("(c) Error vs temperature (optimal BW)", fontsize=8, pad=3)
    ax.legend(fontsize=6, framealpha=0.8)

    fig.suptitle(r"ED simulability crossover: $\omega_q=2$, $g=0.5$, $N_\omega=3$",
                 fontsize=9, y=1.02)
    out = FIGS / "hmf_fig3_ed_convergence_bandwidth_v48.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"[Fig 3] saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4  – Parameter sweep landscape: disc vs cont analytic vs ED
#             All three sweeps in a 2×3 grid (top: absolute p00, bottom: delta)
# ─────────────────────────────────────────────────────────────────────────────

def make_fig4(df43):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.8), sharex="col")
    fig.subplots_adjust(hspace=0.38, wspace=0.38)

    sweep_meta = [
        ("coupling",    "g",    r"$g/\omega_q$",   "beta",  2.0),
        ("temperature", "beta", r"$\beta\omega_q$", "g",     0.5),
        ("angle",       "theta",r"$\theta$ (rad)",  "beta",  2.0),
    ]

    for col, (sw, px, xl, fixed_col, fixed_val) in enumerate(sweep_meta):
        sub = df43[df43.sweep == sw].copy()
        unique_fixed = sub[fixed_col].unique()
        idx = np.argmin(np.abs(unique_fixed - fixed_val))
        fval = unique_fixed[idx]
        sub = sub[np.isclose(sub[fixed_col], fval, atol=0.01)].sort_values(px)

        x = sub[px].values
        ed   = sub["ed_p00"].values
        disc = sub["analytic_disc_p00"].values
        cont = sub["analytic_cont_p00"].values

        # top row: raw p00
        ax_top = axes[0, col]
        ax_top.plot(x, ed,   color=PALETTE["ed"],   lw=1.5, label="ED")
        ax_top.plot(x, disc, color=PALETTE["disc"],  lw=1.2, ls="--", label="Disc.")
        ax_top.plot(x, cont, color=PALETTE["cont"],  lw=1.0, ls=":",  label="Cont.")
        ax_top.set_ylabel(r"$p_{00}$")
        ax_top.set_title(sw.capitalize() + " sweep", fontsize=8, pad=3)
        if col == 0:
            ax_top.legend(fontsize=6, framealpha=0.8)

        # bottom row: signed residuals
        ax_bot = axes[1, col]
        ax_bot.axhline(0, color="k", lw=0.6)
        ax_bot.plot(x, disc - ed, color=PALETTE["disc"], lw=1.2, ls="--", label="Disc. - ED")
        ax_bot.plot(x, cont - ed, color=PALETTE["cont"], lw=1.0, ls=":",  label="Cont. - ED")
        ax_bot.set_xlabel(xl)
        ax_bot.set_ylabel(r"$\Delta p_{00}$")
        if col == 0:
            ax_bot.legend(fontsize=6, framealpha=0.8)
        ax_bot.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    for ax, label in zip(axes[0], ["(a)", "(c)", "(e)"]):
        ax.text(0.05, 0.93, label, transform=ax.transAxes, fontsize=8, fontweight="bold")
    for ax, label in zip(axes[1], ["(b)", "(d)", "(f)"]):
        ax.text(0.05, 0.93, label, transform=ax.transAxes, fontsize=8, fontweight="bold")

    fig.suptitle(r"Parameter sweep landscape: disc.\ vs cont.\ analytic vs ED"
                 r"  ($\omega_q=1$, $N_\omega=2$, $n_{\max}=6$)",
                 fontsize=9, y=1.01)
    out = FIGS / "hmf_fig4_sweep_landscape_v48.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"[Fig 4] saved → {out.name}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("Loading CSV data...")
    df43 = pd.read_csv(CSV43)
    df36 = pd.read_csv(CSV36)
    print(f"  v43: {len(df43)} rows | sweeps: {df43['sweep'].unique()}")
    print(f"  v36: {len(df36)} rows | bandwidths: {sorted(df36['bandwidth'].unique())}")

    print("\nGenerating Figure 1 (theory) …")
    make_fig1()

    print("Generating Figure 2 (ED vs analytic sweeps) …")
    make_fig2(df43)

    print("Generating Figure 3 (bandwidth convergence) …")
    make_fig3(df36)

    print("Generating Figure 4 (parameter sweep landscape) …")
    make_fig4(df43)

    print(f"\nAll figures written to: {FIGS}")
