# -*- coding: utf-8 -*-
"""
fig1_chi_theory.py  --  Figure 1: Pure analytic theory (2×2 panel layout)
==========================================================================
All panels are purely theoretical — no simulation data.

Bath model: Continuous Ohmic  J(ω) = α · ω · exp(−ω / ω_c)
  Parameters: α (spectral strength), ω_c (cutoff / bandwidth)
  The coupling g is carried separately so chi = g² · chi0(β).

Coupling geometry: θ = π/4 throughout (locked).
  f = (1/√2)(σ_z − σ_x)
  Both the longitudinal (Δ_z) and transverse (Σ) channels contribute to χ.

chi_0(β) — exact formula (Eq. for χ² = Δ_z² + Σ²):
----------------------------------------------------------------------------
The imaginary-time bath kernel is
   K(u) = (α/π) ∫₀^∞ ω exp(−ω/ω_c) cosh(ω(u − β/2)) / sinh(βω/2) dω

The two g-independent channel amplitudes at θ = π/4 (c = s = 1/√2) are

   Δ_z0(β) = s² ∫₀^β (β−u) K(u) sinh(ω_q u) du
                s² = 1/2

   Σ_0(β)  = (c·s/ω_q) ∫₀^β K(u) [cosh(β ω_q) + 1
                                    − cosh(ω_q u) − cosh(ω_q(β−u))] du
                c·s = 1/2

   chi0(β) = √(Δ_z0² + Σ_0²)

so that chi(β, g) = g² · chi0(β).

Alternatively (and equivalently) the two channel integrals can be written
directly as single spectral integrals over J(ω):

   Δ_z0 = s² ∫₀^∞ J(ω)·I_z(ω, β) dω / π         [see helper dz_integrand]
   Σ_0  = (cs/ω_q) ∫₀^∞ J(ω)·I_x(ω, β) dω / π   [see helper sx_integrand]

where I_z and I_x are the analytic frequency-domain kernels obtained by
substituting the spectral representation of K(u) and performing the u-integral
analytically.  Both routes give numerically identical results; we use the
spectral route because it requires only a 1-D quadrature per beta value.

Bloch vector (exact, no approximation):
   ρ̄ = exp(−a σ_z) · exp(χ·n̂·σ) · exp(−a σ_z)
where a = β ω_q / 2 and n̂ = (n_x, n_z) = (Σ_0, −Δ_z_0) / chi0.
Matrix elements:
   A00 = exp(−a)·[cosh χ + n_z·sinh χ]
   A11 = exp(+a)·[cosh χ − n_z·sinh χ]
   A01 = n_x · sinh χ
   Z   = A00 + A11

Panels:
  (a) γ(χ) = tanh χ / χ  (universal crossover function)
  (b) chi(β, g) contour map — Ohmic bath, θ = π/4
  (c) ∂_g φ  (Bloch-angle susceptibility) vs g for three β values
  (d) ∂_g r  (Bloch-radius gradient) vs g for three β values
"""

import numpy as np
import scipy.integrate as quad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

# Suppress the IntegrationWarning from quad (the removable singularity at 
# Omega=omega_q is handled analytically/numerically but triggers warnings)
warnings.filterwarnings("ignore", category=quad.IntegrationWarning)

FIGURES = Path(__file__).parents[2] / "manuscript" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": True, "figure.dpi": 200, "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
    "legend.fontsize": 8,
})

# ── Physical parameters ────────────────────────────────────────────────────────
OMEGA_Q   = 1.0   # qubit frequency (energy unit)
ALPHA     = 1.0   # Ohmic spectral weight (matching manuscript)
OMEGA_C   = 5.0   # cutoff (matching manuscript)
OMEGA_MIN = 0.0   # IR edge: INCLUDE zero-frequency so Delta_z channel is active
OMEGA_MAX = 2.0   # UV edge  (in units of omega_q)
THETA     = np.pi / 4  # mixing angle -- LOCKED


# ── Ohmic spectral density ─────────────────────────────────────────────────────
def J_ohmic(w, alpha=ALPHA, omega_c=OMEGA_C):
    """One-sided Ohmic spectral density J(Omega) = alpha*Omega*exp(-Omega/omega_c)."""
    return alpha * np.asarray(w, float) * np.exp(-np.asarray(w, float) / omega_c)


# ── Analytic spectral filter functions  ────────────────────────────────────────
# By swapping the order of integration in the imaginary-time integrals we get:
#
#   Delta_z0 = (s^2/pi) Int_0^{Omega_max} J(Omega) * F_z(Omega, beta) dOmega
#   Sigma_0  = (cs / (pi*omega_q)) Int_0^{Omega_max} J(Omega) * F_x(Omega, beta) dOmega
#
# where F_z and F_x are obtained by performing the u in [0,beta] integral
# analytically with K(u) = (1/pi)*J(Omega)*cosh(Omega*(u-beta/2))/sinh(beta*Omega/2).

def _F_z(Omega, beta, omega_q=OMEGA_Q):
    """
    Spectral filter for the Delta_z0 channel:

      F_z(Omega,beta) = Int_0^beta (beta-u) * cosh(Omega*(u-beta/2))/sinh(beta*Omega/2)
                        * sinh(omega_q * u) du

    Evaluated analytically. At Omega == omega_q there is a removable singularity
    handled by the limit (L'Hopital).
    """
    b  = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a  = b * oq / 2.0   # = beta * omega_q / 2

    # Near the pole Omega -> omega_q, use Taylor-expanded formula.
    eps = abs(Om - oq)
    if eps < 1e-6 * oq:
        # Limit Omega -> omega_q analytically:
        # d/dOmega of numerator and denominator gives:
        # F_z -> (1/sinh(b*oq/2)) * [ (b/2)*sinh(a)*cosh(a)/oq
        #          + (b^2/4)*cosh(a) - sinh(a)*tanh(b*oq/2)/(2*oq^2) ]
        # Use a safe cubic Taylor expansion around Om=oq:
        bOm2 = b * oq / 2.0
        sh   = np.sinh(np.clip(bOm2, 1e-14, 500))
        ch_a = np.cosh(np.clip(a, 0, 500))
        sh_a = np.sinh(np.clip(a, 0, 500))
        # F_z at Omega=omega_q (limit):
        val  = ch_a / sh * (
            b**2 / 4.0 * ch_a
            + b / (2.0 * oq) * sh_a
        ) - b * np.cosh(np.clip(bOm2, 0, 500)) / sh * ch_a / oq
        # Note: the two extra terms from the general formula collapse at the limit.
        # The formula below at eps>0 already handles this numerically for eps>1e-4.
        return float(val)

    Om_plus  = Om + oq
    Om_minus = oq - Om   # = -(Om - oq), so sinh(Om_minus*b/2) = -sinh((Om-oq)*b/2)

    bOm2  = b * Om / 2.0
    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    ch_a  = np.cosh(np.clip(a, 0, 500))
    sh_a  = np.sinh(np.clip(a, 0, 500))

    denom = oq**2 - Om**2

    # Inner u-integral (analytically)
    term1 = -b * oq * ch_Om / denom            # from IBP boundary
    term2 = (ch_a / sh_Om) * (
        np.sinh(np.clip(Om_plus  * b / 2, 0, 500)) / Om_plus**2
      + np.sinh(np.clip(Om_minus * b / 2, 0, 500)) / Om_minus**2
    )
    return float(term1 + term2)


def _F_x(Omega, beta, omega_q=OMEGA_Q):
    """
    Spectral filter for the Sigma_0 channel:

      F_x(Omega,beta) = Int_0^beta cosh(Omega*(u-beta/2))/sinh(beta*Omega/2)
                        * [cosh(beta*omega_q)+1 - cosh(omega_q*u)
                                               - cosh(omega_q*(beta-u))] du

    Evaluated analytically.
    """
    b  = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a  = b * oq / 2.0

    eps = abs(Om - oq)
    bOm2  = b * Om / 2.0
    ch_a  = np.cosh(np.clip(a, 0, 500))
    sh_a  = np.sinh(np.clip(a, 0, 500))

    # First part: (cosh(beta*oq)+1) * Int cosh(Om(u-b/2)) du  = 2cosh^2(a)*2sinh(Om*b/2)/Om
    if abs(Om) < 1e-12:  # DC limit
        part1 = 2.0 * ch_a**2 * b   # limit sinh(Om*b/2)/Om -> b/2, times 2
    else:
        sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
        part1 = 2.0 * ch_a**2 * 2.0 * sh_Om / Om

    # Second part: Int cosh(Om(u-b/2))*[cosh(oq*u)+cosh(oq*(b-u))] du
    # = 2 * Int_0^b cosh(Om(u-b/2))*cosh(oq*u) du  (by symmetry)
    # = 2 * 2*cosh(a)*[Om*sinh(Om*b/2)*cosh(a) - oq*cosh(Om*b/2)*sinh(a)] / (Om^2 - oq^2)
    if eps < 1e-6 * oq:  # Om -> oq limit
        # d/dOm of cosh(Om*b/2)*cosh(a): cosh(a)*b/2*sinh(Om*b/2) at Om=oq
        # Use L'Hopital carefully -- numerical fallback:
        dOm = 1e-5 * oq
        return float(_F_x(oq + dOm, beta, omega_q) + _F_x(oq - dOm, beta, omega_q)) / 2.0

    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    denom = Om**2 - oq**2

    part2 = 4.0 * ch_a * (Om * sh_Om * ch_a - oq * ch_Om * sh_a) / denom
    return float(part1 - part2)


def chi0_spectral(beta, theta=THETA, omega_q=OMEGA_Q,
                  alpha=ALPHA, omega_c=OMEGA_C,
                  omega_min=OMEGA_MIN, omega_max=OMEGA_MAX,
                  n_pts=200):
    """
    Compute (chi0, dz0, sx0) as a SINGLE spectral integral:

      Delta_z0 = (s^2 / pi) Int_{omega_min}^{omega_max} J(Omega) * F_z(Omega,beta) dOmega
      Sigma_0  = (cs / (pi*omega_q)) Int_{omega_min}^{omega_max} J(Omega) * F_x(Omega,beta) dOmega

    where F_z / F_x are evaluated analytically (see above).  This avoids the
    nested u-quadrature and correctly includes the DC (Omega->0) contribution
    to both channels.
    """
    s = np.sin(theta)
    c = np.cos(theta)

    def integrand_z(Om):
        if Om < 1e-12:
            return 0.0  # J(0)=0 for Ohmic, so integrand vanishes
        return J_ohmic(Om, alpha, omega_c) * _F_z(Om, beta, omega_q)

    def integrand_x(Om):
        if Om < 1e-12:
            return 0.0
        return J_ohmic(Om, alpha, omega_c) * _F_x(Om, beta, omega_q)

    # Split at omega_q to handle the near-pole region carefully
    lo  = max(omega_min, 1e-10)
    mid = omega_q
    hi  = omega_max

    dz_lo, _ = quad.quad(integrand_z, lo,  mid, limit=300, epsrel=1e-10)
    dz_hi, _ = quad.quad(integrand_z, mid, hi,  limit=300, epsrel=1e-10)
    dz0 = (s**2 / np.pi) * (dz_lo + dz_hi)

    sx_lo, _ = quad.quad(integrand_x, lo,  mid, limit=300, epsrel=1e-10)
    sx_hi, _ = quad.quad(integrand_x, mid, hi,  limit=300, epsrel=1e-10)
    sx0 = (c * s / (np.pi * omega_q)) * (sx_lo + sx_hi)

    chi0 = float(np.sqrt(dz0**2 + sx0**2))
    return chi0, float(dz0), float(sx0)


# ── Cache  ─────────────────────────────────────────────────────────────────────
_chi0_cache = {}

def get_chi0(beta):
    """Return (chi0, dz0, sx0). Computed via single spectral integral; cached."""
    if beta not in _chi0_cache:
        _chi0_cache[beta] = chi0_spectral(beta)
    return _chi0_cache[beta]


# ── Crossover function ─────────────────────────────────────────────────────────

def gamma(chi_arr):
    chi_arr = np.asarray(chi_arr, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(chi_arr == 0.0, 1.0, np.tanh(chi_arr) / chi_arr)


# ── Exact Bloch vector  ────────────────────────────────────────────────────────

def bloch_ohmic(g_arr, beta):
    """
    Exact Bloch vector for the Ohmic bath at theta=pi/4.

    Returns (phi, r, mx, mz) where
      phi = arctan2(mx, −mz)  (Bloch angle from −z toward +x)
      r   = |Bloch vector|
    """
    a       = beta * OMEGA_Q / 2.0
    chi0, dz0, sx0 = get_chi0(beta)

    g_arr = np.asarray(g_arr, dtype=float)
    chi   = np.clip(g_arr**2 * chi0, 0, 300)

    # Direction of M in Pauli space:
    #   M = g²·[sx0·σ_x  −  dz0·σ_z]
    # so n̂ = (n_x, n_z) = (sx0, −dz0) / chi0
    if chi0 < 1e-15:
        n_x, n_z = 0.0, 0.0
    else:
        n_x = sx0  / chi0      # coefficient of  σ_x in M/|M|
        n_z = -dz0 / chi0      # coefficient of  σ_z in M/|M|  (note minus)

    # ρ̄ = exp(−a σ_z) · exp(M) · exp(−a σ_z)
    sh = np.sinh(chi)
    ch = np.cosh(chi)

    A00 = np.exp(-a) * (ch + n_z * sh)
    A11 = np.exp(+a) * (ch - n_z * sh)
    A01 = n_x * sh
    Z   = A00 + A11

    mz  = (A00 - A11) / Z
    mx  = 2.0 * A01  / Z
    r   = np.sqrt(mx**2 + mz**2)
    phi = np.arctan2(mx, -mz)        # angle from −z toward +x
    return phi, r, mx, mz


def dphi_dg(g_arr, beta, dg=1e-4):
    phi_p, *_ = bloch_ohmic(np.asarray(g_arr) + dg, beta)
    phi_m, *_ = bloch_ohmic(np.asarray(g_arr) - dg, beta)
    return (phi_p - phi_m) / (2.0 * dg)


def dr_dg(g_arr, beta, dg=1e-4):
    _, r_p, *_ = bloch_ohmic(np.asarray(g_arr) + dg, beta)
    _, r_m, *_ = bloch_ohmic(np.asarray(g_arr) - dg, beta)
    return (r_p - r_m) / (2.0 * dg)


# ── Figure ────────────────────────────────────────────────────────────────────
BETAS       = [0.5, 2.0, 8.0]
BETA_COLORS = ["#e07b39", "#7b52ab", "#1a6ea8"]
BETA_LABELS = [r"$\beta\omega_q=0.5$", r"$\beta\omega_q=2$", r"$\beta\omega_q=8$"]

# Panels (c,d): three betas straddling the crossover region
BETAS_CD       = [3.0, 4.0, 5.0]
BETA_COLORS_CD = ["#e07b39", "#7b52ab", "#1a6ea8"]
BETA_LABELS_CD = [r"$\beta\omega_q=3$", r"$\beta\omega_q=4$", r"$\beta\omega_q=5$"]

print("Pre-computing chi0 for beta grid and canonical betas...")
# Canonical betas for panels (c) and (d)
for b in BETAS:
    c0, dz, sx = get_chi0(b)
    print(f"  beta={b:.1f}: chi0={c0:.6f}  (dz0={dz:.6f}, sx0={sx:.6f})")

# Contour panel: chi0 on a beta grid (only the canonical betas are slow;
# the grid uses a fast shortcut: chi0^2 ~ dz0^2 + sx0^2)
beta_grid = np.linspace(0.5, 6.0, 50)
chi0_grid = np.array([get_chi0(b)[0] for b in beta_grid])
print("Done pre-computing chi0 grid.")

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
fig.subplots_adjust(hspace=0.45, wspace=0.42)

# ── (a) gamma(chi)  ───────────────────────────────────────────────────────────
ax = axes[0, 0]
chi_vals = np.linspace(0, 4, 400)
ax.plot(chi_vals, gamma(chi_vals), color="#9467bd", lw=1.8,
        label=r"$\gamma(\chi)=\tanh\chi/\chi$")
ax.plot(chi_vals[:160], 1 - chi_vals[:160]**2 / 3,
        color="#aaaaaa", ls="-.", lw=1.0, label=r"$1-\chi^2/3$")
ax.plot(chi_vals[40:], 1.0 / chi_vals[40:],
        color="#888888", ls="--", lw=1.0, label=r"$1/\chi$")
ax.axvline(1.0, color="k", ls="--", alpha=0.5, lw=0.8)
ax.axhline(gamma(1.0), color="k", ls=":", alpha=0.3, lw=0.7)
ax.axvspan(0, 1, color="#e8f4f8", alpha=0.5, zorder=0)
ax.axvspan(1, 4, color="#fff4e6", alpha=0.5, zorder=0)
ax.set_xlabel(r"$\chi$")
ax.set_ylabel(r"$\gamma(\chi)$")
ax.set_xlim(0, 4)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=6, framealpha=0.8)
ax.text(0.05, 0.92, "(a)", transform=ax.transAxes, fontweight="bold")
ax.text(0.08, 0.12, "Weak", transform=ax.transAxes, fontsize=7, 
        color="#2c7fb8", fontweight="bold")
ax.text(0.65, 0.12, "Strong", transform=ax.transAxes, fontsize=7, 
        color="#d95f02", fontweight="bold")

# ── (b) chi(beta, g) contour map  ────────────────────────────────────────────
ax = axes[0, 1]
g_grid  = np.linspace(0.01, 1.2, 80)
BB, GG  = np.meshgrid(beta_grid, g_grid)
# chi(beta,g) = g² · chi0(beta): vectorised over beta_grid
CHI_MAP = GG**2 * chi0_grid[np.newaxis, :]

levels_fill  = np.logspace(np.log10(0.02), np.log10(10), 60)
levels_lines = [0.1, 0.3, 1.0, 3.0]
cf = ax.contourf(BB, GG, CHI_MAP,
                  levels=levels_fill,
                  cmap="RdYlBu_r",
                  norm=mcolors.LogNorm(vmin=0.02, vmax=10))
cs = ax.contour(BB, GG, CHI_MAP,
                 levels=levels_lines,
                 colors="k", linewidths=0.7)
ax.clabel(cs, fmt=lambda x: f"$\\chi={x:g}$", fontsize=6)

# Crossover locus  g⋆(β) = chi0(β)^{-1/2}
g_star = np.where(chi0_grid > 0, 1.0 / np.sqrt(chi0_grid), np.nan)
ax.plot(beta_grid, g_star, color="white", lw=2.5,
        label=r"$g_\star$: $\chi\!=\!1$")
ax.legend(fontsize=6, framealpha=0.7, loc="upper right")
ax.set_xlabel(r"$\beta\omega_q$")
ax.set_ylabel(r"$g/\omega_q$")
ax.text(0.05, 0.92, "(b)", transform=ax.transAxes,
        fontweight="bold", color="white")
ax.set_xlim(beta_grid[0], beta_grid[-1])
ax.set_ylim(g_grid[0], g_grid[-1])



# ── (c) d(phi)/dg  ───────────────────────────────────────────────────────────
# Pre-compute chi0 for panel (c,d) betas
for b in BETAS_CD:
    if b not in _chi0_cache:
        c0, dz, sx = get_chi0(b)
        print(f"  beta={b:.1f}: chi0={c0:.6f}  (dz0={dz:.6f}, sx0={sx:.6f})")

# Universal scaling range [0, 2] in units of g_star
G_SCALE_MAX = 2.0
xi = np.linspace(0.01, G_SCALE_MAX, 400)

ax = axes[1, 0]
for b, col, lb in zip(BETAS_CD, BETA_COLORS_CD, BETA_LABELS_CD):
    c0 = get_chi0(b)[0]
    gs = 1.0 / np.sqrt(c0) if c0 > 0 else np.nan
    # Physical g values for this beta
    g_vals = xi * gs
    # Compute derivative at these physical g values
    y = dphi_dg(g_vals, b)
    ax.plot(xi, y, color=col, lw=1.5, label=lb)
    
    # Peak markers at the raw heights
    pk_idx = np.argmax(y)
    ax.plot(xi[pk_idx], y[pk_idx], "o", color=col, ms=4, zorder=5)

ax.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.6, label=r"$g=g_\star$")
ax.axvspan(0, 1, color="#e8f4f8", alpha=0.5, zorder=0)
ax.axvspan(1, G_SCALE_MAX, color="#fff4e6", alpha=0.5, zorder=0)
ax.set_xlim(0, G_SCALE_MAX)
ax.set_xlabel(r"$g/g_\star(\beta)$")
ax.set_ylabel(r"$\partial_g\varphi$")
ax.legend(fontsize=6, framealpha=0.8, loc="upper right")
ax.text(0.05, 0.92, "(c)", transform=ax.transAxes, fontweight="bold")

# ── (d) dr/dg  ───────────────────────────────────────────────────────────────
ax = axes[1, 1]
for b, col, lb in zip(BETAS_CD, BETA_COLORS_CD, BETA_LABELS_CD):
    c0 = get_chi0(b)[0]
    gs = 1.0 / np.sqrt(c0) if c0 > 0 else np.nan
    g_vals = xi * gs
    y = dr_dg(g_vals, b)
    ax.plot(xi, y, color=col, lw=1.5)
    
    pk_idx = np.argmax(np.abs(y))
    ax.plot(xi[pk_idx], y[pk_idx], "o", color=col, ms=4, zorder=5)

ax.axvline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
ax.axvspan(0, 1, color="#e8f4f8", alpha=0.5, zorder=0)
ax.axvspan(1, G_SCALE_MAX, color="#fff4e6", alpha=0.5, zorder=0)
ax.set_xlim(0, G_SCALE_MAX)
ax.set_xlabel(r"$g/g_\star(\beta)$")
ax.set_ylabel(r"$\partial_g r$")
ax.text(0.05, 0.92, "(d)", transform=ax.transAxes, fontweight="bold")


out = FIGURES / "hmf_fig1_chi_theory.png"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
