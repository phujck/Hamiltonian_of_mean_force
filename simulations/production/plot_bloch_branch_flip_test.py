"""
plot_bloch_branch_flip_test.py
=================================
Standalone branch diagnostic for the exact qubit mean-force state.

This script is intentionally self-contained and does not import the existing
production plotting modules because those execute figure generation on import.

Why this exists
---------------
The manuscript-visible "reflection" in the folded Bloch portrait
(`sqrt(max(m_x,0))` vs `m_z`) can be caused by a mixture of:

1) a projection artifact (the sign of the transverse component is folded away),
2) a chart singularity in an angle parameterization (`atan` / principal branch),
3) or a genuine physical sign change in a chosen component (e.g. `m_z` or
   `m·f`).

The exact state itself is a Cartesian Bloch vector.  It does not "branch" until
we compress it into an angle chart.

Important distinction (explicitly tested here)
----------------------------------------------
- `arctanh` in `Theta = atanh(u) - a` is *not* the periodic branch source on the
  real exact solution.  The branch issue is instead in the inverse-trigonometric
  angle extraction (`atan` / `atan2`) used to describe orientation.
- Different reference axes produce different chart singularities:
  * Bare-axis chart: singular when `m_z = 0` (equiv. `Theta = 0`)
  * Coupling-axis chart: singular when `m·f = 0` (principal angle hits `±pi/2`)

This script therefore tracks several loci simultaneously:
- `Theta = 0`
- `m_z = 0` (cross-check of the same locus)
- `m·f = 0` (the likely branch point for an `f`-relative angle chart)
- `chi = 1` (crossover marker)

Exploratory branch relabeling (`theta -> pi - theta`)
-----------------------------------------------------
Per the current working hypothesis, an angle relabeling by reflection
`theta -> pi - theta` may be the chart operation that moves an apparent
`f_perp`-leaning branch onto the `f`-aligned branch.  This script computes and
plots such a reflected angle as an *exploratory continuity/relabeling test*.
It is a chart transform, not a physical state update.

Outputs
-------
- `manuscript/figures/hmf_branch_flip_test.png`
- `manuscript/figures/hmf_branch_flip_test.pdf`
- `manuscript/figures/hmf_tilt_magnetisation_branch_test.png`
- `manuscript/figures/hmf_tilt_magnetisation_branch_test.pdf`
- `manuscript/figures/hmf_tstar_branch_continuation_demo.png`
- `manuscript/figures/hmf_tstar_branch_continuation_demo.pdf`
- `manuscript/figures/hmf_tilt_branch_sheet_purity_demo.png`
- `manuscript/figures/hmf_tilt_branch_sheet_purity_demo.pdf`
- `manuscript/figures/hmf_full_bloch_theta_branch_demo.png`
- `manuscript/figures/hmf_full_bloch_theta_branch_demo.pdf`
- `simulations/production/out/hmf_branch_flip_crossings.csv`

What it tests
-------------
1) Reproduces the apparent "reflection" in the Fig. 7(b)-style folded
   projection `sqrt(max(m_x,0))` vs `m_z`.
2) Tracks branch-safe continuous angles and principal-branch angles in both the
   bare-axis and coupling-axis charts.
3) Resolves state and reconstructed `h_eff` components in the coupling-adapted
   basis and compares branch/singularity/crossover loci.
4) Tests the exploratory reflected-angle relabeling `theta -> pi - theta`.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.integrate as quad

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


warnings.filterwarnings("ignore", category=quad.IntegrationWarning)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "manuscript" / "figures"
OUT_DIR = ROOT / "simulations" / "production" / "out"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters (match manuscript / Bloch portrait conventions)
# ---------------------------------------------------------------------------

OMEGA_Q = 1.0
ALPHA = 1.0
OMEGA_C = 5.0
OMEGA_MIN = 0.0
OMEGA_MAX = 2.0
THETA_VAL = np.pi / 4.0

BETA_REF = 2.0
G_MULTS_B = [0.1, 0.3, 0.7, 0.9, 1.1, 1.4, 2.0, 4.0]

BETA_MIN = 0.05
BETA_MAX = 25.0
N_BETA = 420

EPS = 1e-14


# ---------------------------------------------------------------------------
# Self-contained spectral helpers (copied/adapted from fig1_chi_theory.py)
# ---------------------------------------------------------------------------

def J_ohmic(w: float | np.ndarray, alpha: float = ALPHA, omega_c: float = OMEGA_C):
    arr = np.asarray(w, dtype=float)
    return alpha * arr * np.exp(-arr / omega_c)


def _F_z(Omega: float, beta: float, omega_q: float = OMEGA_Q) -> float:
    b = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a = b * oq / 2.0

    eps = abs(Om - oq)
    if eps < 1e-6 * max(oq, 1.0):
        bOm2 = b * oq / 2.0
        sh = np.sinh(np.clip(bOm2, 1e-14, 500))
        ch_a = np.cosh(np.clip(a, 0, 500))
        sh_a = np.sinh(np.clip(a, 0, 500))
        val = ch_a / sh * (
            b**2 / 4.0 * ch_a
            + b / (2.0 * oq) * sh_a
        ) - b * np.cosh(np.clip(bOm2, 0, 500)) / sh * ch_a / oq
        return float(val)

    Om_plus = Om + oq
    Om_minus = oq - Om

    bOm2 = b * Om / 2.0
    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    ch_a = np.cosh(np.clip(a, 0, 500))

    denom = oq**2 - Om**2
    term1 = -b * oq * ch_Om / denom
    term2 = (ch_a / sh_Om) * (
        np.sinh(np.clip(Om_plus * b / 2.0, 0, 500)) / Om_plus**2
        + np.sinh(np.clip(Om_minus * b / 2.0, 0, 500)) / Om_minus**2
    )
    return float(term1 + term2)


def _F_x(Omega: float, beta: float, omega_q: float = OMEGA_Q) -> float:
    b = float(beta)
    oq = float(omega_q)
    Om = float(Omega)
    a = b * oq / 2.0

    eps = abs(Om - oq)
    bOm2 = b * Om / 2.0
    ch_a = np.cosh(np.clip(a, 0, 500))
    sh_a = np.sinh(np.clip(a, 0, 500))

    if abs(Om) < 1e-12:
        part1 = 2.0 * ch_a**2 * b
    else:
        sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
        part1 = 4.0 * ch_a**2 * sh_Om / Om

    if eps < 1e-6 * max(oq, 1.0):
        dOm = 1e-5 * max(oq, 1.0)
        return 0.5 * (_F_x(oq + dOm, beta, omega_q) + _F_x(oq - dOm, beta, omega_q))

    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
    ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    denom = Om**2 - oq**2
    part2 = 4.0 * ch_a * (Om * sh_Om * ch_a - oq * ch_Om * sh_a) / denom
    return float(part1 - part2)


def chi0_spectral(beta: float, theta: float = THETA_VAL):
    s = np.sin(theta)
    c = np.cos(theta)

    def integrand_z(Om: float) -> float:
        if Om < 1e-12:
            return 0.0
        return float(J_ohmic(Om) * _F_z(Om, beta))

    def integrand_x(Om: float) -> float:
        if Om < 1e-12:
            return 0.0
        return float(J_ohmic(Om) * _F_x(Om, beta))

    lo = max(OMEGA_MIN, 1e-10)
    mid = OMEGA_Q
    hi = OMEGA_MAX

    dz_lo, _ = quad.quad(integrand_z, lo, mid, limit=300, epsrel=1e-10)
    dz_hi, _ = quad.quad(integrand_z, mid, hi, limit=300, epsrel=1e-10)
    dz0 = (s**2 / np.pi) * (dz_lo + dz_hi)

    sx_lo, _ = quad.quad(integrand_x, lo, mid, limit=300, epsrel=1e-10)
    sx_hi, _ = quad.quad(integrand_x, mid, hi, limit=300, epsrel=1e-10)
    sx0 = (c * s / (np.pi * OMEGA_Q)) * (sx_lo + sx_hi)

    chi0 = float(np.sqrt(max(dz0 * dz0 + sx0 * sx0, 0.0)))
    return float(chi0), float(dz0), float(sx0)


_CH_CACHE: dict[tuple[float, float], tuple[float, float, float]] = {}


def get_channels0(beta: float, theta: float = THETA_VAL):
    key = (float(beta), float(theta))
    if key not in _CH_CACHE:
        _CH_CACHE[key] = chi0_spectral(beta, theta=theta)
    return _CH_CACHE[key]


def gamma_of_chi(chi: float) -> float:
    if abs(chi) < 1e-10:
        chi2 = chi * chi
        return float(1.0 - chi2 / 3.0 + 2.0 * chi2 * chi2 / 15.0)
    return float(np.tanh(chi) / chi)


# ---------------------------------------------------------------------------
# Exact state + branch-resolved diagnostics
# ---------------------------------------------------------------------------

@dataclass
class BranchPoint:
    beta: float
    a: float
    g: float
    g_mult_ref: float
    chi0: float
    chi: float
    gamma: float
    mx: float
    mz: float
    r: float
    u: float
    Theta: float
    kappa: float
    m_perp_signed: float  # manuscript plane gauge component (transverse)
    m_f: float
    m_f_perp: float
    mhat_f: float
    mhat_f_perp: float
    h_perp: float         # h_eff component in (e_perp^(s), n_s) plane-gauge x
    h_z: float            # h_eff component in (e_perp^(s), n_s) plane-gauge z
    h_f: float
    h_f_perp: float
    hhat_f: float
    hhat_f_perp: float
    phi_wrap: float       # bare-axis chart: atan2(m_perp_signed, -m_z)
    phi_cont: float       # bare-axis chart: unwrap(phi_wrap)
    phi_tan_principal: float  # bare-axis principal atan from tan(phi)
    alpha_f_wrap: float       # f-chart: atan2(m·n_perp^(f), m·f)
    alpha_f_cont: float       # f-chart: unwrap(alpha_f_wrap)
    alpha_f_axis: float       # f-chart axis angle modulo pi in [0,pi)
    alpha_f_axis_reflect: float  # exploratory relabeling: pi - alpha_f_axis
    theta_ns_wrap: float         # angle from +n_s chart: atan2(m_perp_signed, m_z)
    theta_ns_axis: float         # modulo-pi axis angle from +n_s chart
    theta_ns_axis_reflect: float # exploratory relabeling: pi - theta_ns_axis


def exact_state_from_channels(g: float, beta: float, chi0: float, dz0: float, sx0: float):
    """
    Returns physical components (mx, mz) using the same direct-matrix convention
    as the existing Bloch portrait production scripts.
    """
    a = 0.5 * beta * OMEGA_Q
    g2 = g * g
    chi = min(g2 * chi0, 300.0)

    if chi0 < EPS:
        n_x = 0.0
        n_z = 0.0
    else:
        n_x = sx0 / chi0
        n_z = -dz0 / chi0

    sh = np.sinh(chi)
    ch = np.cosh(chi)
    A00 = np.exp(-a) * (ch + n_z * sh)
    A11 = np.exp(+a) * (ch - n_z * sh)
    A01 = n_x * sh
    Z = A00 + A11

    mz = float((A00 - A11) / Z)
    mx = float(2.0 * A01 / Z)
    r = float(np.sqrt(max(mx * mx + mz * mz, 0.0)))
    return a, chi, mx, mz, r


def _atanh_clip(x: float, eps: float = 1e-14) -> float:
    return float(np.arctanh(np.clip(x, -1.0 + eps, 1.0 - eps)))


def reconstruct_branch_variables(mx: float, mz: float, a: float):
    """
    Reconstruct manuscript-sign variables (u, Theta, kappa, m_perp_signed)
    from the exact state and a known 'a = beta*omega_q/2'.

    Convention:
      m_perp_signed = -mx
    i.e. the manuscript coupling-plane transverse basis points opposite to the
    physical +sigma_x axis used by the production Bloch portrait.
    """
    t = float(np.tanh(a))
    denom_u = 1.0 + mz * t
    if abs(denom_u) < EPS:
        denom_u = EPS if denom_u >= 0.0 else -EPS
    u = float((mz + t) / denom_u)

    Theta = _atanh_clip(u) - a
    m_perp_signed = float(-mx)

    # kappa from m_perp = kappa * sech(Theta)  =>  kappa = m_perp * cosh(Theta)
    kappa = float(m_perp_signed * np.cosh(np.clip(Theta, -300, 300)))
    return u, Theta, kappa, m_perp_signed


def build_branch_point(
    g: float,
    g_mult_ref: float,
    beta: float,
    phi_cont_value: float | None = None,
    alpha_f_cont_value: float | None = None,
) -> BranchPoint:
    chi0, dz0, sx0 = get_channels0(beta, theta=THETA_VAL)
    a, chi, mx, mz, r = exact_state_from_channels(g, beta, chi0, dz0, sx0)

    # Reconstruct manuscript-sign branch variables from exact state.
    u, Theta, kappa, m_perp_signed = reconstruct_branch_variables(mx, mz, a)
    gamma = gamma_of_chi(chi)

    # Recover the symmetrized-exponent vector h_eff in the manuscript plane
    # basis (e_perp^(s), n_s) from (u, kappa, chi). This avoids convention
    # ambiguity in the direct channel signs used by the production scripts.
    one_minus_u2 = max(1.0 - u * u, EPS)
    if abs(gamma) < EPS:
        h_z = 0.0
        h_perp = 0.0
    else:
        h_z = float(u / gamma)
        h_perp = float(kappa * np.sqrt(one_minus_u2) / gamma)

    c = float(np.cos(THETA_VAL))
    s = float(np.sin(THETA_VAL))
    f_vec = np.array([s, c], dtype=float)
    f_perp_vec = np.array([-c, s], dtype=float)  # n_perp^(f)

    m_plane = np.array([m_perp_signed, mz], dtype=float)
    m_f = float(np.dot(m_plane, f_vec))
    m_f_perp = float(np.dot(m_plane, f_perp_vec))
    m_norm = max(float(np.linalg.norm(m_plane)), EPS)
    mhat_f = m_f / m_norm
    mhat_f_perp = m_f_perp / m_norm

    h_plane = np.array([h_perp, h_z], dtype=float)
    h_f = float(np.dot(h_plane, f_vec))
    h_f_perp = float(np.dot(h_plane, f_perp_vec))
    h_norm = max(float(np.linalg.norm(h_plane)), EPS)
    hhat_f = h_f / h_norm
    hhat_f_perp = h_f_perp / h_norm

    # Bare-axis chart used implicitly by tan(phi) = -kappa/sinh(Theta):
    #   phi_wrap = atan2(m_perp_signed, -m_z)
    # The principal-atan form below is the branch-sensitive chart.
    phi_wrap = float(np.arctan2(m_perp_signed, -mz))
    shTheta = float(np.sinh(np.clip(Theta, -300, 300)))
    if abs(shTheta) < 1e-12:
        phi_tan_principal = float(np.sign(-kappa) * np.pi / 2.0) if abs(kappa) > 0 else 0.0
    else:
        phi_tan_principal = float(np.arctan((-kappa) / shTheta))

    # Coupling-axis chart: branch singularity occurs when m·f = 0, where the
    # principal angle hits +/- pi/2.
    alpha_f_wrap = float(np.arctan2(m_f_perp, m_f))

    # Axis-angle versions modulo pi (director-style chart, not vector chart).
    # These are useful for exploratory relabelings such as theta -> pi-theta.
    alpha_f_axis = float(np.mod(alpha_f_wrap, np.pi))
    alpha_f_axis_reflect = float(np.pi - alpha_f_axis)

    theta_ns_wrap = float(np.arctan2(m_perp_signed, mz))
    theta_ns_axis = float(np.mod(theta_ns_wrap, np.pi))
    theta_ns_axis_reflect = float(np.pi - theta_ns_axis)

    return BranchPoint(
        beta=float(beta),
        a=float(a),
        g=float(g),
        g_mult_ref=float(g_mult_ref),
        chi0=float(chi0),
        chi=float(chi),
        gamma=float(gamma),
        mx=float(mx),
        mz=float(mz),
        r=float(r),
        u=float(u),
        Theta=float(Theta),
        kappa=float(kappa),
        m_perp_signed=float(m_perp_signed),
        m_f=float(m_f),
        m_f_perp=float(m_f_perp),
        mhat_f=float(mhat_f),
        mhat_f_perp=float(mhat_f_perp),
        h_perp=float(h_perp),
        h_z=float(h_z),
        h_f=float(h_f),
        h_f_perp=float(h_f_perp),
        hhat_f=float(hhat_f),
        hhat_f_perp=float(hhat_f_perp),
        phi_wrap=float(phi_wrap),
        phi_cont=float(phi_wrap if phi_cont_value is None else phi_cont_value),
        phi_tan_principal=float(phi_tan_principal),
        alpha_f_wrap=float(alpha_f_wrap),
        alpha_f_cont=float(alpha_f_wrap if alpha_f_cont_value is None else alpha_f_cont_value),
        alpha_f_axis=float(alpha_f_axis),
        alpha_f_axis_reflect=float(alpha_f_axis_reflect),
        theta_ns_wrap=float(theta_ns_wrap),
        theta_ns_axis=float(theta_ns_axis),
        theta_ns_axis_reflect=float(theta_ns_axis_reflect),
    )


def find_sign_changes(arr: np.ndarray) -> list[int]:
    idxs: list[int] = []
    for i in range(len(arr) - 1):
        a = arr[i]
        b = arr[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0 or b == 0.0 or (a < 0.0 < b) or (a > 0.0 > b):
            idxs.append(i)
    return idxs


def interp_root(x0: float, y0: float, x1: float, y1: float) -> float:
    if abs(y1 - y0) < 1e-16:
        return 0.5 * (x0 + x1)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


def reflected_director_continuation_above_beta_star(
    beta: np.ndarray,
    phi_axis: np.ndarray,
    beta_star: float | None,
    phi_ref_axis: float,
) -> np.ndarray:
    """
    Piecewise director-angle continuation used for the appendix visualization.

    Interpretation:
    - Keep the exact axis chart `phi_axis` for T <= T_* (beta >= beta_*).
    - For T > T_* (beta < beta_*), reflect the apparent branch about the
      reference axis `phi_ref_axis` and then lift by integer multiples of pi to
      enforce continuity when approaching the crossover point from the high-T
      side.

    This is an interpretive chart-continuation overlay, not a physical update
    of the exact equilibrium state.
    """
    out = np.array(phi_axis, dtype=float).copy()
    if beta_star is None or not np.isfinite(beta_star):
        return out

    beta = np.asarray(beta, dtype=float)
    phi_axis = np.asarray(phi_axis, dtype=float)
    mask_hiT = beta < float(beta_star)  # T > T_*  <=> beta < beta_*
    idxs = np.where(mask_hiT)[0]
    if idxs.size == 0:
        return out

    # Anchor the continuation to the nearest point on/above the crossover.
    last_hi_idx = int(idxs[-1])
    if last_hi_idx + 1 < len(out):
        prev = float(out[last_hi_idx + 1])
    else:
        prev = float(out[last_hi_idx])

    period = np.pi  # director chart periodicity
    for i in range(last_hi_idx, -1, -1):
        if not mask_hiT[i]:
            continue
        base = float(2.0 * phi_ref_axis - phi_axis[i])  # mirror about phi_ref_axis
        cands = np.array([base + k * period for k in (-2, -1, 0, 1, 2)], dtype=float)
        out[i] = float(cands[np.argmin(np.abs(cands - prev))])
        prev = float(out[i])
    return out


def splice_after_index(a: np.ndarray, b: np.ndarray, idx_split: int) -> np.ndarray:
    """Return array that follows a up to idx_split and b afterwards."""
    out = np.array(a, dtype=float).copy()
    if idx_split + 1 < len(out):
        out[idx_split + 1 :] = np.asarray(b, dtype=float)[idx_split + 1 :]
    return out


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 180,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )

    print("Precomputing reference coupling scale...")
    chi0_ref, _, _ = get_channels0(BETA_REF, theta=THETA_VAL)
    gstar_ref = 1.0 / np.sqrt(chi0_ref) if chi0_ref > 0 else np.inf
    g_vals = [m * gstar_ref for m in G_MULTS_B]
    print(f"  beta_ref={BETA_REF:.3f}  chi0_ref={chi0_ref:.6f}  g*_ref={gstar_ref:.6f}")

    beta_grid = np.geomspace(BETA_MIN, BETA_MAX, N_BETA)
    print(f"Precomputing channel cache on beta grid ({N_BETA} points)...")
    for b in beta_grid:
        get_channels0(float(b), theta=THETA_VAL)

    # Per-coupling traces.
    traces: dict[float, dict[str, np.ndarray]] = {}
    crossings_rows: list[dict[str, float]] = []

    print("Computing exact branch diagnostics...")
    for mult, g in zip(G_MULTS_B, g_vals):
        pts = [build_branch_point(g=float(g), g_mult_ref=float(mult), beta=float(b)) for b in beta_grid]

        # Unwrap both bare-axis and coupling-axis angle charts.
        phi_wrap_arr = np.array([p.phi_wrap for p in pts], dtype=float)
        phi_cont_arr = np.unwrap(phi_wrap_arr)
        alpha_f_wrap_arr = np.array([p.alpha_f_wrap for p in pts], dtype=float)
        alpha_f_cont_arr = np.unwrap(alpha_f_wrap_arr)
        for p, phi_c, alpha_c in zip(pts, phi_cont_arr, alpha_f_cont_arr):
            p.phi_cont = float(phi_c)
            p.alpha_f_cont = float(alpha_c)

        # Collect arrays for plotting.
        arr = {
            "beta": beta_grid.copy(),
            "mx": np.array([p.mx for p in pts], dtype=float),
            "mz": np.array([p.mz for p in pts], dtype=float),
            "r": np.array([p.r for p in pts], dtype=float),
            "m_perp_signed": np.array([p.m_perp_signed for p in pts], dtype=float),
            "Theta": np.array([p.Theta for p in pts], dtype=float),
            "phi_wrap": np.array([p.phi_wrap for p in pts], dtype=float),
            "phi_cont": np.array([p.phi_cont for p in pts], dtype=float),
            "phi_tan_principal": np.array([p.phi_tan_principal for p in pts], dtype=float),
            "alpha_f_wrap": np.array([p.alpha_f_wrap for p in pts], dtype=float),
            "alpha_f_cont": np.array([p.alpha_f_cont for p in pts], dtype=float),
            "alpha_f_axis": np.array([p.alpha_f_axis for p in pts], dtype=float),
            "alpha_f_axis_reflect": np.array([p.alpha_f_axis_reflect for p in pts], dtype=float),
            "theta_ns_wrap": np.array([p.theta_ns_wrap for p in pts], dtype=float),
            "theta_ns_axis": np.array([p.theta_ns_axis for p in pts], dtype=float),
            "theta_ns_axis_reflect": np.array([p.theta_ns_axis_reflect for p in pts], dtype=float),
            "m_f": np.array([p.m_f for p in pts], dtype=float),
            "m_f_perp": np.array([p.m_f_perp for p in pts], dtype=float),
            "mhat_f": np.array([p.mhat_f for p in pts], dtype=float),
            "mhat_f_perp": np.array([p.mhat_f_perp for p in pts], dtype=float),
            "h_f": np.array([p.h_f for p in pts], dtype=float),
            "h_f_perp": np.array([p.h_f_perp for p in pts], dtype=float),
            "hhat_f": np.array([p.hhat_f for p in pts], dtype=float),
            "hhat_f_perp": np.array([p.hhat_f_perp for p in pts], dtype=float),
            "u": np.array([p.u for p in pts], dtype=float),
            "kappa": np.array([p.kappa for p in pts], dtype=float),
            "chi": np.array([p.chi for p in pts], dtype=float),
            "gamma": np.array([p.gamma for p in pts], dtype=float),
        }

        # Detect multiple candidate loci:
        # - Theta=0 (bare-axis chart singularity; equivalent to m_z=0)
        # - m_f=0 (coupling-axis chart singularity; principal angle hits +/-pi/2)
        # - chi=1  (crossover marker)
        event_specs = [
            ("theta0", "Theta", 0.0),
            ("mz0", "mz", 0.0),
            ("mf0", "m_f", 0.0),
            ("chi1", "chi", 1.0),
        ]
        event_betas: dict[str, list[float]] = {name: [] for name, _, _ in event_specs}

        for event_name, key, target in event_specs:
            series = arr[key] - target
            idxs = find_sign_changes(series)
            for i in idxs:
                b0 = float(beta_grid[i])
                b1 = float(beta_grid[i + 1])
                y0 = float(series[i])
                y1 = float(series[i + 1])
                beta_cross = interp_root(b0, y0, b1, y1)
                event_betas[event_name].append(beta_cross)

                p_before = pts[i]
                p_after = pts[i + 1]
                p_cross = build_branch_point(g=float(g), g_mult_ref=float(mult), beta=float(beta_cross))

                phi_cont_cross = float(np.interp(beta_cross, [b0, b1], [arr["phi_cont"][i], arr["phi_cont"][i + 1]]))
                alpha_f_cont_cross = float(
                    np.interp(beta_cross, [b0, b1], [arr["alpha_f_cont"][i], arr["alpha_f_cont"][i + 1]])
                )
                alpha_f_axis_reflect_cross = float(
                    np.interp(
                        beta_cross,
                        [b0, b1],
                        [arr["alpha_f_axis_reflect"][i], arr["alpha_f_axis_reflect"][i + 1]],
                    )
                )

                crossings_rows.append(
                    {
                        "event_type": event_name,
                        "event_series": key,
                        "event_target": float(target),
                        "g": float(g),
                        "g_over_gstar_ref": float(mult),
                        "beta_cross": float(beta_cross),
                        "chi_cross": float(p_cross.chi),
                        "Theta_before": float(p_before.Theta),
                        "Theta_after": float(p_after.Theta),
                        "m_z_before": float(p_before.mz),
                        "m_z_after": float(p_after.mz),
                        "m_perp_signed_before": float(p_before.m_perp_signed),
                        "m_perp_signed_after": float(p_after.m_perp_signed),
                        "m_f_before": float(p_before.m_f),
                        "m_f_after": float(p_after.m_f),
                        "m_f_perp_before": float(p_before.m_f_perp),
                        "m_f_perp_after": float(p_after.m_f_perp),
                        "mhat_f_cross": float(p_cross.mhat_f),
                        "mhat_f_perp_cross": float(p_cross.mhat_f_perp),
                        "heff_f_cross": float(p_cross.h_f),
                        "heff_f_perp_cross": float(p_cross.h_f_perp),
                        "heffhat_f_cross": float(p_cross.hhat_f),
                        "heffhat_f_perp_cross": float(p_cross.hhat_f_perp),
                        "phi_wrap_before": float(arr["phi_wrap"][i]),
                        "phi_wrap_after": float(arr["phi_wrap"][i + 1]),
                        "phi_tan_principal_before": float(arr["phi_tan_principal"][i]),
                        "phi_tan_principal_after": float(arr["phi_tan_principal"][i + 1]),
                        "phi_cont_cross": float(phi_cont_cross),
                        "alpha_f_wrap_before": float(arr["alpha_f_wrap"][i]),
                        "alpha_f_wrap_after": float(arr["alpha_f_wrap"][i + 1]),
                        "alpha_f_cont_cross": float(alpha_f_cont_cross),
                        "alpha_f_axis_cross": float(p_cross.alpha_f_axis),
                        "alpha_f_axis_reflect_cross": float(alpha_f_axis_reflect_cross),
                        "theta_ns_axis_cross": float(p_cross.theta_ns_axis),
                        "theta_ns_axis_reflect_cross": float(p_cross.theta_ns_axis_reflect),
                        "angle_jump_phi_tan_principal": float(
                            arr["phi_tan_principal"][i + 1] - arr["phi_tan_principal"][i]
                        ),
                        "angle_jump_phi_wrap": float(wrap_to_pi(arr["phi_wrap"][i + 1] - arr["phi_wrap"][i])),
                        "angle_jump_phi_cont": float(arr["phi_cont"][i + 1] - arr["phi_cont"][i]),
                        "angle_jump_alpha_f_wrap": float(
                            wrap_to_pi(arr["alpha_f_wrap"][i + 1] - arr["alpha_f_wrap"][i])
                        ),
                        "angle_jump_alpha_f_cont": float(arr["alpha_f_cont"][i + 1] - arr["alpha_f_cont"][i]),
                    }
                )

        arr["event_betas"] = {k: np.array(v, dtype=float) for k, v in event_betas.items()}
        traces[float(g)] = arr

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    csv_path = OUT_DIR / "hmf_branch_flip_crossings.csv"
    if crossings_rows:
        fieldnames = list(crossings_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(crossings_rows)
    else:
        # still emit a header-only file for deterministic downstream behavior
        fieldnames = [
            "event_type",
            "event_series",
            "event_target",
            "g",
            "g_over_gstar_ref",
            "beta_cross",
            "chi_cross",
            "Theta_before",
            "Theta_after",
            "m_z_before",
            "m_z_after",
            "m_f_before",
            "m_f_after",
            "m_f_perp_before",
            "m_f_perp_after",
            "heff_f_cross",
            "heff_f_perp_cross",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print(f"Wrote branch crossing summary -> {csv_path}")
    print(f"  total detected events: {len(crossings_rows)}")
    if crossings_rows:
        counts_by_event: dict[str, int] = {}
        for row in crossings_rows:
            counts_by_event[row["event_type"]] = counts_by_event.get(row["event_type"], 0) + 1
        for key in sorted(counts_by_event):
            print(f"    {key}: {counts_by_event[key]}")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    cmap = mpl.colormaps["viridis"]
    colors_all = [cmap(v) for v in np.linspace(0.0, 0.9, len(G_MULTS_B))]
    color_of_mult = {m: c for m, c in zip(G_MULTS_B, colors_all)}
    g_by_mult = {m: g for m, g in zip(G_MULTS_B, g_vals)}

    # Representative subset for the angle/component panels.
    mults_focus = [1.4, 2.0, 4.0]

    fig = plt.figure(figsize=(8.4, 3.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.1, 1.15], wspace=0.35)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Panel A: Fig. 7(b)-style projection
    X_bound = np.linspace(0.0, 1.05, 500)
    Z_bound = -np.sqrt(np.clip(1.0 - X_bound**4, 0.0, 1.0))
    axA.plot(X_bound, Z_bound, color="black", lw=1.0, zorder=1)
    axA.fill_between(X_bound, Z_bound, 0.1, color="#f7f7f7", zorder=0)
    axA.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.3)
    axA.axvline(0.0, color="k", ls=":", lw=0.5, alpha=0.3)

    c = float(np.cos(THETA_VAL))
    s = float(np.sin(THETA_VAL))
    for alpha_line, label in [(THETA_VAL / 2.0, r"$\theta/2$"), (THETA_VAL / 4.0, r"$\theta/4$")]:
        dX = np.sqrt(np.sin(alpha_line))
        dZ = -np.cos(alpha_line)
        axA.plot([0.0, 2.0 * dX], [0.0, 2.0 * dZ], color="#aaaaaa", ls=":", lw=0.8, alpha=0.6)
        axA.text(dX, dZ, label, fontsize=7, color="#888888", ha="left", va="top")

    for mult, g, color in zip(G_MULTS_B, g_vals, colors_all):
        tr = traces[float(g)]
        xproj = np.sqrt(np.clip(tr["mx"], 0.0, None))
        axA.plot(xproj, tr["mz"], color=color, lw=1.4, alpha=0.95, zorder=2)
        axA.plot(xproj[0], tr["mz"][0], "o", ms=3.8, color=color, mec="white", mew=0.5, zorder=5)
        axA.plot(xproj[-1], tr["mz"][-1], "s", ms=3.5, color=color, mec="white", mew=0.4, zorder=5)

        # mark candidate loci on the folded trajectory:
        #   x = m·f = 0  (coupling-chart singularity)
        #   * = chi = 1  (crossover)
        #   + = Theta=0  (bare-chart singularity)
        for beta_cross in tr["event_betas"]["mf0"]:
            x_cross = float(np.interp(beta_cross, tr["beta"], xproj))
            z_cross = float(np.interp(beta_cross, tr["beta"], tr["mz"]))
            axA.plot(x_cross, z_cross, marker="x", ms=5, mew=1.0, color=color, zorder=6)
        for beta_cross in tr["event_betas"]["chi1"]:
            x_cross = float(np.interp(beta_cross, tr["beta"], xproj))
            z_cross = float(np.interp(beta_cross, tr["beta"], tr["mz"]))
            axA.plot(x_cross, z_cross, marker="*", ms=6.0, mew=0.6, color=color, zorder=6, mec="white")
        for beta_cross in tr["event_betas"]["theta0"]:
            x_cross = float(np.interp(beta_cross, tr["beta"], xproj))
            z_cross = float(np.interp(beta_cross, tr["beta"], tr["mz"]))
            axA.plot(x_cross, z_cross, marker="+", ms=5, mew=1.0, color=color, zorder=6)

    axA.set_xlim(-0.01, 1.10)
    axA.set_ylim(-1.05, 0.05)
    axA.set_xlabel(r"$\sqrt{\max(m_x,0)}$")
    axA.set_ylabel(r"$m_z$")
    axA.set_title("A. Folded Projection + Candidate Loci")
    panelA_handles = [
        mpl.lines.Line2D([], [], marker="x", color="k", ls="none", ms=5, label=r"$m\!\cdot\!\hat f=0$"),
        mpl.lines.Line2D([], [], marker="*", color="k", ls="none", ms=6, label=r"$\chi=1$"),
        mpl.lines.Line2D([], [], marker="+", color="k", ls="none", ms=5, label=r"$\Theta=0$"),
    ]
    axA.legend(handles=panelA_handles, fontsize=6, loc="lower left", framealpha=0.9)

    # Panel B: coupling-axis chart continuity + exploratory reflected relabeling
    for mult in mults_focus:
        g = g_by_mult[mult]
        color = color_of_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]

        axB.plot(beta, tr["alpha_f_cont"], color=color, lw=1.6, label=rf"${mult:.1f}\,g_{{\star}}^{{(\rm ref)}}$")
        axB.plot(beta, tr["alpha_f_wrap"], color=color, lw=0.9, ls="--", alpha=0.85)
        # Exploratory chart relabeling suggested in discussion: theta -> pi-theta
        # applied here to the modulo-pi coupling-axis angle (director chart).
        axB.plot(beta, tr["alpha_f_axis_reflect"], color=color, lw=0.9, ls="-.", alpha=0.9)
        # Retain the original bare-axis principal-atan chart as a faint comparator.
        axB.plot(beta, tr["phi_tan_principal"], color=color, lw=0.7, ls=":", alpha=0.35)

        for beta_cross in tr["event_betas"]["mf0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["alpha_f_cont"]))
            axB.plot(beta_cross, y_cross, marker="x", ms=5, mew=1.0, color=color)
        for beta_cross in tr["event_betas"]["chi1"]:
            y_cross = float(np.interp(beta_cross, beta, tr["alpha_f_cont"]))
            axB.plot(beta_cross, y_cross, marker="*", ms=6, mew=0.6, color=color, mec="white")
        for beta_cross in tr["event_betas"]["theta0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["alpha_f_cont"]))
            axB.plot(beta_cross, y_cross, marker="+", ms=5, mew=1.0, color=color)

    axB.set_xscale("log")
    axB.set_xlabel(r"$\beta\omega_q$")
    axB.set_ylabel(r"Angle (rad)")
    axB.set_title(r"B. $f$-Chart Continuity and $\theta\to\pi-\theta$")
    axB.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    axB.axhline(np.pi / 2.0, color="k", ls=":", lw=0.7, alpha=0.35)
    axB.axhline(-np.pi / 2.0, color="k", ls=":", lw=0.7, alpha=0.20)
    axB.grid(True, which="both", alpha=0.2, lw=0.4)
    handles, labels = axB.get_legend_handles_labels()
    angle_style_handles = [
        mpl.lines.Line2D([], [], color="k", lw=1.4, label=r"$\alpha_f^{\rm cont}$ (unwrap)"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls="--", label=r"$\alpha_f^{\rm wrap}$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls="-.", label=r"reflected axis chart: $\pi-\alpha_f^{(\rm axis)}$"),
        mpl.lines.Line2D([], [], color="k", lw=0.7, ls=":", label=r"bare chart principal $\arctan(-\kappa/\sinh\Theta)$"),
        mpl.lines.Line2D([], [], color="k", marker="x", ls="none", ms=5, label=r"$m\!\cdot\!\hat f=0$"),
        mpl.lines.Line2D([], [], color="k", marker="*", ls="none", ms=6, label=r"$\chi=1$"),
        mpl.lines.Line2D([], [], color="k", marker="+", ls="none", ms=5, label=r"$\Theta=0$"),
    ]
    axB.legend(handles=handles + angle_style_handles, fontsize=6, loc="lower right", framealpha=0.9)

    # Panel C: sign-sensitive state components + h_eff direction in coupling basis
    for mult in mults_focus:
        g = g_by_mult[mult]
        color = color_of_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]

        axC.plot(beta, tr["m_f"], color=color, lw=1.5)
        axC.plot(beta, tr["m_f_perp"], color=color, lw=1.2, ls="--")
        axC.plot(beta, tr["mz"], color=color, lw=0.9, ls=":")
        axC.plot(beta, tr["hhat_f"], color=color, lw=0.9, ls="-.", alpha=0.95)
        axC.plot(beta, tr["hhat_f_perp"], color=color, lw=0.8, ls=(0, (1.2, 1.2)), alpha=0.9)

        for beta_cross in tr["event_betas"]["mf0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["m_f"]))
            axC.plot(beta_cross, y_cross, marker="x", ms=5, mew=1.0, color=color)
        for beta_cross in tr["event_betas"]["chi1"]:
            y_cross = float(np.interp(beta_cross, beta, tr["m_f"]))
            axC.plot(beta_cross, y_cross, marker="*", ms=6, mew=0.6, color=color, mec="white")
        for beta_cross in tr["event_betas"]["theta0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["m_f"]))
            axC.plot(beta_cross, y_cross, marker="+", ms=5, mew=1.0, color=color)

    axC.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.3)
    axC.set_xscale("log")
    axC.set_ylim(-1.05, 1.05)
    axC.set_xlabel(r"$\beta\omega_q$")
    axC.set_ylabel(r"Signed components / projections")
    axC.set_title(r"C. State Signs + $\hat h_{\rm eff}$ Direction")
    axC.grid(True, which="both", alpha=0.2, lw=0.4)
    comp_handles = [
        mpl.lines.Line2D([], [], color="k", lw=1.5, label=r"$m\cdot\hat f$"),
        mpl.lines.Line2D([], [], color="k", lw=1.2, ls="--", label=r"$m\cdot\hat n_\perp^{(f)}$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls=":", label=r"$m_z$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls="-.", label=r"$\hat h_{\rm eff}\cdot\hat f$"),
        mpl.lines.Line2D([], [], color="k", lw=0.8, ls=(0, (1.2, 1.2)), label=r"$\hat h_{\rm eff}\cdot\hat n_\perp^{(f)}$"),
        mpl.lines.Line2D([], [], color="k", marker="x", ls="none", ms=5, label=r"$m\!\cdot\!\hat f=0$"),
        mpl.lines.Line2D([], [], color="k", marker="*", ls="none", ms=6, label=r"$\chi=1$"),
        mpl.lines.Line2D([], [], color="k", marker="+", ls="none", ms=5, label=r"$\Theta=0$"),
    ]
    axC.legend(handles=comp_handles, fontsize=6, loc="lower right", framealpha=0.9)

    fig.suptitle("Exact Qubit Branch Diagnostics (Standalone Test)", y=1.02, fontsize=10)
    fig.tight_layout()

    out_png = FIGURES_DIR / "hmf_branch_flip_test.png"
    out_pdf = FIGURES_DIR / "hmf_branch_flip_test.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")

    # -----------------------------------------------------------------------
    # Figure 2: Tilt-angle branch test with explicit magnetisation consequences
    # -----------------------------------------------------------------------
    fig2 = plt.figure(figsize=(9.0, 3.8), constrained_layout=True)
    gs2 = GridSpec(1, 3, figure=fig2, width_ratios=[1.25, 1.0, 1.05], wspace=0.32)
    axM = fig2.add_subplot(gs2[0, 0])
    axT = fig2.add_subplot(gs2[0, 1])
    axR = fig2.add_subplot(gs2[0, 2])

    mults_mag = [0.9, 1.1, 1.4]
    # Panel M: magnetisation components (exact) across beta
    for mult in mults_mag:
        g = g_by_mult[mult]
        color = color_of_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]
        axM.plot(beta, tr["mz"], color=color, lw=1.6, label=rf"${mult:.1f}\,g_{{\star}}^{{(\rm ref)}}$: $m_z$")
        axM.plot(beta, tr["m_perp_signed"], color=color, lw=1.1, ls="--", alpha=0.95,
                 label=rf"${mult:.1f}\,g_{{\star}}^{{(\rm ref)}}$: $m_\perp^{{\rm (sgn)}}$")
        axM.plot(beta, tr["m_f"], color=color, lw=0.9, ls=":", alpha=0.95,
                 label=rf"${mult:.1f}\,g_{{\star}}^{{(\rm ref)}}$: $m\!\cdot\!\hat f$")

        for beta_cross in tr["event_betas"]["chi1"]:
            y_cross = float(np.interp(beta_cross, beta, tr["mz"]))
            axM.plot(beta_cross, y_cross, marker="*", ms=6, color=color, mec="white", mew=0.6)
        for beta_cross in tr["event_betas"]["theta0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["mz"]))
            axM.plot(beta_cross, y_cross, marker="+", ms=5, color=color, mew=1.0)
        for beta_cross in tr["event_betas"]["mf0"]:
            y_cross = float(np.interp(beta_cross, beta, tr["m_f"]))
            axM.plot(beta_cross, y_cross, marker="x", ms=5, color=color, mew=1.0)

    axM.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.35)
    axM.set_xscale("log")
    axM.set_ylim(-1.05, 1.05)
    axM.set_xlabel(r"$\beta\omega_q$")
    axM.set_ylabel("Magnetisation components")
    axM.set_title("A. Exact Magnetisation vs Temperature")
    axM.grid(True, which="both", alpha=0.2, lw=0.4)
    style_handles_M = [
        mpl.lines.Line2D([], [], color="k", lw=1.6, label=r"$m_z$"),
        mpl.lines.Line2D([], [], color="k", lw=1.1, ls="--", label=r"$m_\perp^{(\rm signed)}$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls=":", label=r"$m\cdot \hat f$"),
        mpl.lines.Line2D([], [], color="k", marker="*", ls="none", ms=6, label=r"$\chi=1$"),
        mpl.lines.Line2D([], [], color="k", marker="+", ls="none", ms=5, label=r"$\Theta=0$"),
        mpl.lines.Line2D([], [], color="k", marker="x", ls="none", ms=5, label=r"$m\cdot \hat f=0$"),
    ]
    color_handles_M = [
        mpl.lines.Line2D([], [], color=color_of_mult[m], lw=2.0, label=rf"${m:.1f}\,g_{{\star}}^{{(\rm ref)}}$")
        for m in mults_mag
    ]
    axM.legend(handles=color_handles_M + style_handles_M, fontsize=5.7, loc="lower left", framealpha=0.9, ncol=2)

    # Panel T: manuscript tilt-angle charts (phi) for a representative curve
    mult_tilt = 1.1
    g_tilt = g_by_mult[mult_tilt]
    trT = traces[float(g_tilt)]
    betaT = trT["beta"]
    phi_axis = np.mod(trT["phi_wrap"], np.pi)
    phi_axis_reflect = np.pi - phi_axis

    axT.plot(betaT, trT["phi_tan_principal"], color="#c63d2f", lw=1.2, ls=":", label=r"$\varphi_{\rm principal}=\arctan(-\kappa/\sinh\Theta)$")
    axT.plot(betaT, trT["phi_wrap"], color="#5d63d6", lw=1.0, ls="--", label=r"$\varphi_{\rm wrap}=\operatorname{atan2}(m_\perp,-m_z)$")
    axT.plot(betaT, trT["phi_cont"], color="#1f4ea8", lw=1.7, label=r"$\varphi_{\rm cont}$ (unwrap)")
    axT.plot(betaT, phi_axis, color="#2a8f5a", lw=1.0, ls="-.", label=r"$\varphi_{\rm axis}\in[0,\pi)$")
    axT.plot(betaT, phi_axis_reflect, color="#00a6a6", lw=1.1, ls=(0, (2.5, 1.2)),
             label=r"reflected axis chart: $\pi-\varphi_{\rm axis}$")

    for beta_cross in trT["event_betas"]["chi1"]:
        y_cross = float(np.interp(beta_cross, betaT, trT["phi_cont"]))
        axT.plot(beta_cross, y_cross, marker="*", ms=6, color="k", mec="white", mew=0.6)
    for beta_cross in trT["event_betas"]["theta0"]:
        y_cross = float(np.interp(beta_cross, betaT, trT["phi_cont"]))
        axT.plot(beta_cross, y_cross, marker="+", ms=5, color="k", mew=1.0)

    axT.axhline(np.pi / 2.0, color="k", ls=":", lw=0.7, alpha=0.35)
    axT.axhline(-np.pi / 2.0, color="k", ls=":", lw=0.5, alpha=0.20)
    axT.set_xscale("log")
    axT.set_xlabel(r"$\beta\omega_q$")
    axT.set_ylabel(r"Tilt angle $\varphi$ (rad)")
    axT.set_title(r"B. Tilt-Angle Chart Continuity ($\varphi$)")
    axT.grid(True, which="both", alpha=0.2, lw=0.4)
    axT.legend(fontsize=5.5, loc="lower right", framealpha=0.9)

    # Panel R: magnetisation implications of enforcing the reflected tilt-axis chart
    rT = np.sqrt(np.clip(trT["mz"] ** 2 + trT["m_perp_signed"] ** 2, 0.0, None))
    mperp_reflect = rT * np.sin(phi_axis_reflect)
    mz_reflect = -rT * np.cos(phi_axis_reflect)

    axR.plot(betaT, trT["mz"], color="#202020", lw=1.6, label=r"exact $m_z$")
    axR.plot(betaT, mz_reflect, color="#d62728", lw=1.4, ls="--", label=r"$m_z$ from reflected $\pi-\varphi_{\rm axis}$")
    axR.plot(betaT, trT["m_perp_signed"], color="#6b6b6b", lw=1.1, label=r"exact $m_\perp^{(\rm signed)}$")
    axR.plot(betaT, mperp_reflect, color="#2ca02c", lw=1.1, ls="--",
             label=r"$m_\perp$ from reflected $\pi-\varphi_{\rm axis}$")

    for beta_cross in trT["event_betas"]["chi1"]:
        y_cross = float(np.interp(beta_cross, betaT, trT["mz"]))
        axR.plot(beta_cross, y_cross, marker="*", ms=6, color="k", mec="white", mew=0.6)
    for beta_cross in trT["event_betas"]["theta0"]:
        y_cross = float(np.interp(beta_cross, betaT, trT["mz"]))
        axR.plot(beta_cross, y_cross, marker="+", ms=5, color="k", mew=1.0)

    axR.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.35)
    axR.set_xscale("log")
    axR.set_ylim(-1.05, 1.05)
    axR.set_xlabel(r"$\beta\omega_q$")
    axR.set_ylabel("Magnetisation")
    axR.set_title(r"C. Magnetisation Implied by $\varphi\mapsto\pi-\varphi$")
    axR.grid(True, which="both", alpha=0.2, lw=0.4)
    axR.legend(fontsize=5.5, loc="lower right", framealpha=0.9)

    fig2.suptitle(
        rf"Tilt-Angle Branch Test with Magnetisation (representative curve: $g={mult_tilt:.1f}\,g_{{\star}}^{{(\rm ref)}}$)",
        y=1.03,
        fontsize=10,
    )

    out2_png = FIGURES_DIR / "hmf_tilt_magnetisation_branch_test.png"
    out2_pdf = FIGURES_DIR / "hmf_tilt_magnetisation_branch_test.pdf"
    fig2.savefig(out2_png, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved -> {out2_png}")
    print(f"Saved -> {out2_pdf}")

    # -----------------------------------------------------------------------
    # Figure 3: Explicit T>T_* branch-continuation overlay (user-requested)
    # -----------------------------------------------------------------------
    fig3 = plt.figure(figsize=(10.2, 3.7), constrained_layout=True)
    gs3 = GridSpec(1, 3, figure=fig3, width_ratios=[1.0, 1.2, 1.2], wspace=0.30)
    axF = fig3.add_subplot(gs3[0, 0])  # folded context
    axAng = fig3.add_subplot(gs3[0, 1])  # tilt axis angle piecewise continuation
    axS = fig3.add_subplot(gs3[0, 2])  # signed-plane overlay

    # Couplings for which chi=1 exists on the chosen beta window.
    mults_with_chi1 = [m for m in G_MULTS_B if len(traces[float(g_by_mult[m])]["event_betas"]["chi1"]) > 0]
    mults_demo = [m for m in mults_with_chi1 if m in (0.7, 0.9, 1.1)] or mults_with_chi1[-3:]

    phi_f_axis = float(np.pi - THETA_VAL)  # interaction direction in the tilt-axis chart

    # Panel F: context (Fig. 7(b)-style folded trajectories + T*=chi=1 markers)
    X_bound3 = np.linspace(0.0, 1.05, 500)
    Z_bound3 = -np.sqrt(np.clip(1.0 - X_bound3**4, 0.0, 1.0))
    axF.plot(X_bound3, Z_bound3, color="black", lw=1.0, zorder=1)
    axF.fill_between(X_bound3, Z_bound3, 0.1, color="#f7f7f7", zorder=0)
    axF.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.3)
    axF.axvline(0.0, color="k", ls=":", lw=0.5, alpha=0.3)
    for mult in mults_with_chi1:
        g = g_by_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]
        color = color_of_mult[mult]
        xproj = np.sqrt(np.clip(tr["mx"], 0.0, None))
        axF.plot(xproj, tr["mz"], color=color, lw=1.25, alpha=0.9)
        for beta_star in tr["event_betas"]["chi1"]:
            x_star = float(np.interp(beta_star, beta, xproj))
            z_star = float(np.interp(beta_star, beta, tr["mz"]))
            axF.plot(x_star, z_star, marker="*", ms=6, color=color, mec="white", mew=0.6)
    axF.set_xlim(-0.01, 1.10)
    axF.set_ylim(-1.05, 0.05)
    axF.set_xlabel(r"$\sqrt{\max(m_x,0)}$")
    axF.set_ylabel(r"$m_z$")
    axF.set_title(r"A. Fig.~7(b)-Style Curves with $T_\star$ Markers")
    axF.legend(
        handles=[
            mpl.lines.Line2D([], [], marker="*", color="k", ls="none", ms=6, label=r"$\chi=1 \;\Leftrightarrow\; T=T_\star(g)$"),
        ],
        fontsize=6,
        loc="lower left",
        framealpha=0.9,
    )

    # Panel Ang: apply a branch-continuation transform only for T>T_*
    for mult in mults_with_chi1:
        g = g_by_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]
        color = color_of_mult[mult]
        phi_axis = np.mod(tr["phi_wrap"], np.pi)
        beta_star = float(tr["event_betas"]["chi1"][0]) if len(tr["event_betas"]["chi1"]) else None
        phi_axis_bc = reflected_director_continuation_above_beta_star(beta, phi_axis, beta_star, phi_f_axis)

        # exact (apparent/reflected axis chart) and continued branch overlay
        axAng.plot(beta, phi_axis, color=color, lw=0.95, alpha=0.85)
        axAng.plot(beta, phi_axis_bc, color=color, lw=1.9, ls="--", alpha=0.95,
                   label=rf"${mult:.1f}\,g_{{\star}}^{{(\rm ref)}}$")

        if beta_star is not None and np.isfinite(beta_star):
            y_star = float(np.interp(beta_star, beta, phi_axis_bc))
            axAng.plot(beta_star, y_star, marker="*", ms=6, color=color, mec="white", mew=0.6)

    axAng.axhline(phi_f_axis, color="k", ls="-.", lw=0.9, alpha=0.7, label=r"interaction axis: $\varphi_f=\pi-\theta$")
    axAng.axhline(float(THETA_VAL), color="k", ls=":", lw=0.8, alpha=0.5, label=r"reflected image: $\theta$")
    axAng.set_xscale("log")
    axAng.set_xlabel(r"$\beta\omega_q$ (high $T$ on the left)")
    axAng.set_ylabel(r"Tilt axis angle $\varphi_{\rm axis}$ / continued lift")
    axAng.set_title(r"B. Apply Branch Continuity for $T>T_\star$ Only")
    axAng.grid(True, which="both", alpha=0.2, lw=0.4)
    angle_demo_handles = [
        mpl.lines.Line2D([], [], color="k", lw=0.95, label=r"apparent branch ($\varphi_{\rm axis}$)"),
        mpl.lines.Line2D([], [], color="k", lw=1.9, ls="--", label=r"continued branch for $T>T_\star$"),
        mpl.lines.Line2D([], [], color="k", marker="*", ls="none", ms=6, label=r"$T=T_\star(g)$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls="-.", label=r"$\pi-\theta$"),
        mpl.lines.Line2D([], [], color="k", lw=0.8, ls=":", label=r"$\theta$"),
    ]
    color_handles_demo = [
        mpl.lines.Line2D([], [], color=color_of_mult[m], lw=2.0, label=rf"${m:.1f}\,g_{{\star}}^{{(\rm ref)}}$")
        for m in mults_with_chi1
    ]
    axAng.legend(handles=color_handles_demo + angle_demo_handles, fontsize=5.8, loc="center right", framealpha=0.92)

    # Panel S: map the branch-continuation angle back into the (m_perp, m_z) plane
    # (interpretive overlay only; the exact equilibrium state remains the solid curve)
    t_line = np.linspace(0.0, 1.0, 2)
    axS.plot(t_line * np.sin(phi_f_axis), -t_line * np.cos(phi_f_axis), color="k", ls="-.", lw=0.9, alpha=0.7)
    axS.plot(t_line * np.sin(float(THETA_VAL)), -t_line * np.cos(float(THETA_VAL)), color="k", ls=":", lw=0.8, alpha=0.5)

    for mult in mults_demo:
        g = g_by_mult[mult]
        tr = traces[float(g)]
        beta = tr["beta"]
        color = color_of_mult[mult]
        phi_axis = np.mod(tr["phi_wrap"], np.pi)
        beta_star = float(tr["event_betas"]["chi1"][0]) if len(tr["event_betas"]["chi1"]) else None
        phi_axis_bc = reflected_director_continuation_above_beta_star(beta, phi_axis, beta_star, phi_f_axis)

        r_arr = tr["r"]
        mperp_bc = r_arr * np.sin(phi_axis_bc)
        mz_bc = -r_arr * np.cos(phi_axis_bc)

        axS.plot(tr["m_perp_signed"], tr["mz"], color=color, lw=1.0, alpha=0.45)

        if beta_star is not None and np.isfinite(beta_star):
            mask_hiT = beta < beta_star
            mask_loT = ~mask_hiT
        else:
            mask_hiT = np.zeros_like(beta, dtype=bool)
            mask_loT = np.ones_like(beta, dtype=bool)

        # Keep the exact low-T branch as context; replace only the high-T segment by the continued chart image.
        axS.plot(tr["m_perp_signed"][mask_loT], tr["mz"][mask_loT], color=color, lw=1.4, alpha=0.95)
        if np.any(mask_hiT):
            axS.plot(mperp_bc[mask_hiT], mz_bc[mask_hiT], color=color, lw=1.8, ls="--", alpha=0.95)
            # Connect to the crossover point for readability
            i_last = int(np.where(mask_hiT)[0][-1])
            if i_last + 1 < len(beta):
                axS.plot(
                    [mperp_bc[i_last], tr["m_perp_signed"][i_last + 1]],
                    [mz_bc[i_last], tr["mz"][i_last + 1]],
                    color=color,
                    lw=0.8,
                    ls="--",
                    alpha=0.6,
                )
        for beta_star in tr["event_betas"]["chi1"]:
            x_star = float(np.interp(beta_star, beta, tr["m_perp_signed"]))
            z_star = float(np.interp(beta_star, beta, tr["mz"]))
            axS.plot(x_star, z_star, marker="*", ms=6, color=color, mec="white", mew=0.6)

    axS.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    axS.axvline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    axS.set_xlim(-1.05, 1.05)
    axS.set_ylim(-1.05, 1.05)
    axS.set_xlabel(r"signed transverse magnetisation $m_\perp^{(\rm signed)}$")
    axS.set_ylabel(r"$m_z$")
    axS.set_title(r"C. Interpreted Continuation in the $(m_\perp,m_z)$ Plane")
    axS.grid(True, alpha=0.18, lw=0.4)
    signed_demo_handles = [
        mpl.lines.Line2D([], [], color="k", lw=1.0, alpha=0.45, label="exact trajectory (all T)"),
        mpl.lines.Line2D([], [], color="k", lw=1.8, ls="--", label=r"branch-continued image for $T>T_\star$"),
        mpl.lines.Line2D([], [], color="k", marker="*", ls="none", ms=6, label=r"$T=T_\star(g)$"),
        mpl.lines.Line2D([], [], color="k", lw=0.9, ls="-.", label=r"$\varphi_f=\pi-\theta$ direction"),
        mpl.lines.Line2D([], [], color="k", lw=0.8, ls=":", label=r"reflected image $\theta$"),
    ]
    axS.legend(handles=signed_demo_handles, fontsize=5.8, loc="lower left", framealpha=0.92)

    fig3.suptitle(
        r"Branch-Continuation Applied Above $T_\star$: Precession Continuation vs Apparent Reflection",
        y=1.02,
        fontsize=10,
    )

    out3_png = FIGURES_DIR / "hmf_tstar_branch_continuation_demo.png"
    out3_pdf = FIGURES_DIR / "hmf_tstar_branch_continuation_demo.pdf"
    fig3.savefig(out3_png, bbox_inches="tight")
    fig3.savefig(out3_pdf, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved -> {out3_png}")
    print(f"Saved -> {out3_pdf}")

    # -----------------------------------------------------------------------
    # Figure 4: Tilt branch-sheet connections at splitting extremum + purity
    # -----------------------------------------------------------------------
    mult_sheet = 1.1 if 1.1 in g_by_mult else sorted(g_by_mult.keys())[0]
    trS = traces[float(g_by_mult[mult_sheet])]
    betaS = trS["beta"]
    rS = trS["r"]

    # Director-style tilt axis chart and its reflected image (mod pi).
    phi_axis_S = np.mod(trS["phi_wrap"], np.pi)
    phi_ref_S = np.pi - phi_axis_S

    # Use the lower branch hump (reflected image in the current plot convention)
    # to define the "splitting maximum" where branch-sheet hopping is visualized.
    i_split = int(np.argmax(phi_ref_S))
    beta_split = float(betaS[i_split])

    beta_star_S = None
    if len(trS["event_betas"]["chi1"]):
        beta_star_S = float(trS["event_betas"]["chi1"][0])

    # Four lifted sheet images on the universal-cover angle axis.
    sheet_lo_0 = phi_ref_S
    sheet_hi_0 = phi_axis_S
    sheet_lo_1 = phi_ref_S + np.pi
    sheet_hi_1 = phi_axis_S + np.pi

    # Example connected continuations (branch hops at the splitting extremum).
    # These are interpretive branch-sheet continuations, not changes in the
    # exact Cartesian state.
    path_A = sheet_lo_0                               # no hop
    path_B = splice_after_index(sheet_lo_0, sheet_hi_0, i_split)  # hop at split
    path_C = sheet_lo_1                               # parity-lifted no hop
    path_D = splice_after_index(sheet_lo_1, sheet_hi_1, i_split)  # parity-lifted hop

    fig4 = plt.figure(figsize=(10.2, 3.9), constrained_layout=True)
    gs4 = GridSpec(1, 3, figure=fig4, width_ratios=[1.25, 1.1, 0.95], wspace=0.28)
    ax4A = fig4.add_subplot(gs4[0, 0])
    ax4B = fig4.add_subplot(gs4[0, 1])
    ax4C = fig4.add_subplot(gs4[0, 2])

    # Panel A: sheet structure + split/crossover markers.
    ax4A.plot(betaS, sheet_lo_0, color="#27bfc7", lw=1.5, label=r"$\pi-\varphi_{\rm axis}$")
    ax4A.plot(betaS, sheet_hi_0, color="#2ea35f", lw=1.5, ls="-.", label=r"$\varphi_{\rm axis}$")
    ax4A.plot(betaS, sheet_lo_1, color="#4b7cd0", lw=1.2, label=r"$\pi-\varphi_{\rm axis}+\pi$")
    ax4A.plot(betaS, sheet_hi_1, color="#2947a3", lw=1.2, ls="-.", label=r"$\varphi_{\rm axis}+\pi$")
    ax4A.axvline(beta_split, color="#b32020", lw=0.9, ls="--", alpha=0.8)
    if beta_star_S is not None:
        ax4A.axvline(beta_star_S, color="k", lw=0.8, ls=":", alpha=0.7)
    for y in [sheet_lo_0[i_split], sheet_hi_0[i_split], sheet_lo_1[i_split], sheet_hi_1[i_split]]:
        ax4A.plot(beta_split, float(y), marker="o", ms=3.8, color="#b32020", mec="white", mew=0.4, zorder=6)
    # Visual "possible hop" connectors at the splitting extremum.
    ax4A.plot([beta_split, beta_split], [sheet_lo_0[i_split], sheet_hi_0[i_split]], color="#b32020", ls="--", lw=0.8)
    ax4A.plot([beta_split, beta_split], [sheet_lo_1[i_split], sheet_hi_1[i_split]], color="#b32020", ls="--", lw=0.8)
    ax4A.set_xscale("log")
    ax4A.set_xlabel(r"$\beta\omega_q$")
    ax4A.set_ylabel(r"Lifted tilt-axis sheets (rad)")
    ax4A.set_title(r"A. Tilt Branch Sheets + Splitting Maximum")
    ax4A.grid(True, which="both", alpha=0.2, lw=0.4)
    handlesA = [
        mpl.lines.Line2D([], [], color="#27bfc7", lw=1.5, label=r"$\pi-\varphi_{\rm axis}$"),
        mpl.lines.Line2D([], [], color="#2ea35f", lw=1.5, ls="-.", label=r"$\varphi_{\rm axis}$"),
        mpl.lines.Line2D([], [], color="#4b7cd0", lw=1.2, label=r"$\pi-\varphi_{\rm axis}+\pi$"),
        mpl.lines.Line2D([], [], color="#2947a3", lw=1.2, ls="-.", label=r"$\varphi_{\rm axis}+\pi$"),
        mpl.lines.Line2D([], [], color="#b32020", lw=0.9, ls="--", label=r"split extremum $\beta_{\rm split}$"),
        mpl.lines.Line2D([], [], color="k", lw=0.8, ls=":", label=r"$\beta_\star$ ($\chi=1$)"),
    ]
    ax4A.legend(handles=handlesA, fontsize=5.7, loc="center left", framealpha=0.92)

    # Panel B: connected branch continuations (example branch hops).
    branch_specs = [
        ("A: no hop", path_A, "#27bfc7", "-"),
        ("B: hop at split", path_B, "#cf2a2a", "--"),
        ("C: no hop (+π)", path_C, "#4b7cd0", "-"),
        ("D: hop at split (+π)", path_D, "#6f3db8", "--"),
    ]
    for label, arr, color, ls in branch_specs:
        ax4B.plot(betaS, arr, color=color, lw=1.6, ls=ls, label=label)
        ax4B.plot(betaS[-1], arr[-1], marker="o", ms=3.5, color=color, mec="white", mew=0.4)
    ax4B.axvline(beta_split, color="#b32020", lw=0.9, ls="--", alpha=0.8)
    if beta_star_S is not None:
        ax4B.axvline(beta_star_S, color="k", lw=0.8, ls=":", alpha=0.7)
    ax4B.set_xscale("log")
    ax4B.set_xlabel(r"$\beta\omega_q$ (increasing toward the right)")
    ax4B.set_ylabel(r"Connected branch angle (lifted chart)")
    ax4B.set_title(r"B. Example Branch Connections at $\beta_{\rm split}$")
    ax4B.grid(True, which="both", alpha=0.2, lw=0.4)
    ax4B.legend(fontsize=5.6, loc="center left", framealpha=0.92)

    # Panel C: purity of each branch (identical by construction).
    for label, arr, color, ls in branch_specs:
        ax4C.plot(betaS, rS, color=color, lw=1.5, ls=ls, alpha=0.95, label=label)
    ax4C.axvline(beta_split, color="#b32020", lw=0.9, ls="--", alpha=0.8)
    if beta_star_S is not None:
        ax4C.axvline(beta_star_S, color="k", lw=0.8, ls=":", alpha=0.7)
    ax4C.set_xscale("log")
    ax4C.set_ylim(-0.02, 1.02)
    ax4C.set_xlabel(r"$\beta\omega_q$")
    ax4C.set_ylabel(r"Purity $r=\sqrt{m_z^2+m_\perp^2}$")
    ax4C.set_title(r"C. Purity Is Branch-Invariant (Curves Overlap)")
    ax4C.grid(True, which="both", alpha=0.2, lw=0.4)
    ax4C.text(
        0.05,
        0.10,
        "same exact radius\nfor all branch-sheet\ncontinuations",
        transform=ax4C.transAxes,
        fontsize=6.3,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )

    fig4.suptitle(
        rf"Tilt-Angle Branch Sheets, Split-Extremum Connections, and Purity (representative: $g={mult_sheet:.1f}\,g_{{\star}}^{{(\rm ref)}}$)",
        y=1.02,
        fontsize=10,
    )

    out4_png = FIGURES_DIR / "hmf_tilt_branch_sheet_purity_demo.png"
    out4_pdf = FIGURES_DIR / "hmf_tilt_branch_sheet_purity_demo.pdf"
    fig4.savefig(out4_png, bbox_inches="tight")
    fig4.savefig(out4_pdf, bbox_inches="tight")
    plt.close(fig4)

    print(f"Saved -> {out4_png}")
    print(f"Saved -> {out4_pdf}")

    # -----------------------------------------------------------------------
    # Figure 5: Full Bloch-circle view with theta-offset branch versions
    # -----------------------------------------------------------------------
    mult_theta_demo = 1.1 if 1.1 in g_by_mult else sorted(g_by_mult.keys())[0]
    tr5 = traces[float(g_by_mult[mult_theta_demo])]
    beta5 = tr5["beta"]
    r5 = tr5["r"]
    phi_axis5 = np.mod(tr5["phi_wrap"], np.pi)
    delta5 = np.pi - phi_axis5  # small positive branch separation that -> 0 as beta->infty

    # "Versions" of the tilt angle that asymptote to the original mixing angle theta.
    # Measured in the same polar convention as phi_wrap: angle from -z toward +m_perp.
    alpha_theta_plus = THETA_VAL + delta5
    alpha_theta_minus = THETA_VAL - delta5
    alpha_theta_plus_lift = alpha_theta_plus + np.pi
    alpha_theta_minus_lift = alpha_theta_minus + np.pi

    # Branch-selected continuation at T*=beta*:
    # choose the "minus" sheet above T* (high T, beta<beta*) and the "plus"
    # sheet below T* (low T, beta>beta*) so the selected branch tends to theta
    # as beta->infty.
    beta_star5 = float(tr5["event_betas"]["chi1"][0]) if len(tr5["event_betas"]["chi1"]) else None
    if beta_star5 is not None:
        mask_hiT_5 = beta5 < beta_star5
    else:
        mask_hiT_5 = np.zeros_like(beta5, dtype=bool)
    mask_loT_5 = ~mask_hiT_5
    alpha_theta_selected = np.where(mask_hiT_5, alpha_theta_minus, alpha_theta_plus)

    # Map angle versions back to the full (m_perp_signed, m_z) Bloch disk.
    # Polar convention: alpha=0 is along -z, so (m_perp, mz)=(r sin alpha, -r cos alpha).
    mperp_plus = r5 * np.sin(alpha_theta_plus)
    mz_plus = -r5 * np.cos(alpha_theta_plus)
    mperp_minus = r5 * np.sin(alpha_theta_minus)
    mz_minus = -r5 * np.cos(alpha_theta_minus)
    mperp_sel = r5 * np.sin(alpha_theta_selected)
    mz_sel = -r5 * np.cos(alpha_theta_selected)

    fig5 = plt.figure(figsize=(10.4, 3.9), constrained_layout=True)
    gs5 = GridSpec(1, 3, figure=fig5, width_ratios=[1.15, 1.1, 0.95], wspace=0.28)
    ax5A = fig5.add_subplot(gs5[0, 0])
    ax5B = fig5.add_subplot(gs5[0, 1])
    ax5C = fig5.add_subplot(gs5[0, 2])

    # Panel A: full Bloch circle (no sqrt fold)
    t_ang = np.linspace(0.0, 2.0 * np.pi, 800)
    ax5A.plot(np.cos(t_ang), np.sin(t_ang), color="black", lw=1.0, alpha=0.9)
    ax5A.axhline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    ax5A.axvline(0.0, color="k", ls=":", lw=0.5, alpha=0.25)
    # Reference rays: theta and theta+pi in this polar convention.
    for ang, ls, col, lbl in [
        (float(THETA_VAL), "-.", "#444444", r"$\theta$"),
        (float(THETA_VAL + np.pi), ":", "#777777", r"$\theta+\pi$"),
    ]:
        rr = np.linspace(0.0, 1.0, 2)
        ax5A.plot(rr * np.sin(ang), -rr * np.cos(ang), ls=ls, lw=0.9, color=col, label=lbl)

    # Exact trajectory (physical) and theta-offset branch versions (interpretive).
    ax5A.plot(tr5["m_perp_signed"], tr5["mz"], color="#8a8a8a", lw=1.1, alpha=0.6, label="exact trajectory")
    ax5A.plot(mperp_plus, mz_plus, color="#1f77b4", lw=1.6, label=r"$\alpha_\theta^{(+)}=\theta+(\pi-\varphi_{\rm axis})$")
    ax5A.plot(mperp_minus, mz_minus, color="#d62728", lw=1.3, ls="--", label=r"$\alpha_\theta^{(-)}=\theta-(\pi-\varphi_{\rm axis})$")

    # Highlight the branch-selected continuation (switch at T*=beta*; then tends to theta)
    ax5A.plot(mperp_sel, mz_sel, color="#0f9d58", lw=2.0, alpha=0.95, label=r"selected branch at $T_\star$")
    ax5A.plot(mperp_sel[-1], mz_sel[-1], marker="o", ms=4.5, color="#0f9d58", mec="white", mew=0.5, zorder=6)

    if beta_star5 is not None:
        # mark T*=beta* on all plotted branches for direct visual comparison
        for x_arr, z_arr, mk_col in [
            (tr5["m_perp_signed"], tr5["mz"], "#8a8a8a"),
            (mperp_plus, mz_plus, "#1f77b4"),
            (mperp_minus, mz_minus, "#d62728"),
            (mperp_sel, mz_sel, "#0f9d58"),
        ]:
            x_star = float(np.interp(beta_star5, beta5, x_arr))
            z_star = float(np.interp(beta_star5, beta5, z_arr))
            ax5A.plot(x_star, z_star, marker="*", ms=6, color=mk_col, mec="white", mew=0.6, zorder=7)

    ax5A.set_aspect("equal", adjustable="box")
    ax5A.set_xlim(-1.05, 1.05)
    ax5A.set_ylim(-1.05, 1.05)
    ax5A.set_xlabel(r"signed transverse magnetisation $m_\perp^{(\rm signed)}$")
    ax5A.set_ylabel(r"$m_z$")
    ax5A.set_title(r"A. Full Bloch Circle (No Fold) + $\theta$-Offset Branches")
    ax5A.grid(True, alpha=0.18, lw=0.4)
    ax5A.legend(fontsize=5.3, loc="lower left", framealpha=0.92)

    # Panel B: angle versions and selected branch, with explicit theta asymptote line.
    ax5B.plot(beta5, alpha_theta_plus, color="#1f77b4", lw=1.6, label=r"$\alpha_\theta^{(+)}$")
    ax5B.plot(beta5, alpha_theta_minus, color="#d62728", lw=1.3, ls="--", label=r"$\alpha_\theta^{(-)}$")
    ax5B.plot(beta5, alpha_theta_plus_lift, color="#4f6ad7", lw=1.0, alpha=0.75, label=r"$\alpha_\theta^{(+)}+\pi$")
    ax5B.plot(beta5, alpha_theta_minus_lift, color="#b25ad6", lw=1.0, ls="--", alpha=0.75, label=r"$\alpha_\theta^{(-)}+\pi$")
    ax5B.plot(beta5, alpha_theta_selected, color="#0f9d58", lw=2.0, label=r"selected branch (switch at $T_\star$)")
    ax5B.axhline(float(THETA_VAL), color="#444444", lw=0.9, ls="-.", label=r"asymptote target $\theta$")
    ax5B.axhline(float(THETA_VAL + np.pi), color="#777777", lw=0.8, ls=":", label=r"parity image $\theta+\pi$")
    if beta_star5 is not None:
        ax5B.axvline(beta_star5, color="k", lw=0.8, ls=":", alpha=0.75)
        y_star_sel = float(np.interp(beta_star5, beta5, alpha_theta_selected))
        ax5B.plot(beta_star5, y_star_sel, marker="*", ms=6, color="#0f9d58", mec="white", mew=0.6)
    ax5B.set_xscale("log")
    ax5B.set_xlabel(r"$\beta\omega_q$")
    ax5B.set_ylabel(r"$\theta$-offset branch angles (rad)")
    ax5B.set_title(r"B. Branch Choice at $T_\star$ with $\beta\to\infty$ Endpoint at $\theta$")
    ax5B.grid(True, which="both", alpha=0.2, lw=0.4)
    ax5B.legend(fontsize=5.5, loc="center left", framealpha=0.92)

    # Panel C: corresponding purities (all identical)
    for color, ls, label in [
        ("#1f77b4", "-", r"$\alpha_\theta^{(+)}$"),
        ("#d62728", "--", r"$\alpha_\theta^{(-)}$"),
        ("#0f9d58", "-", r"selected at $T_\star$"),
    ]:
        ax5C.plot(beta5, r5, color=color, lw=1.5, ls=ls, alpha=0.95, label=label)
    if beta_star5 is not None:
        ax5C.axvline(beta_star5, color="k", lw=0.8, ls=":", alpha=0.75)
    ax5C.set_xscale("log")
    ax5C.set_ylim(-0.02, 1.02)
    ax5C.set_xlabel(r"$\beta\omega_q$")
    ax5C.set_ylabel(r"Purity $r$")
    ax5C.set_title(r"C. Purity for the Same Branch Versions (Overlap)")
    ax5C.grid(True, which="both", alpha=0.2, lw=0.4)
    ax5C.text(
        0.05,
        0.10,
        r"all use the same exact $r(\beta)=\tanh\chi$",
        transform=ax5C.transAxes,
        fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )
    ax5C.legend(fontsize=5.6, loc="lower right", framealpha=0.92)

    fig5.suptitle(
        rf"Full Bloch-Circle View of $\theta$-Offset Branch Versions (representative: $g={mult_theta_demo:.1f}\,g_{{\star}}^{{(\rm ref)}}$)",
        y=1.02,
        fontsize=10,
    )

    out5_png = FIGURES_DIR / "hmf_full_bloch_theta_branch_demo.png"
    out5_pdf = FIGURES_DIR / "hmf_full_bloch_theta_branch_demo.pdf"
    fig5.savefig(out5_png, bbox_inches="tight")
    fig5.savefig(out5_pdf, bbox_inches="tight")
    plt.close(fig5)

    print(f"Saved -> {out5_png}")
    print(f"Saved -> {out5_pdf}")

    # -----------------------------------------------------------------------
    # Brief terminal summary for the current run
    # -----------------------------------------------------------------------
    print("\nEvent-locus comparison (focus couplings):")
    for mult in mults_focus:
        g = g_by_mult[mult]
        tr = traces[float(g)]
        betas_mf0 = tr["event_betas"]["mf0"]
        betas_chi1 = tr["event_betas"]["chi1"]
        betas_theta0 = tr["event_betas"]["theta0"]

        beta_end = float(tr["beta"][-1])
        mf_end = float(tr["m_f"][-1])
        mfp_end = float(tr["m_f_perp"][-1])
        mz_end = float(tr["mz"][-1])
        hff_end = float(tr["hhat_f"][-1])
        hfperp_end = float(tr["hhat_f_perp"][-1])
        alphaf_end = float(tr["alpha_f_axis"][-1])
        alphaf_ref_end = float(tr["alpha_f_axis_reflect"][-1])

        line = (
            f"  g/g*_ref={mult:.2f}: "
            f"mf0={len(betas_mf0)}, chi1={len(betas_chi1)}, Theta0={len(betas_theta0)} | "
            f"beta_max={beta_end:.2f} "
            f"m_f={mf_end:+.4f}, m_f_perp={mfp_end:+.4f}, m_z={mz_end:+.4f} | "
            f"hhat_f={hff_end:+.4f}, hhat_perp={hfperp_end:+.4f} | "
            f"alpha_f^(axis)={alphaf_end:.3f}, pi-alpha={alphaf_ref_end:.3f}"
        )
        print(line)

        if len(betas_mf0) and len(betas_chi1):
            dists = [abs(float(bm) - float(bc)) for bm in betas_mf0 for bc in betas_chi1]
            print(f"    nearest |beta(m·f=0)-beta(chi=1)| = {min(dists):.4e}")
        elif len(betas_chi1):
            print("    chi=1 occurs, but no m·f=0 crossing on this beta window/grid.")
        elif len(betas_mf0):
            print("    m·f=0 occurs, but chi=1 does not occur on this beta window/grid.")
        else:
            print("    neither chi=1 nor m·f=0 occurs on this beta window/grid.")

    if crossings_rows:
        print("\nSample detected events (first 10 rows):")
        for row in crossings_rows[: min(10, len(crossings_rows))]:
            print(
                "  [{event_type}] g/g*_ref={g_over_gstar_ref:.2f} beta={beta_cross:.5f} chi={chi_cross:.3f} | "
                "m_f: {m_f_before:+.4f}->{m_f_after:+.4f}  "
                "m_f_perp: {m_f_perp_before:+.4f}->{m_f_perp_after:+.4f}  "
                "m_z: {m_z_before:+.4f}->{m_z_after:+.4f} | "
                "alpha_f jump(wrap)={angle_jump_alpha_f_wrap:+.3f}  "
                "alpha_f^(cont) jump={angle_jump_alpha_f_cont:+.3f}".format(
                    **row
                )
            )
    else:
        print("\nNo event crossings were detected on the chosen beta grid.")


if __name__ == "__main__":
    main()
