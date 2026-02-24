# -*- coding: utf-8 -*-
"""
hmf_colab_pi4_sweep.py  –  GPU ED: θ=π/4, high n_cut, bandwidth study
=======================================================================
Designed for Google Colab (T4 / A100) with CUDA-JAX.
Primary focus: n_modes=2 with the highest n_cut the GPU can handle.

HOW TO RUN ON COLAB
-------------------
1.  Runtime → Change runtime type → T4 GPU (or A100 if available)
2.  Upload this file, or paste into a code cell.
3.  First time only: uncomment the pip install line in Section 0 and run it.
4.  Edit the CONFIG section below, then Runtime → Run all.
5.  Download the CSV(s) when done.

PHYSICS
-------
  H(g) = H_static + g · H_int
  H_static = (ωq/2) σz ⊗ I_B  +  I_S ⊗ Σk ωk nk
  H_int    = X ⊗ B
  X = cos(θ) σz − sin(θ) σx  [at θ=π/4: (σz − σx)/√2,  X² = I → counterterm = scalar]
  B = Σk ck (ak + ak†),  ck = √(J(ωk) Δω),  J(ω) = Q τc ω exp(−τc ω)

  Hilbert space: qubit ⊗ mode_1 ⊗ … ⊗ mode_N,  dim = 2 × n_cut^n_modes

KEY INSIGHT — BANDWIDTH vs N_CUT
----------------------------------
  For n_modes=2 the two mode frequencies are:
      ω₁ = omega_min,   ω₂ = omega_max   (two linspace points)
  Thermal occupation of mode k:  n_k(β) = 1/(exp(β ωk) − 1)
  Convergence requires n_cut >> n_k  (otherwise high Fock states are populated).

  Consequence:
    ↓ omega_min  →  ↑ n_k(β)  →  need ↑ n_cut  →  ↑ cost
    ↑ omega_min  →  miss infrared spectral weight  →  less accurate physics

  This script runs both sweeps so you can see this tradeoff directly.

CONVERGENCE STUDY
-----------------
  Sweep A — n_cut convergence (fixed bandwidth):
    n_modes=2, n_cut = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60(T4)/90(A100)]
    Also n_modes=3 and 4 at smaller n_cut for cross-check.
    Grid: β × g  with θ=π/4.

  Sweep B — bandwidth effect (fixed n_modes=2, fixed n_cut):
    Vary (omega_min, omega_max) across several windows.
    Observe how the ED result and chi change.
"""

# ── 0. SETUP ─────────────────────────────────────────────────────────────────
# Uncomment on first Colab run:
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#                 "jax[cuda12]", "matplotlib", "pandas"], check=False)

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.linalg import eigh as scipy_eigh

# JAX — must set x64 flag BEFORE any jax.numpy usage
JAX_OK = False
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import vmap
    JAX_OK = True
    _devs   = jax.devices()
    _kinds  = [d.device_kind for d in _devs]
    _is_gpu = any("gpu" in k.lower() for k in _kinds)
    print(f"[JAX {jax.__version__}] devices: {_kinds}  |  GPU: {_is_gpu}")
except Exception as _e:
    print(f"[WARN] JAX not available ({_e}).  Falling back to SciPy on CPU.")
    _is_gpu = False

# ── Output paths ──────────────────────────────────────────────────────────────
# On Colab, /content/ is the working directory.
_here    = Path(__file__).parent if "__file__" in dir() else Path("/content")
OUT_DIR  = _here / "data"
FIG_DIR  = _here / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ╔══════════════════════════════════════════════════════╗
# ║                   CONFIG — edit here                 ║
# ╚══════════════════════════════════════════════════════╝
# =============================================================================

# ── Fixed physics ─────────────────────────────────────────────────────────────
THETA     = np.pi / 4   # do not change — the whole script assumes this
OMEGA_Q   = 2.0
Q_STRENGTH = 10.0
TAU_C     = 1.0

# ── Default spectral window ───────────────────────────────────────────────────
OMEGA_MIN_DEFAULT = 0.5
OMEGA_MAX_DEFAULT = 8.0

# ── β × g sweep grid (applies to both sweeps) ────────────────────────────────
G_VALS    = np.linspace(0.05, 1.8, 25)     # coupling strengths
BETA_VALS = np.linspace(0.3, 10.0, 30)     # inverse temperatures

# ── Sweep A: n_cut convergence (n_modes=2 focus) ─────────────────────────────
# These will be filtered by the GPU memory check at runtime.
# T4 (16 GB):  n_cut up to ~60 in float64
# A100 (80 GB): n_cut up to ~90 in float64
NCUT_2M = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]   # n_modes=2
NCUT_3M = [4, 5, 6, 8, 10, 12, 15]                                   # n_modes=3
NCUT_4M = [4, 5, 6, 8]                                                # n_modes=4

# ── Sweep B: bandwidth (n_modes=2, fixed n_cut) ───────────────────────────────
NCUT_BW_STUDY = 20    # n_cut used for the bandwidth comparison
BANDWIDTH_CONFIGS = [
    # (omega_min, omega_max, label) — keep sorted by bandwidth width
    (1.5, 2.5,  "narrow  [1.5–2.5]"),
    (1.0, 3.0,  "medium  [1.0–3.0]"),
    (0.5, 5.0,  "wide    [0.5–5.0]"),
    (0.5, 8.0,  "full    [0.5–8.0]"),    # standard
    (0.5, 12.0, "broad   [0.5–12.0]"),
    (2.5, 8.0,  "no-IR   [2.5–8.0]"),   # cut off below resonance
]

# ── Memory guard ──────────────────────────────────────────────────────────────
# JAX eigh workspace ≈ 5 × N² × bytes_per_element (complex128=16, complex64=8)
# Adjust VRAM_GB to match your Colab instance.
VRAM_GB       = 15.0   # T4 = 16 GB, set 15 to be safe; A100 = 80 GB
VRAM_FRAC     = 0.30   # use at most this fraction for the eigh solve
BYTES_PER_EL  = 16     # complex128 (float64); set to 8 for complex64 (faster, less accurate)

# =============================================================================
# 1. PHYSICS CORE
# =============================================================================

def thermal_occupation(omega: float, beta: float) -> float:
    """Bose-Einstein mean occupation: n̄ = 1/(exp(βω) − 1)."""
    x = beta * omega
    if x < 1e-10:
        return float("inf")
    if x > 100:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def vram_limit_bytes() -> float:
    """Available VRAM for the eigh solve (bytes)."""
    return VRAM_GB * 1e9 * VRAM_FRAC


def max_safe_dim() -> int:
    """Maximum Hilbert space dimension that fits in GPU memory."""
    lim = vram_limit_bytes()
    return int(np.sqrt(lim / (5.0 * BYTES_PER_EL)))


def memory_ok(dim: int) -> bool:
    return 5 * dim**2 * BYTES_PER_EL <= vram_limit_bytes()


def _annihilation(n_cut: int) -> np.ndarray:
    a = np.zeros((n_cut, n_cut), dtype=np.complex128)
    for n in range(1, n_cut):
        a[n - 1, n] = np.sqrt(float(n))
    return a


def build_operators(
    n_modes: int,
    n_cut: int,
    omega_min: float = OMEGA_MIN_DEFAULT,
    omega_max: float = OMEGA_MAX_DEFAULT,
    omega_q: float = OMEGA_Q,
    q_strength: float = Q_STRENGTH,
    tau_c: float = TAU_C,
    theta: float = THETA,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Returns (h_static, h_int, dim_b, omegas, c_k) where
        H(g) = h_static + g * h_int.

    h_static is real diagonal (stored as dense for compatibility with JAX eigh).
    Counterterm g² Q · I omitted (scalar at θ=π/4 since X²=I).
    """
    sz  = np.array([[1,  0], [ 0, -1]], dtype=np.complex128)
    sx  = np.array([[0,  1], [ 1,  0]], dtype=np.complex128)
    id2 = np.eye(2, dtype=np.complex128)

    if n_modes == 1:
        omegas  = np.array([0.5 * (omega_min + omega_max)])
        d_omega = omega_max - omega_min
    else:
        omegas  = np.linspace(omega_min, omega_max, n_modes)
        d_omega = float(omegas[1] - omegas[0])

    j_vals = q_strength * tau_c * omegas * np.exp(-tau_c * omegas)
    c_k    = np.sqrt(np.maximum(j_vals, 0.0) * d_omega)

    dim_b = n_cut ** n_modes
    id_m  = np.eye(n_cut, dtype=np.complex128)
    a_s   = _annihilation(n_cut)
    ad_s  = a_s.conj().T
    x_s   = a_s + ad_s   # position-like (a + a†)
    n_s   = ad_s @ a_s   # number operator

    h_b  = np.zeros((dim_b, dim_b), dtype=np.complex128)
    b_op = np.zeros((dim_b, dim_b), dtype=np.complex128)

    for k in range(n_modes):
        hk = np.array([[1.0 + 0j]])
        vk = np.array([[1.0 + 0j]])
        for j in range(n_modes):
            hk = np.kron(hk, n_s if j == k else id_m)
            vk = np.kron(vk, x_s if j == k else id_m)
        h_b  += omegas[k] * hk
        b_op += c_k[k]   * vk

    x_q      = np.cos(theta) * sz - np.sin(theta) * sx
    h_s      = 0.5 * omega_q * sz
    h_static = np.kron(h_s, np.eye(dim_b, dtype=np.complex128)) + np.kron(id2, h_b)
    h_int    = np.kron(x_q, b_op)

    return h_static, h_int, dim_b, omegas, c_k


# =============================================================================
# 2. ANALYTIC χ  (discrete bath, same discretisation as ED)
# =============================================================================

def compute_chi(
    g: float,
    beta: float,
    n_modes: int,
    omega_min: float = OMEGA_MIN_DEFAULT,
    omega_max: float = OMEGA_MAX_DEFAULT,
    omega_q: float = OMEGA_Q,
    q_strength: float = Q_STRENGTH,
    tau_c: float = TAU_C,
    theta: float = THETA,
    n_grid: int = 801,
) -> float:
    """
    χ(g, β) = g² √(δz₀² + σ₊₀ σ₋₀) from analytic HMF theory.
    Uses the same discrete bath as the ED for self-consistency.
    χ ≈ 1 marks the onset of the non-perturbative crossover.
    """
    if n_modes == 1:
        omegas  = np.array([0.5 * (omega_min + omega_max)])
        d_omega = omega_max - omega_min
    else:
        omegas  = np.linspace(omega_min, omega_max, n_modes)
        d_omega = float(omegas[1] - omegas[0])

    j_vals = q_strength * tau_c * omegas * np.exp(-tau_c * omegas)
    g2_k   = np.maximum(j_vals, 0.0) * d_omega

    u  = np.linspace(0.0, beta, n_grid)
    k0 = np.zeros_like(u)
    for wk, g2 in zip(omegas, g2_k):
        den = np.sinh(0.5 * beta * wk)
        k0 += g2 * (np.cosh(wk * (0.5 * beta - u)) / den
                    if abs(den) > 1e-14
                    else (2.0 / max(beta * wk, 1e-14)))

    def lap(ws: float) -> float:
        return float(np.trapezoid(k0 * np.exp(ws * u), u))

    def res(ws: float) -> float:
        return float(np.trapezoid((beta - u) * k0 * np.exp(ws * u), u))

    c, s = np.cos(theta), np.sin(theta)
    k00  = lap(0.0)
    k0p  = lap(+omega_q)
    k0m  = lap(-omega_q)
    r0p  = res(+omega_q)
    r0m  = res(-omega_q)

    sp0 = (c * s / omega_q) * ((1.0 + np.exp(+beta * omega_q)) * k00 - 2.0 * k0p)
    sm0 = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * k00 - 2.0 * k0m)
    dz0 = s**2 * 0.5 * (r0p - r0m)

    chi0 = float(np.sqrt(max(dz0**2 + sp0 * sm0, 0.0)))
    return g**2 * chi0


# =============================================================================
# 3. DIAGONALISATION + BETA SWEEP
# =============================================================================

def diag_g(
    h_static: np.ndarray,
    h_int: np.ndarray,
    g: float,
    dim_b: int,
    use_jax: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Form H(g) = H_static + g·H_int, diagonalise, compute bath-traced
    overlap vectors.  Returns (evals, ov_uu, ov_dd, ov_ud, wall_time).

    The three 1D overlap arrays are all that's needed for the beta sweep:
      ov_uu[i] = Σ_n |⟨↑,n|ψ_i⟩|²   (bath-traced |↑⟩ weight of eigenstate i)
      ov_dd[i] = Σ_n |⟨↓,n|ψ_i⟩|²
      ov_ud[i] = Σ_n ⟨↑,n|ψ_i⟩* ⟨↓,n|ψ_i⟩   (complex)

    Eigenvectors are NOT returned — freeing VRAM immediately after overlaps.
    """
    h_full = h_static + g * h_int   # still NumPy here

    t0 = time.perf_counter()

    if use_jax and JAX_OK:
        h_j   = jnp.array(h_full)
        ev, V = jnp.linalg.eigh(h_j)
        ev.block_until_ready()

        psi_up = V[:dim_b, :]                                   # (dim_b, N)
        psi_dn = V[dim_b:, :]
        ov_uu  = jnp.einsum('ij,ij->j', psi_up.conj(), psi_up).real
        ov_dd  = jnp.einsum('ij,ij->j', psi_dn.conj(), psi_dn).real
        ov_ud  = jnp.einsum('ij,ij->j', psi_up.conj(), psi_dn)
        ov_uu.block_until_ready()

        evals = np.asarray(ev,    dtype=np.float64)
        ov_uu = np.asarray(ov_uu, dtype=np.float64)
        ov_dd = np.asarray(ov_dd, dtype=np.float64)
        ov_ud = np.asarray(ov_ud, dtype=np.complex128)
        del V, psi_up, psi_dn   # free GPU memory explicitly
    else:
        evals, evecs = scipy_eigh(h_full)
        psi_up = evecs[:dim_b, :]
        psi_dn = evecs[dim_b:, :]
        ov_uu  = np.einsum('ij,ij->j', psi_up.conj(), psi_up).real
        ov_dd  = np.einsum('ij,ij->j', psi_dn.conj(), psi_dn).real
        ov_ud  = np.einsum('ij,ij->j', psi_up.conj(), psi_dn)

    dt = time.perf_counter() - t0
    return evals, ov_uu, ov_dd, ov_ud, dt


def beta_sweep(
    evals: np.ndarray,
    ov_uu: np.ndarray,
    ov_dd: np.ndarray,
    ov_ud: np.ndarray,
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised beta sweep over all temperatures at once (NumPy).
    One eigh per g value → all betas free.
    Returns p00, p11, re01, im01 each of shape (n_beta,).
    """
    shifted = evals - evals.min()
    log_w   = -np.outer(betas, shifted)           # (n_beta, N_eig)
    log_w  -= log_w.max(axis=1, keepdims=True)    # numerical stability
    w       = np.exp(log_w)
    w      /= w.sum(axis=1, keepdims=True)

    p00  = (w * ov_uu[None, :]).sum(axis=1)
    p11  = (w * ov_dd[None, :]).sum(axis=1)
    re01 = (w * ov_ud.real[None, :]).sum(axis=1)
    im01 = (w * ov_ud.imag[None, :]).sum(axis=1)
    return p00, p11, re01, im01


# =============================================================================
# 4. BANDWIDTH DIAGNOSTIC  (printed before sweep)
# =============================================================================

def bandwidth_diagnostic(
    n_modes: int,
    omega_min: float,
    omega_max: float,
    beta_vals_diag: np.ndarray,
    g_sample: float = 0.5,
) -> None:
    """
    Prints a table: for each mode in this n_modes/bandwidth config,
    show its frequency, J(ω), coupling amplitude c_k, and thermal
    occupations at representative betas.  Also shows estimated n_cut
    needed for convergence at each beta.

    This makes the bandwidth → n_cut requirement explicit.
    """
    if n_modes == 1:
        omegas  = np.array([0.5 * (omega_min + omega_max)])
        d_omega = omega_max - omega_min
    else:
        omegas  = np.linspace(omega_min, omega_max, n_modes)
        d_omega = float(omegas[1] - omegas[0])

    j_vals = Q_STRENGTH * TAU_C * omegas * np.exp(-TAU_C * omegas)
    c_k    = np.sqrt(np.maximum(j_vals, 0.0) * d_omega)

    print(f"\n{'─'*70}")
    print(f"  Bandwidth diagnostic: n_modes={n_modes}, "
          f"ω ∈ [{omega_min:.2f}, {omega_max:.2f}], Δω={d_omega:.3f}")
    print(f"{'─'*70}")

    header = f"  {'k':>3}  {'ωk':>6}  {'J(ωk)':>8}  {'ck':>8}"
    for b in beta_vals_diag:
        header += f"  {'n̄(β='+f'{b:.1f}'+')':>10}"
    header += f"  {'n_cut≥(β_min)':>14}"
    print(header)

    beta_min = float(beta_vals_diag.min())
    for k, (wk, jk, ck) in enumerate(zip(omegas, j_vals, c_k)):
        row = f"  {k:>3}  {wk:>6.3f}  {jk:>8.4f}  {ck:>8.5f}"
        n_max_req = 0
        for b in beta_vals_diag:
            n_th = thermal_occupation(wk, b)
            n_max_req = max(n_max_req, n_th)
            row += f"  {n_th:>10.2f}"
        n_cut_needed = int(np.ceil(max(3.0 * n_max_req, 4.0) + 2))
        row += f"  {n_cut_needed:>14d}"
        print(row)

    # Chi at sample g
    chi_lo = compute_chi(g_sample, float(beta_vals_diag.max()),
                         n_modes, omega_min, omega_max)
    chi_hi = compute_chi(g_sample, float(beta_vals_diag.min()),
                         n_modes, omega_min, omega_max)
    print(f"\n  χ(g={g_sample}, β={beta_vals_diag.max():.1f}) = {chi_lo:.4f}  "
          f"|  χ(g={g_sample}, β={beta_vals_diag.min():.1f}) = {chi_hi:.4f}")
    print(f"  [χ≈1 crossover at β={beta_vals_diag.max():.1f} "
          f"near g≈{1.0/np.sqrt(max(chi_lo/g_sample**2, 1e-12)):.2f}]")
    print(f"{'─'*70}\n")


# =============================================================================
# 5. SWEEP RUNNER  (shared by both sweeps)
# =============================================================================

def run_sweep(
    configs: list[tuple[int, int, float, float]],   # (n_modes, n_cut, omega_min, omega_max)
    g_vals: np.ndarray,
    beta_vals: np.ndarray,
    use_jax: bool,
    label: str = "sweep",
    out_csv: Path | None = None,
    checkpoint_every: int = 5,
) -> pd.DataFrame:
    """
    Main sweep loop.  For each (n_modes, n_cut, omega_min, omega_max):
      - Build H once
      - For each g: diagonalise, then vectorise over all β
      - Compute chi analytically at each (g, β)
      - Append to CSV incrementally (crash recovery)

    Parameters
    ----------
    configs : list of (n_modes, n_cut, omega_min, omega_max)
    checkpoint_every : flush CSV to disk every N g-values
    """
    all_rows: list[dict] = []
    n_flushed = 0

    dim_max = max_safe_dim()
    print(f"\n[{label}] GPU max safe dim: {dim_max}  "
          f"(VRAM={VRAM_GB:.0f}GB × {VRAM_FRAC:.0%}, dtype={'f64' if BYTES_PER_EL==16 else 'f32'})")

    for (nm, nc, omin, omax) in configs:
        dim = 2 * (nc ** nm)

        if dim > dim_max and use_jax:
            print(f"  [SKIP] n_modes={nm}, n_cut={nc}  dim={dim} > safe limit {dim_max}")
            continue

        t_build = time.perf_counter()
        h_static, h_int, dim_b, omegas, c_k = build_operators(
            nm, nc, omin, omax
        )
        t_build = time.perf_counter() - t_build

        tag = f"n_m={nm} n_c={nc:>3} dim={dim:>6} ω∈[{omin:.1f},{omax:.1f}]"
        print(f"\n[CONFIG] {tag}  (build: {t_build:.2f}s)")
        bandwidth_diagnostic(nm, omin, omax, BETA_VALS[:3])   # quick preview

        g_rows: list[dict] = []
        t_config = time.perf_counter()

        for ig, g in enumerate(g_vals):
            evals, ov_uu, ov_dd, ov_ud, dt_diag = diag_g(
                h_static, h_int, g, dim_b, use_jax
            )
            p00, p11, re01, im01 = beta_sweep(evals, ov_uu, ov_dd, ov_ud, beta_vals)

            # Pre-compute chi for all betas at this g (cheap, avoid repeated calls)
            chi_betas = [compute_chi(g, float(b), nm, omin, omax) for b in beta_vals]

            for ib, beta in enumerate(beta_vals):
                _re01 = float(re01[ib])
                _im01 = float(im01[ib])
                g_rows.append({
                    # --- physical parameters (everything needed for local postprocessing) ---
                    "n_modes":    nm,
                    "n_cut":      nc,
                    "dim":        dim,
                    "omega_min":  omin,
                    "omega_max":  omax,
                    "omega_q":    OMEGA_Q,
                    "q_strength": Q_STRENGTH,
                    "tau_c":      TAU_C,
                    "g":          float(g),
                    "beta":       float(beta),
                    "theta":      float(THETA),
                    # --- ED observables (full qubit density matrix) ---
                    "ed_p00":     float(p00[ib]),
                    "ed_p11":     float(p11[ib]),
                    "ed_re01":    _re01,
                    "ed_im01":    _im01,
                    "ed_coh":     2.0 * float(np.sqrt(_re01**2 + _im01**2)),
                    "ed_mz":      float(p00[ib] - p11[ib]),
                    # --- crossover parameter ---
                    "chi":        float(chi_betas[ib]),
                    # --- timing ---
                    "diag_s":     dt_diag,
                })

            # Incremental checkpoint
            if out_csv and (ig + 1) % checkpoint_every == 0:
                chunk = pd.DataFrame.from_records(all_rows + g_rows)
                chunk.to_csv(out_csv, index=False)
                print(f"  checkpoint → {out_csv.name}  "
                      f"({len(chunk)} rows)")

            print(f"  g={g:.3f}  diag={dt_diag:.3f}s  "
                  f"χ∈[{min(compute_chi(g,b,nm,omin,omax) for b in [beta_vals[0],beta_vals[-1]]):.2f},"
                  f"{max(compute_chi(g,b,nm,omin,omax) for b in [beta_vals[0],beta_vals[-1]]):.2f}]  "
                  f"mz∈[{p00.min()-p11.min():.3f},{p00.max()-p11.max():.3f}]")

        all_rows.extend(g_rows)
        print(f"  Config done in {time.perf_counter()-t_config:.1f}s")

    df = pd.DataFrame.from_records(all_rows)
    if out_csv and not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"\n[SAVED] {out_csv}  ({len(df)} rows)")
    return df


# =============================================================================
# 6. PLOTS
# =============================================================================

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "lines.linewidth": 1.2, "axes.linewidth": 0.8,
})


def plot_ncut_convergence(df: pd.DataFrame, out: Path) -> None:
    """
    Left: mz vs n_cut at the (β, g) point nearest χ=1  (for each n_modes).
    Right: heatmap of mz(β, g) for the largest (n_modes=2, n_cut).
    """
    sub2 = df[(df["n_modes"] == 2)].copy()
    if sub2.empty:
        print("[WARN] No n_modes=2 data for convergence plot.")
        return

    # Find crossover point in n_modes=2 largest n_cut
    nc_max = sub2["n_cut"].max()
    ref = sub2[sub2["n_cut"] == nc_max]
    idx = (ref["chi"] - 1.0).abs().idxmin()
    g_c, b_c, chi_c = ref.loc[idx, ["g", "beta", "chi"]]
    g_c, b_c, chi_c = float(g_c), float(b_c), float(chi_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ── Left: convergence in n_cut ────────────────────────────────────────────
    n_modes_avail = sorted(df["n_modes"].unique())
    cmap_c = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_modes_avail)))

    for nm, col in zip(n_modes_avail, cmap_c):
        sub_m = df[df["n_modes"] == nm]
        pts = []
        for nc in sorted(sub_m["n_cut"].unique()):
            rows = sub_m[
                (sub_m["n_cut"] == nc) &
                (np.isclose(sub_m["beta"], b_c, atol=0.5)) &
                (np.isclose(sub_m["g"],    g_c,  atol=0.05))
            ]
            if not rows.empty:
                pts.append((nc, float(rows["ed_mz"].iloc[0])))
        if pts:
            xc, ym = zip(*pts)
            ax1.plot(xc, ym, "o-", color=col, lw=1.1, ms=3.5,
                     label=rf"$n_m={nm}$")

    # Reference line: largest-n_cut result
    ref_mz = float(ref.loc[idx, "ed_mz"])
    ax1.axhline(ref_mz, color="gray", ls="--", lw=0.9,
                label=rf"$n_c={nc_max}$ (ref)")
    ax1.set_xlabel(r"Fock cutoff $n_\mathrm{cut}$")
    ax1.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax1.set_title(rf"Convergence at $g={g_c:.2f}$, $\beta={b_c:.2f}$ ($\chi\approx{chi_c:.2f}$)")
    ax1.legend(loc="best")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Right: landscape heatmap ───────────────────────────────────────────────
    ref_df = sub2[sub2["n_cut"] == nc_max].copy()
    g_u  = np.sort(ref_df["g"].unique())
    b_u  = np.sort(ref_df["beta"].unique())
    G, B = np.meshgrid(g_u, b_u)

    piv_mz  = ref_df.pivot_table(index="beta", columns="g", values="ed_mz",  aggfunc="mean")
    piv_chi = ref_df.pivot_table(index="beta", columns="g", values="chi", aggfunc="mean")
    mz_g  = piv_mz.reindex(index=b_u, columns=g_u).values
    chi_g = piv_chi.reindex(index=b_u, columns=g_u).values

    pcm = ax2.pcolormesh(G, B, mz_g, cmap="RdBu_r", shading="auto",
                         vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax2, pad=0.02, label=r"$\langle\sigma_z\rangle$")
    cs = ax2.contour(G, B, chi_g, levels=[0.5, 1.0, 2.0],
                     colors=["gold", "white", "cyan"], linewidths=0.9)
    ax2.clabel(cs, fmt={0.5: r"$\chi\!=\!0.5$", 1.0: r"$\chi\!=\!1$",
                        2.0: r"$\chi\!=\!2$"}, fontsize=6, inline=True)
    ax2.set_xlabel(r"$g$")
    ax2.set_ylabel(r"$\beta$")
    ax2.set_title(rf"$n_m=2$, $n_c={nc_max}$, $\theta=\pi/4$")
    ax2.scatter([g_c], [b_c], color="lime", s=30, zorder=5,
                label="crossover pt")
    ax2.legend(fontsize=6)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
        print(f"  {out.with_suffix(ext).name}")
    plt.close(fig)


def plot_bandwidth_study(df_bw: pd.DataFrame, out: Path) -> None:
    """
    For the bandwidth sweep:
      Left: mz vs g at fixed β (mid-range) for each bandwidth config.
      Right: chi vs g for the same slice.
    Shows how bandwidth choice affects the ED result.
    """
    if df_bw.empty:
        print("[WARN] Empty bandwidth DataFrame — skipping plot.")
        return

    beta_mid = float(np.median(df_bw["beta"].unique()))
    # pick the beta closest to median
    beta_use = float(df_bw["beta"].unique()[
        np.abs(df_bw["beta"].unique() - beta_mid).argmin()])

    sub = df_bw[np.isclose(df_bw["beta"], beta_use, atol=0.3)].copy()

    bw_groups = sorted(sub.groupby(["omega_min", "omega_max"]).groups.keys())
    cmap_bw = plt.cm.plasma(np.linspace(0.05, 0.95, len(bw_groups)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for (omin, omax), col in zip(bw_groups, cmap_bw):
        rows = sub[
            np.isclose(sub["omega_min"], omin, atol=0.01) &
            np.isclose(sub["omega_max"], omax, atol=0.01)
        ].sort_values("g")
        if rows.empty:
            continue
        lbl = rf"[{omin:.1f},{omax:.1f}]"
        ax1.plot(rows["g"], rows["ed_mz"],  color=col, lw=1.1, label=lbl)
        ax2.plot(rows["g"], rows["chi"], color=col, lw=1.1, label=lbl)

    ax2.axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\chi=1$")
    for ax in (ax1, ax2):
        ax.set_xlabel(r"$g$")
        ax.legend(fontsize=6, loc="best", title="[ωmin, ωmax]",
                  title_fontsize=6)
    ax1.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax2.set_ylabel(r"$\chi(g,\beta)$")
    ax1.set_title(rf"$\beta={beta_use:.1f}$, $n_m=2$, $n_c={NCUT_BW_STUDY}$")
    ax2.set_title("Crossover parameter")

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
        print(f"  {out.with_suffix(ext).name}")
    plt.close(fig)


def plot_ncut_needed_vs_bandwidth(out: Path) -> None:
    """
    Standalone diagnostic plot: required n_cut vs omega_min,
    for several beta values.  No ED needed — purely from thermal occupation.

    Shows the bandwidth–performance tradeoff without running a sweep.
    """
    omega_min_vals = np.linspace(0.2, 4.0, 80)
    beta_show = [0.5, 1.0, 2.0, 5.0, 10.0]
    cmap_b = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(beta_show)))

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for beta, col in zip(beta_show, cmap_b):
        n_req = []
        for omin in omega_min_vals:
            n_th = thermal_occupation(omin, beta)
            n_req.append(int(np.ceil(3.0 * n_th + 3)))
        ax.semilogy(omega_min_vals, n_req, color=col, lw=1.2,
                    label=rf"$\beta={beta:.1f}$")

    ax.axvline(OMEGA_Q, color="k", ls="--", lw=0.8, label=r"$\omega_q$")
    ax.set_xlabel(r"Lowest mode frequency $\omega_\mathrm{min}$")
    ax.set_ylabel(r"Required $n_\mathrm{cut}$ (estimate)")
    ax.set_title("Bandwidth–convergence tradeoff (n_modes=2, lower mode)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(1, 1e4)

    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
        print(f"  {out.with_suffix(ext).name}")
    plt.close(fig)


# =============================================================================
# 7. MAIN
# =============================================================================

def main() -> None:
    use_jax = JAX_OK

    print("\n" + "="*65)
    print("  HMF Colab ED  |  θ = π/4 (fixed)  |  n_modes=2 focus")
    print(f"  Backend : {'JAX/GPU' if (use_jax and _is_gpu) else 'SciPy CPU'}")
    print(f"  g   : [{G_VALS[0]:.2f}, {G_VALS[-1]:.2f}]  ({len(G_VALS)} pts)")
    print(f"  β   : [{BETA_VALS[0]:.2f}, {BETA_VALS[-1]:.2f}]  ({len(BETA_VALS)} pts)")
    print("="*65)

    # ── Bandwidth-convergence diagnostic plot (no ED needed) ──────────────────
    print("\n[PLOT] Bandwidth–n_cut tradeoff diagnostic...")
    plot_ncut_needed_vs_bandwidth(FIG_DIR / "hmf_pi4_bw_tradeoff")

    # ── Sweep A: n_cut convergence (default bandwidth) ────────────────────────
    configs_A: list[tuple[int, int, float, float]] = []
    omin_d, omax_d = OMEGA_MIN_DEFAULT, OMEGA_MAX_DEFAULT

    for nc in NCUT_2M:
        configs_A.append((2, nc, omin_d, omax_d))
    for nc in NCUT_3M:
        configs_A.append((3, nc, omin_d, omax_d))
    for nc in NCUT_4M:
        configs_A.append((4, nc, omin_d, omax_d))

    csv_A = OUT_DIR / "pi4_ncut_convergence.csv"
    print(f"\n{'─'*65}")
    print(f"  SWEEP A: n_cut convergence  ({len(configs_A)} configs)")
    print(f"{'─'*65}")
    df_A = run_sweep(configs_A, G_VALS, BETA_VALS, use_jax,
                     label="SweepA-ncut", out_csv=csv_A)

    if not df_A.empty:
        print("\n[PLOT] n_cut convergence + landscape...")
        plot_ncut_convergence(df_A, FIG_DIR / "hmf_pi4_ncut_convergence")

    # ── Sweep B: bandwidth study (n_modes=2, fixed n_cut) ─────────────────────
    configs_B: list[tuple[int, int, float, float]] = [
        (2, NCUT_BW_STUDY, omin, omax)
        for (omin, omax, _) in BANDWIDTH_CONFIGS
    ]

    csv_B = OUT_DIR / "pi4_bandwidth_study.csv"
    print(f"\n{'─'*65}")
    print(f"  SWEEP B: bandwidth study  ({len(configs_B)} configs, n_cut={NCUT_BW_STUDY})")
    print(f"{'─'*65}")
    df_B = run_sweep(configs_B, G_VALS, BETA_VALS, use_jax,
                     label="SweepB-bandwidth", out_csv=csv_B)

    if not df_B.empty:
        print("\n[PLOT] Bandwidth effect...")
        plot_bandwidth_study(df_B, FIG_DIR / "hmf_pi4_bandwidth_study")

    print("\n" + "="*65)
    print(f"  Sweep A  : {len(df_A)} rows  →  {csv_A.name}")
    print(f"  Sweep B  : {len(df_B)} rows  →  {csv_B.name}")
    print(f"  Figures  : {FIG_DIR}")
    print("="*65)
    print("\nDone.  Download CSVs from Files panel on the left.")


if __name__ == "__main__":
    main()
