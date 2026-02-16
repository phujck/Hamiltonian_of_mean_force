"""
Stochastic auxiliary-field simulation for all-to-all 2-local targets.

We simulate an HS-decoupled free-spin trajectory model where each trajectory is
a product of single-site imaginary-time propagators. Interactions are induced by
averaging over shared Gaussian fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Tuple

import numpy as np
import psutil

from designability_alltoall_models import AllToAllModel, model_to_term_arrays


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)
SIGMAS = (SIGMA_X, SIGMA_Y, SIGMA_Z)


@dataclass
class StochasticResult:
    method: str
    runtime_s: float
    peak_mem_mb: float
    energy_density: float
    stderr_energy_density: float
    mx: float
    my: float
    mz: float
    stderr_mx: float
    stderr_my: float
    stderr_mz: float
    two_body: Dict[str, float]
    stderr_two_body: Dict[str, float]
    effective_samples: float
    mean_weight: float


def _exp_pauli(vec: np.ndarray) -> np.ndarray:
    """Return exp(vec . sigma) for real vec in R^3."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-14:
        return IDENTITY_2 + vec[0] * SIGMA_X + vec[1] * SIGMA_Y + vec[2] * SIGMA_Z
    pref = np.sinh(norm) / norm
    return np.cosh(norm) * IDENTITY_2 + pref * (
        vec[0] * SIGMA_X + vec[1] * SIGMA_Y + vec[2] * SIGMA_Z
    )


def _weighted_mean_stderr(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    w_raw = np.asarray(weights, dtype=float)
    scale = float(np.max(np.abs(w_raw))) if w_raw.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    w = w_raw / scale
    x = np.asarray(values, dtype=float)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return float(np.mean(x)), float(np.std(x) / np.sqrt(max(len(x), 1))), 0.0

    mean = float(np.sum(w * x) / w_sum)
    centered = x - mean
    var = float(np.sum(w * centered * centered) / w_sum)
    w_sq_sum = float(np.sum(w * w))
    if w_sq_sum <= 1e-18:
        neff = float(len(x))
    else:
        neff = (w_sum * w_sum) / w_sq_sum
    stderr = float(np.sqrt(max(var, 0.0) / max(neff, 1.0)))
    return mean, stderr, neff


def _complex_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    den = np.sum(weights)
    if abs(den) < 1e-18:
        return float(np.mean(np.real(values)))
    num = np.sum(weights * values)
    return float(np.real(num / den))


def _sample_one_trajectory(
    model: AllToAllModel,
    tau_steps: int,
    rng: np.random.Generator,
    term_arrays: Dict[str, np.ndarray],
) -> Tuple[complex, complex, np.ndarray]:
    """
    Returns:
      weight, energy_density, local_magnetizations (n_sites, 3)
    """
    n = model.n_sites
    dtau = model.beta / tau_steps

    i_idx = term_arrays["i_idx"]
    j_idx = term_arrays["j_idx"]
    a_idx = term_arrays["a_idx"]
    b_idx = term_arrays["b_idx"]
    j_val = term_arrays["j_val"]
    j_sign = term_arrays["j_sign"]

    # sqrt(dt * |J|) pre-factor from HS decoupling
    hs_pref = np.sqrt(np.abs(j_val) * dtau)

    # Deterministic free-spin half-step fields.
    h_half = np.empty((n, 2, 2), dtype=complex)
    for site in range(n):
        h_vec = np.array([-0.5 * dtau * model.omega_x[site], 0.0, -0.5 * dtau * model.omega_z[site]])
        h_half[site] = _exp_pauli(h_vec)

    m_local = np.repeat(IDENTITY_2[np.newaxis, :, :], n, axis=0)

    n_terms = len(j_val)
    for _ in range(tau_steps):
        xi = rng.normal(loc=0.0, scale=1.0, size=n_terms)

        # Strang local half-step.
        for site in range(n):
            m_local[site] = h_half[site] @ m_local[site]

        # Exact per-term update order within each slice.
        for t in range(n_terms):
            x = hs_pref[t] * xi[t]
            if x == 0.0:
                continue
            i = i_idx[t]
            j = j_idx[t]
            ai = a_idx[t]
            bj = b_idx[t]
            xj = -j_sign[t] * x

            exp_i = np.cosh(x) * IDENTITY_2 + np.sinh(x) * SIGMAS[ai]
            exp_j = np.cosh(xj) * IDENTITY_2 + np.sinh(xj) * SIGMAS[bj]
            m_local[i] = exp_i @ m_local[i]
            m_local[j] = exp_j @ m_local[j]

        # Strang local half-step.
        for site in range(n):
            m_local[site] = h_half[site] @ m_local[site]

    z_local = np.trace(m_local, axis1=1, axis2=2)
    weight = np.prod(z_local)

    # Local Bloch components per site in each trajectory.
    r = np.empty((n, 3), dtype=complex)
    for site in range(n):
        denom = z_local[site]
        if abs(denom) < 1e-14:
            r[site] = 0.0
            continue
        for a, sigma in enumerate(SIGMAS):
            val = np.trace(m_local[site] @ sigma) / denom
            r[site, a] = val

    # Energy density for this trajectory under the target model.
    e_loc = np.dot(model.omega_x, r[:, 0]) + np.dot(model.omega_z, r[:, 2])
    e_pair = 0.0
    for t, j in enumerate(j_val):
        e_pair += j * r[i_idx[t], a_idx[t]] * r[j_idx[t], b_idx[t]]
    energy_density = (e_loc + e_pair) / n

    return weight, energy_density, r


def run_stochastic(
    model: AllToAllModel,
    tau_steps: int,
    samples: int,
    seed: int,
) -> StochasticResult:
    rng = np.random.default_rng(seed)
    term_arrays = model_to_term_arrays(model)
    process = psutil.Process()
    mem0 = process.memory_info().rss
    t0 = perf_counter()

    weights = np.empty(samples, dtype=complex)
    energies = np.empty(samples, dtype=complex)
    mags = np.empty((samples, 3), dtype=complex)
    two_body_vals = {f"{a}{b}": np.empty(samples, dtype=complex) for a in "xyz" for b in "xyz"}

    n = model.n_sites
    n_pairs = n * (n - 1) // 2

    for s in range(samples):
        w_complex, e_s, r = _sample_one_trajectory(model, tau_steps=tau_steps, rng=rng, term_arrays=term_arrays)
        w = w_complex
        weights[s] = w
        energies[s] = e_s
        mags[s, 0] = np.mean(r[:, 0])
        mags[s, 1] = np.mean(r[:, 1])
        mags[s, 2] = np.mean(r[:, 2])

        # All-to-all two-body averages for each Pauli pair.
        for a, a_label in enumerate("xyz"):
            ra = r[:, a]
            for b, b_label in enumerate("xyz"):
                rb = r[:, b]
                corr_mat = np.outer(ra, rb)
                val = np.sum(np.triu(corr_mat, 1)) / max(n_pairs, 1)
                two_body_vals[f"{a_label}{b_label}"][s] = val

    runtime_s = perf_counter() - t0
    mem1 = process.memory_info().rss
    peak_mem_mb = max(mem0, mem1) / (1024.0 * 1024.0)

    abs_w = np.abs(weights)
    energy_mean = _complex_weighted_mean(energies, weights)
    mx = _complex_weighted_mean(mags[:, 0], weights)
    my = _complex_weighted_mean(mags[:, 1], weights)
    mz = _complex_weighted_mean(mags[:, 2], weights)
    _, energy_stderr, neff = _weighted_mean_stderr(np.real(energies), abs_w)
    _, mx_se, _ = _weighted_mean_stderr(np.real(mags[:, 0]), abs_w)
    _, my_se, _ = _weighted_mean_stderr(np.real(mags[:, 1]), abs_w)
    _, mz_se, _ = _weighted_mean_stderr(np.real(mags[:, 2]), abs_w)

    two_body_mean: Dict[str, float] = {}
    two_body_stderr: Dict[str, float] = {}
    for key, values in two_body_vals.items():
        two_body_mean[key] = _complex_weighted_mean(values, weights)
        _, se_k, _ = _weighted_mean_stderr(np.real(values), abs_w)
        two_body_stderr[key] = se_k

    return StochasticResult(
        method="stochastic",
        runtime_s=float(runtime_s),
        peak_mem_mb=float(peak_mem_mb),
        energy_density=float(energy_mean),
        stderr_energy_density=float(energy_stderr),
        mx=float(mx),
        my=float(my),
        mz=float(mz),
        stderr_mx=float(mx_se),
        stderr_my=float(my_se),
        stderr_mz=float(mz_se),
        two_body=two_body_mean,
        stderr_two_body=two_body_stderr,
        effective_samples=float(neff),
        mean_weight=float(np.real(np.mean(weights))),
    )
