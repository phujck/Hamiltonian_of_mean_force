"""
Analytic bridge for PRL 127, 250601 qubit limits.

This benchmark compares exact finite-bath reduced states against:
1) direct-kernel Gaussian qubit state from Eq. (76)-style Delta,
2) finite-coupling ordered-Gaussian HMF model from deterministic quadrature,
3) ultrastrong projected qubit state (PRL Eq. (7)/(8)).
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh, expm

from prl127_qubit_benchmark import (
    BenchmarkConfig,
    IDENTITY_2,
    SIGMA_X,
    SIGMA_Z,
    bare_gibbs_state,
    build_static_operators,
    l1_coherence_in_basis,
    partial_trace_bath,
    thermal_state,
    trace_distance,
    ultrastrong_projected_state,
)

SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


@dataclass
class OrderedQuadratureContext:
    dt: float
    x_grid: np.ndarray
    kl_basis: np.ndarray
    eta_nodes: np.ndarray
    weights: np.ndarray
    left_factor: np.ndarray
    n_time_slices: int
    kl_rank: int
    gh_order: int
    n_nodes: int


def _project_density(rho: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (rho + rho.conj().T)
    evals, evecs = eigh(hermitian)
    evals = np.clip(np.real(evals), 0.0, None)
    if float(np.sum(evals)) <= 0.0:
        return 0.5 * IDENTITY_2
    rho_psd = (evecs * evals) @ evecs.conj().T
    rho_psd /= np.trace(rho_psd)
    return 0.5 * (rho_psd + rho_psd.conj().T)


def _bloch_components(rho: np.ndarray) -> Tuple[float, float, float]:
    x = float(np.real(np.trace(rho @ SIGMA_X)))
    y = float(np.real(np.trace(rho @ SIGMA_Y)))
    z = float(np.real(np.trace(rho @ SIGMA_Z)))
    return x, y, z


def _discrete_bath(config: BenchmarkConfig) -> Tuple[np.ndarray, float, np.ndarray]:
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)], dtype=float)
        delta_omega = float(config.omega_max - config.omega_min)
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes, dtype=float)
        delta_omega = float(omegas[1] - omegas[0])
    j0 = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    return omegas, delta_omega, j0


def _kernel_profile(config: BenchmarkConfig, u_abs: np.ndarray) -> np.ndarray:
    """Discrete-bath imaginary-time kernel profile K_0(u), with u in [0, beta]."""
    beta = config.beta
    omegas, delta_omega, j0 = _discrete_bath(config)
    g2 = np.maximum(j0, 0.0) * delta_omega

    kernel = np.zeros_like(u_abs, dtype=float)
    for omega_k, g2_k in zip(omegas, g2):
        denom = np.sinh(0.5 * beta * omega_k)
        if abs(denom) < 1e-14:
            # omega -> 0 limit of cosh(omega*(beta/2-u))/sinh(beta*omega/2)
            kernel += (2.0 * g2_k) / max(beta * omega_k, 1e-14)
        else:
            kernel += g2_k * np.cosh(omega_k * (0.5 * beta - u_abs)) / denom
    return kernel


def _x_tilde_operator(config: BenchmarkConfig, tau: float) -> np.ndarray:
    return np.cos(config.theta) * SIGMA_Z - np.sin(config.theta) * (
        np.cosh(config.omega_q * tau) * SIGMA_X + 1.0j * np.sinh(config.omega_q * tau) * SIGMA_Y
    )


def _build_ordered_quadrature_context(
    config: BenchmarkConfig,
    n_time_slices: int,
    kl_rank: int,
    gh_order: int,
    max_nodes: int,
) -> OrderedQuadratureContext:
    if n_time_slices < 2:
        raise ValueError("ordered-time-slices must be >= 2.")
    if kl_rank < 0:
        raise ValueError("ordered-kl-rank must be >= 0.")
    if gh_order < 2:
        raise ValueError("ordered-gh-order must be >= 2.")

    t = np.linspace(0.0, config.beta, int(n_time_slices), dtype=float)
    dt = float(t[1] - t[0])
    x_grid = np.asarray([_x_tilde_operator(config, tau) for tau in t], dtype=complex)

    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, config.beta - u)
    cov = _kernel_profile(config, u)

    evals, evecs = eigh(cov)
    idx_desc = np.argsort(evals)[::-1]
    evals = np.clip(evals[idx_desc], 0.0, None)
    evecs = evecs[:, idx_desc]
    positive = evals > 1e-14
    effective_rank = int(min(max(0, kl_rank), int(np.count_nonzero(positive))))

    if effective_rank > 0:
        kl_basis = evecs[:, :effective_rank] * np.sqrt(evals[:effective_rank])
    else:
        kl_basis = np.zeros((len(t), 0), dtype=float)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(int(gh_order))
    eta_1d = np.sqrt(2.0) * gh_x
    weight_1d = gh_w / np.sqrt(np.pi)

    n_nodes = int(gh_order**effective_rank) if effective_rank > 0 else 1
    if n_nodes > max_nodes:
        raise ValueError(
            f"ordered quadrature requires {n_nodes} nodes (> max {max_nodes}). "
            "Reduce ordered-kl-rank / ordered-gh-order."
        )

    eta_nodes = np.zeros((n_nodes, effective_rank), dtype=float)
    weights = np.zeros(n_nodes, dtype=float)
    if effective_rank == 0:
        weights[0] = 1.0
    else:
        for row, inds in enumerate(itertools.product(range(gh_order), repeat=effective_rank)):
            ind_arr = np.asarray(inds, dtype=int)
            eta_nodes[row, :] = eta_1d[ind_arr]
            weights[row] = float(np.prod(weight_1d[ind_arr]))

    h_s = 0.5 * config.omega_q * SIGMA_Z
    left_factor = _scaled_expm_general(-config.beta * h_s)
    return OrderedQuadratureContext(
        dt=dt,
        x_grid=x_grid,
        kl_basis=kl_basis,
        eta_nodes=eta_nodes,
        weights=weights,
        left_factor=left_factor,
        n_time_slices=int(n_time_slices),
        kl_rank=effective_rank,
        gh_order=int(gh_order),
        n_nodes=n_nodes,
    )


def compute_delta_coefficients(config: BenchmarkConfig, n_kernel_grid: int) -> Tuple[float, float, float, float]:
    r"""
    Direct Eq. (76)-style evaluation using the time-ordered bilinear:
      Delta = (1/2) * \int\int K(tau-tau') T[X~(tau) X~(tau')] dtau dtau'
    for the PRL qubit parametrization.
    """
    if abs(config.omega_q) < 1e-14:
        raise ValueError("omega_q must be nonzero for the qubit bridge benchmark.")

    sin_theta = np.sin(config.theta)
    cos_theta = np.cos(config.theta)
    omega_q = config.omega_q
    beta = config.beta
    n_points = max(401, int(n_kernel_grid))
    if n_points % 2 == 0:
        n_points += 1
    u = np.linspace(0.0, beta, n_points)
    kernel_u = _kernel_profile(config, u)
    weight_u = beta - u

    # Identity sector (even in tau-tau').
    alpha_factor = (cos_theta**2) + (sin_theta**2) * np.cosh(omega_q * u)
    alpha0 = float(np.trapezoid(weight_u * kernel_u * alpha_factor, u))

    # Time-ordering generates sign(u) * commutator channels.
    delta_z = float(
        (sin_theta**2) * np.trapezoid(weight_u * kernel_u * np.sinh(omega_q * u), u)
    )

    x_factor = np.cosh(beta * omega_q) + 1.0 - np.cosh(omega_q * u) - np.cosh(omega_q * (beta - u))
    delta_x = float((sin_theta * cos_theta / omega_q) * np.trapezoid(kernel_u * x_factor, u))

    y_factor = np.sinh(beta * omega_q) - np.sinh(omega_q * u) - np.sinh(omega_q * (beta - u))
    delta_y = float((sin_theta * cos_theta / omega_q) * np.trapezoid(kernel_u * y_factor, u))

    return alpha0, delta_x, delta_y, delta_z


def weak_state_from_delta(config: BenchmarkConfig, coupling_lambda: float, delta_x: float, delta_z: float) -> np.ndarray:
    tau_s = bare_gibbs_state(0.5 * config.omega_q * SIGMA_Z, config.beta)
    t_beta = np.tanh(0.5 * config.beta * config.omega_q)
    correction = 0.5 * delta_x * SIGMA_X
    correction += 0.5 * (1.0 - t_beta**2) * delta_z * SIGMA_Z
    rho = tau_s + (coupling_lambda**2) * correction
    return _project_density(rho)


def _scaled_expm_general(matrix: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvals(matrix)
    shift = float(np.max(np.real(eigvals)))
    return expm(matrix - shift * IDENTITY_2)


def finite_hmf_collapsed_product_state(
    config: BenchmarkConfig,
    coupling_lambda: float,
    delta_x: float,
    delta_y: float,
    delta_z: float,
) -> np.ndarray:
    h_s = 0.5 * config.omega_q * SIGMA_Z
    delta_op = (coupling_lambda**2) * (delta_x * SIGMA_X + 1.0j * delta_y * SIGMA_Y + delta_z * SIGMA_Z)

    # Noncommuting qubit prediction: evaluate the product form directly,
    # without collapsing to H_S - Delta/beta.
    left = _scaled_expm_general(-config.beta * h_s)
    right = _scaled_expm_general(delta_op)
    rho_bar = left @ right
    return _project_density(rho_bar)


def finite_hmf_ordered_gaussian_state(
    coupling_lambda: float,
    ordered_ctx: OrderedQuadratureContext,
) -> np.ndarray:
    w_avg = np.zeros((2, 2), dtype=complex)
    dt = ordered_ctx.dt
    x_grid = ordered_ctx.x_grid
    kl_basis = ordered_ctx.kl_basis

    for eta, weight in zip(ordered_ctx.eta_nodes, ordered_ctx.weights):
        if kl_basis.shape[1] == 0:
            xi = np.zeros(ordered_ctx.n_time_slices, dtype=float)
        else:
            xi = coupling_lambda * (kl_basis @ eta)

        u_op = IDENTITY_2.copy()
        for n in range(ordered_ctx.n_time_slices):
            v = dt * xi[n]
            step = np.cosh(v) * IDENTITY_2 - np.sinh(v) * x_grid[n]
            u_op = step @ u_op
        w_avg += weight * u_op

    rho_bar = ordered_ctx.left_factor @ w_avg
    return _project_density(rho_bar)


def ultrastrong_closed_qubit_state(config: BenchmarkConfig) -> np.ndarray:
    x_op = np.cos(config.theta) * SIGMA_Z - np.sin(config.theta) * SIGMA_X
    arg = 0.5 * config.beta * config.omega_q * np.cos(config.theta)
    rho = 0.5 * (IDENTITY_2 - x_op * np.tanh(arg))
    return _project_density(rho)


def verify_eq8_identity() -> float:
    betas = [0.5, 1.0, 2.0]
    omegas = [2.0, 3.0, 4.0]
    thetas = np.linspace(0.0, 0.5 * np.pi, 7)
    max_abs_diff = 0.0

    for beta in betas:
        for omega_q in omegas:
            h_s = 0.5 * omega_q * SIGMA_Z
            for theta in thetas:
                x_op = np.cos(theta) * SIGMA_Z - np.sin(theta) * SIGMA_X
                rho_projector = ultrastrong_projected_state(h_s, x_op, beta)
                arg = 0.5 * beta * omega_q * np.cos(theta)
                rho_closed = 0.5 * (IDENTITY_2 - x_op * np.tanh(arg))
                diff = float(np.max(np.abs(rho_projector - rho_closed)))
                if diff > max_abs_diff:
                    max_abs_diff = diff
    return max_abs_diff


def _record_state(prefix: str, rho: np.ndarray, hs_basis: np.ndarray, record: Dict[str, float]) -> None:
    bx, by, bz = _bloch_components(rho)
    record[f"{prefix}_bloch_x"] = bx
    record[f"{prefix}_bloch_y"] = by
    record[f"{prefix}_bloch_z"] = bz
    record[f"{prefix}_coherence_hs"] = l1_coherence_in_basis(rho, hs_basis)


def run_scan(
    config: BenchmarkConfig,
    n_kernel_grid: int,
    ordered_time_slices: int,
    ordered_kl_rank: int,
    ordered_gh_order: int,
    ordered_max_nodes: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)
    dim_system = h_s.shape[0]
    dim_bath = h_static.shape[0] // dim_system

    tau_s = bare_gibbs_state(h_s, config.beta)
    rho_us_projector = ultrastrong_projected_state(h_s, x_op, config.beta)
    rho_us_closed = ultrastrong_closed_qubit_state(config)
    rho_us_diff = float(np.max(np.abs(rho_us_projector - rho_us_closed)))

    _, hs_basis = eigh(h_s)
    alpha0, delta_x, delta_y, delta_z = compute_delta_coefficients(config, n_kernel_grid)
    ordered_ctx = _build_ordered_quadrature_context(
        config,
        n_time_slices=ordered_time_slices,
        kl_rank=ordered_kl_rank,
        gh_order=ordered_gh_order,
        max_nodes=ordered_max_nodes,
    )

    lambda_values = np.linspace(config.lambda_min, config.lambda_max, config.lambda_points)
    records = []

    for lam in lambda_values:
        h_tot = h_static + lam * h_int + (lam**2) * h_counter
        rho_tot = thermal_state(h_tot, config.beta)
        rho_exact = partial_trace_bath(rho_tot, dim_system=dim_system, dim_bath=dim_bath)

        rho_weak = weak_state_from_delta(config, lam, delta_x, delta_z)
        rho_hmf = finite_hmf_ordered_gaussian_state(lam, ordered_ctx)
        rho_hmf_collapsed = finite_hmf_collapsed_product_state(config, lam, delta_x, delta_y, delta_z)

        record: Dict[str, float] = {
            "lambda": float(lam),
            "d_exact_tau": trace_distance(rho_exact, tau_s),
            "d_exact_weak": trace_distance(rho_exact, rho_weak),
            "d_exact_hmf": trace_distance(rho_exact, rho_hmf),
            "d_exact_hmf_collapsed": trace_distance(rho_exact, rho_hmf_collapsed),
            "d_exact_us": trace_distance(rho_exact, rho_us_closed),
            "d_weak_tau": trace_distance(rho_weak, tau_s),
            "d_us_projector_vs_closed": rho_us_diff,
            "alpha0": alpha0,
            "delta_x0": delta_x,
            "delta_y0": delta_y,
            "delta_z0": delta_z,
        }
        _record_state("exact", rho_exact, hs_basis, record)
        _record_state("tau", tau_s, hs_basis, record)
        _record_state("weak", rho_weak, hs_basis, record)
        _record_state("hmf", rho_hmf, hs_basis, record)
        _record_state("hmf_collapsed", rho_hmf_collapsed, hs_basis, record)
        _record_state("us", rho_us_closed, hs_basis, record)
        records.append(record)

    df = pd.DataFrame.from_records(records).sort_values("lambda").reset_index(drop=True)
    summary = build_summary(
        df,
        config,
        n_kernel_grid,
        alpha0,
        delta_x,
        delta_y,
        delta_z,
        rho_us_diff,
        ordered_ctx,
    )
    return df, summary


def build_summary(
    df: pd.DataFrame,
    config: BenchmarkConfig,
    n_kernel_grid: int,
    alpha0: float,
    delta_x: float,
    delta_y: float,
    delta_z: float,
    rho_us_diff: float,
    ordered_ctx: OrderedQuadratureContext,
) -> Dict[str, float]:
    row0 = df.iloc[0]
    row_last = df.iloc[-1]

    weak_window = df[df["lambda"] > 0.0].head(min(6, len(df)))
    weak_window_exact = weak_window[weak_window["d_exact_tau"] > 1e-14]

    weak_exact_slope = float("nan")
    if len(weak_window_exact) >= 3:
        weak_exact_slope = float(
            np.polyfit(np.log(weak_window_exact["lambda"]), np.log(weak_window_exact["d_exact_tau"]), 1)[0]
        )

    weak_model_slope = float("nan")
    weak_ratio_rel_spread = float("nan")
    small_lambdas = np.array([1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3, 1.6e-2], dtype=float)
    weak_dists = []
    tau_ref = bare_gibbs_state(0.5 * config.omega_q * SIGMA_Z, config.beta)
    for lam_small in small_lambdas:
        rho_small = weak_state_from_delta(config, lam_small, delta_x, delta_z)
        weak_dists.append(trace_distance(rho_small, tau_ref))
    weak_dists = np.asarray(weak_dists, dtype=float)
    valid = weak_dists > 1e-12
    if np.count_nonzero(valid) >= 3:
        weak_model_slope = float(np.polyfit(np.log(small_lambdas[valid]), np.log(weak_dists[valid]), 1)[0])
        ratios = weak_dists[valid] / (small_lambdas[valid] ** 2)
        mean_ratio = float(np.mean(ratios))
        if mean_ratio > 1e-16:
            weak_ratio_rel_spread = float((np.max(ratios) - np.min(ratios)) / mean_ratio)
    elif np.all(weak_dists <= 1e-18):
        weak_model_slope = 0.0
        weak_ratio_rel_spread = 0.0

    theta0_cfg = replace(config, theta=0.0)
    _, theta0_dx, theta0_dy, theta0_dz = compute_delta_coefficients(theta0_cfg, n_kernel_grid)
    theta0_weak = weak_state_from_delta(theta0_cfg, coupling_lambda=1.0, delta_x=theta0_dx, delta_z=theta0_dz)
    _, hs_basis = eigh(0.5 * config.omega_q * SIGMA_Z)
    theta0_coh = l1_coherence_in_basis(theta0_weak, hs_basis)

    theta90_cfg = replace(config, theta=0.5 * np.pi)
    rho_us_pi2 = ultrastrong_closed_qubit_state(theta90_cfg)
    theta_pi2_mixed_max_abs = float(np.max(np.abs(rho_us_pi2 - 0.5 * IDENTITY_2)))

    eq8_identity_global_max_abs = verify_eq8_identity()

    summary = {
        "beta": config.beta,
        "omega_q": config.omega_q,
        "theta": config.theta,
        "n_kernel_grid": n_kernel_grid,
        "ordered_time_slices": ordered_ctx.n_time_slices,
        "ordered_kl_rank": ordered_ctx.kl_rank,
        "ordered_gh_order": ordered_ctx.gh_order,
        "ordered_nodes": ordered_ctx.n_nodes,
        "alpha0": alpha0,
        "delta_x0": delta_x,
        "delta_y0": delta_y,
        "delta_z0": delta_z,
        "eq8_identity_max_abs_global": eq8_identity_global_max_abs,
        "eq8_identity_max_abs_current_theta": rho_us_diff,
        "weak_lambda2_slope_model": weak_model_slope,
        "weak_lambda2_slope_exact": weak_exact_slope,
        "weak_lambda2_ratio_rel_spread": weak_ratio_rel_spread,
        "ultrastrong_distance_lambda_min": float(row0["d_exact_us"]),
        "ultrastrong_distance_lambda_max": float(row_last["d_exact_us"]),
        "ultrastrong_distance_delta": float(row0["d_exact_us"] - row_last["d_exact_us"]),
        "ordered_distance_lambda_max": float(row_last["d_exact_hmf"]),
        "collapsed_distance_lambda_max": float(row_last["d_exact_hmf_collapsed"]),
        "theta0_delta_x_abs": abs(theta0_dx),
        "theta0_delta_y_abs": abs(theta0_dy),
        "theta0_weak_coherence_hs": float(theta0_coh),
        "theta_pi2_us_mixed_max_abs": theta_pi2_mixed_max_abs,
        "weak_equals_tau_max_dist": float(np.max(np.abs(df["d_weak_tau"]))),
        "pass_direct_noncommuting_channel": float(abs(delta_x) > 1e-8),
        "pass_theta0_delta_channels_off": float(abs(theta0_dx) < 1e-10 and abs(theta0_dy) < 1e-10),
        "pass_ordered_beats_collapsed_lambda_max": float(row_last["d_exact_hmf"] < row_last["d_exact_hmf_collapsed"]),
        "pass_eq8_identity": float(eq8_identity_global_max_abs < 1e-12),
        "pass_weak_lambda2_scaling": float(1.5 <= weak_model_slope <= 2.5) if np.isfinite(weak_model_slope) else 0.0,
        "pass_ultrastrong_trend": float(row_last["d_exact_us"] < row0["d_exact_us"]),
        "pass_theta0_no_coherence": float(theta0_coh < 1e-10),
        "pass_theta_pi2_mixed": float(theta_pi2_mixed_max_abs < 1e-12),
    }
    return summary


def make_alignment_figure(df: pd.DataFrame, out_path: Path, config: BenchmarkConfig) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["lambda"], df["d_exact_weak"], label="Exact vs direct-kernel weak target")
    axes[0].plot(df["lambda"], df["d_exact_hmf"], label="Exact vs ordered-Gaussian finite model")
    axes[0].plot(df["lambda"], df["d_exact_hmf_collapsed"], linestyle="--", label="Exact vs collapsed-product model")
    axes[0].plot(df["lambda"], df["d_exact_us"], label="Exact vs ultrastrong PRL")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Trace distance")
    axes[0].set_title("State alignment vs coupling")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["lambda"], df["exact_bloch_x"], linewidth=2, label=r"Exact $\langle \sigma_x \rangle$")
    axes[1].plot(
        df["lambda"], df["hmf_bloch_x"], linestyle="--", label=r"Ordered-Gaussian model $\langle \sigma_x \rangle$"
    )
    axes[1].plot(df["lambda"], df["us_bloch_x"], linestyle=":", label=r"US PRL $\langle \sigma_x \rangle$")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Bloch-x")
    axes[1].set_title("Coherence-bearing component")
    axes[1].legend(fontsize=8)

    axes[2].plot(df["lambda"], df["exact_bloch_z"], linewidth=2, label=r"Exact $\langle \sigma_z \rangle$")
    axes[2].plot(
        df["lambda"], df["hmf_bloch_z"], linestyle="--", label=r"Ordered-Gaussian model $\langle \sigma_z \rangle$"
    )
    axes[2].plot(df["lambda"], df["us_bloch_z"], linestyle=":", label=r"US PRL $\langle \sigma_z \rangle$")
    axes[2].set_xlabel(r"$\lambda$")
    axes[2].set_ylabel("Bloch-z")
    axes[2].set_title(
        rf"Population component ($\beta={config.beta}$, $\theta={config.theta:.2f}$)"
    )
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(df: pd.DataFrame, summary: Dict[str, float], config: BenchmarkConfig) -> Tuple[Path, Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "simulations" / "results" / "data"
    fig_dir = project_root / "simulations" / "results" / "figures"
    ms_fig_dir = project_root / "manuscript" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ms_fig_dir.mkdir(parents=True, exist_ok=True)

    scan_path = data_dir / f"{config.output_prefix}_scan.csv"
    summary_path = data_dir / f"{config.output_prefix}_summary.csv"
    fig_path = fig_dir / f"{config.output_prefix}_alignment.png"
    ms_fig_path = ms_fig_dir / "hmf_prl_qubit_analytic_bridge_alignment.png"
    df.to_csv(scan_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    make_alignment_figure(df, fig_path, config)
    shutil.copy2(fig_path, ms_fig_path)
    return scan_path, summary_path, fig_path, ms_fig_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRL 127.250601 qubit analytic-bridge benchmark.")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--omega-q", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--lambda-min", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=6.0)
    parser.add_argument("--lambda-points", type=int, default=49)
    parser.add_argument("--n-modes", type=int, default=2)
    parser.add_argument("--n-cut", type=int, default=4)
    parser.add_argument("--omega-min", type=float, default=0.5)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--q-strength", type=float, default=10.0)
    parser.add_argument("--tau-c", type=float, default=1.0)
    parser.add_argument("--n-kernel-grid", type=int, default=4001)
    parser.add_argument("--n-matsubara", type=int, default=None, help="Deprecated alias for --n-kernel-grid.")
    parser.add_argument("--ordered-time-slices", type=int, default=80)
    parser.add_argument("--ordered-kl-rank", type=int, default=5)
    parser.add_argument("--ordered-gh-order", type=int, default=3)
    parser.add_argument("--ordered-max-nodes", type=int, default=20000)
    parser.add_argument("--output-prefix", type=str, default="prl127_qubit_analytic_bridge")
    return parser


def run_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float], Path, Path, Path, Path]:
    config = BenchmarkConfig(
        beta=args.beta,
        omega_q=args.omega_q,
        theta=args.theta,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_points=args.lambda_points,
        n_modes=args.n_modes,
        n_cut=args.n_cut,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        q_strength=args.q_strength,
        tau_c=args.tau_c,
        output_prefix=args.output_prefix,
    )
    n_kernel_grid = args.n_kernel_grid if args.n_matsubara is None else args.n_matsubara
    df, summary = run_scan(
        config,
        n_kernel_grid=n_kernel_grid,
        ordered_time_slices=args.ordered_time_slices,
        ordered_kl_rank=args.ordered_kl_rank,
        ordered_gh_order=args.ordered_gh_order,
        ordered_max_nodes=args.ordered_max_nodes,
    )
    scan_path, summary_path, fig_path, ms_fig_path = write_outputs(df, summary, config)
    return df, summary, scan_path, summary_path, fig_path, ms_fig_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df, summary, scan_path, summary_path, fig_path, ms_fig_path = run_from_args(args)
    print("PRL 127.250601 qubit analytic bridge complete.")
    print(f"Rows: {len(df)}")
    print(f"Scan CSV: {scan_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Figure: {fig_path}")
    print(f"Manuscript figure: {ms_fig_path}")
    print(f"Eq8 identity max abs diff: {summary['eq8_identity_max_abs_global']:.3e}")


if __name__ == "__main__":
    main()
