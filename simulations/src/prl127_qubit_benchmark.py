"""
Single-qubit benchmark aligned with PRL 127, 250601 (2021).

This script compares:
1) Exact finite-bath reduced equilibrium states across coupling strengths.
2) Bare Gibbs state of H_S.
3) Ultrastrong projected state from Eq. (7), specialized to the qubit Eq. (8).

Outputs:
- CSV scan in simulations/results/data
- Summary figure in simulations/results/figures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass
class BenchmarkConfig:
    beta: float = 1.0
    omega_q: float = 3.0
    theta: float = 0.25
    lambda_min: float = 0.0
    lambda_max: float = 6.0
    lambda_points: int = 49
    n_modes: int = 2
    n_cut: int = 4
    omega_min: float = 0.5
    omega_max: float = 8.0
    q_strength: float = 10.0
    tau_c: float = 1.0
    output_prefix: str = "prl127_qubit"


def spectral_density_exp(omega: np.ndarray, q_strength: float, tau_c: float) -> np.ndarray:
    """PRL Fig. 1 style spectral density: J(w) = Q * tau_c * w * exp(-tau_c * w)."""
    return q_strength * tau_c * omega * np.exp(-tau_c * omega)


def annihilation_operator(n_cut: int) -> np.ndarray:
    op = np.zeros((n_cut, n_cut), dtype=complex)
    for n in range(1, n_cut):
        op[n - 1, n] = np.sqrt(n)
    return op


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for op in operators:
        out = np.kron(out, op)
    return out


def embed_mode_operator(op: np.ndarray, mode_index: int, n_modes: int, n_cut: int) -> np.ndarray:
    identity = np.eye(n_cut, dtype=complex)
    operators = []
    for idx in range(n_modes):
        operators.append(op if idx == mode_index else identity)
    return kron_all(operators)


def thermal_state(hamiltonian: np.ndarray, beta: float) -> np.ndarray:
    evals, evecs = eigh(hamiltonian)
    evals_shifted = evals - np.min(evals)
    weights = np.exp(-beta * evals_shifted)
    weights /= np.sum(weights)
    rho = (evecs * weights) @ evecs.conj().T
    return 0.5 * (rho + rho.conj().T)


def partial_trace_bath(rho_full: np.ndarray, dim_system: int, dim_bath: int) -> np.ndarray:
    reshaped = rho_full.reshape(dim_system, dim_bath, dim_system, dim_bath)
    rho_system = np.trace(reshaped, axis1=1, axis2=3)
    return 0.5 * (rho_system + rho_system.conj().T)


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    diff = 0.5 * ((rho_a - rho_b) + (rho_a - rho_b).conj().T)
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * float(np.sum(np.abs(eigvals)))


def l1_coherence_in_basis(rho: np.ndarray, basis_vectors: np.ndarray) -> float:
    rho_basis = basis_vectors.conj().T @ rho @ basis_vectors
    off_diag = rho_basis - np.diag(np.diag(rho_basis))
    return float(np.sum(np.abs(off_diag)))


def ultrastrong_projected_state(hamiltonian_s: np.ndarray, coupling_op: np.ndarray, beta: float) -> np.ndarray:
    """
    Eq. (7) projection form specialized to finite-dimensional systems:
    rho_US ∝ exp[-beta * sum_n P_n H_S P_n],
    where P_n are eigenprojectors of X.
    """
    eigvals_x, eigvecs_x = eigh(coupling_op)
    dim_s = coupling_op.shape[0]
    h_eff = np.zeros((dim_s, dim_s), dtype=complex)
    for col in range(dim_s):
        vec = eigvecs_x[:, col]
        projector = np.outer(vec, vec.conj())
        h_eff = h_eff + projector @ hamiltonian_s @ projector
    return thermal_state(h_eff, beta)


def bare_gibbs_state(hamiltonian_s: np.ndarray, beta: float) -> np.ndarray:
    return thermal_state(hamiltonian_s, beta)


def build_static_operators(config: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Build static operators for:
      H = H_S + H_B + lambda * X⊗B + lambda^2 * Q * X^2
    with finite-mode discretization of J(w).
    """
    h_s = 0.5 * config.omega_q * SIGMA_Z
    x_op = np.cos(config.theta) * SIGMA_Z - np.sin(config.theta) * SIGMA_X
    xx = x_op @ x_op

    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)])
        delta_omega = config.omega_max - config.omega_min
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes)
        delta_omega = omegas[1] - omegas[0]

    j_vals = spectral_density_exp(omegas, config.q_strength, config.tau_c)
    g_vals = np.sqrt(np.maximum(j_vals, 0.0) * delta_omega)

    dim_b = config.n_cut ** config.n_modes
    identity_b = np.eye(dim_b, dtype=complex)

    a_single = annihilation_operator(config.n_cut)
    adag_single = a_single.conj().T

    h_b = np.zeros((dim_b, dim_b), dtype=complex)
    b_op = np.zeros((dim_b, dim_b), dtype=complex)

    for mode_idx in range(config.n_modes):
        a_k = embed_mode_operator(a_single, mode_idx, config.n_modes, config.n_cut)
        adag_k = embed_mode_operator(adag_single, mode_idx, config.n_modes, config.n_cut)
        number_k = adag_k @ a_k
        h_b = h_b + omegas[mode_idx] * number_k
        b_op = b_op + g_vals[mode_idx] * (a_k + adag_k)

    # Discrete approximation of Q = integral J(w)/w dw
    q_reorg = float(np.sum((g_vals**2) / np.maximum(omegas, 1e-12)))

    h_static = np.kron(h_s, identity_b) + np.kron(IDENTITY_2, h_b)
    h_int = np.kron(x_op, b_op)
    h_counter = np.kron(xx, identity_b) * q_reorg

    return h_static, h_int, h_counter, h_s, x_op


def run_scan(config: BenchmarkConfig) -> pd.DataFrame:
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)

    dim_system = h_s.shape[0]
    dim_bath = h_static.shape[0] // dim_system

    tau_s = bare_gibbs_state(h_s, config.beta)
    rho_us = ultrastrong_projected_state(h_s, x_op, config.beta)

    _, hs_basis = eigh(h_s)
    _, x_basis = eigh(x_op)

    lambda_values = np.linspace(config.lambda_min, config.lambda_max, config.lambda_points)
    records = []

    for lam in lambda_values:
        h_tot = h_static + lam * h_int + (lam**2) * h_counter
        rho_tot = thermal_state(h_tot, config.beta)
        rho_s = partial_trace_bath(rho_tot, dim_system=dim_system, dim_bath=dim_bath)

        c_hs = l1_coherence_in_basis(rho_s, hs_basis)
        c_x = l1_coherence_in_basis(rho_s, x_basis)

        rec = {
            "lambda": float(lam),
            "trace_distance_to_tau": trace_distance(rho_s, tau_s),
            "trace_distance_to_ultrastrong": trace_distance(rho_s, rho_us),
            "coherence_hs_basis": c_hs,
            "coherence_x_basis": c_x,
            "expect_sigma_z": float(np.real(np.trace(rho_s @ SIGMA_Z))),
            "expect_x": float(np.real(np.trace(rho_s @ x_op))),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


def write_outputs(df: pd.DataFrame, config: BenchmarkConfig) -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "simulations" / "results" / "data"
    fig_dir = project_root / "simulations" / "results" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"{config.output_prefix}_scan.csv"
    fig_path = fig_dir / f"{config.output_prefix}_scan.png"

    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["lambda"], df["trace_distance_to_tau"], label=r"$D(\rho_S,\tau_S)$")
    axes[0].plot(df["lambda"], df["trace_distance_to_ultrastrong"], label=r"$D(\rho_S,\rho_{US})$")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Trace distance")
    axes[0].set_title("State-distance crossover")
    axes[0].legend()

    axes[1].plot(df["lambda"], df["coherence_hs_basis"], label=r"$C_{HS}$")
    axes[1].plot(df["lambda"], df["coherence_x_basis"], label=r"$C_{X}$")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\ell_1$ coherence")
    axes[1].set_title("Basis-dependent coherence")
    axes[1].legend()

    axes[2].plot(df["lambda"], df["expect_sigma_z"], label=r"$\langle \sigma_z \rangle$")
    axes[2].plot(df["lambda"], df["expect_x"], label=r"$\langle X \rangle$")
    axes[2].set_xlabel(r"$\lambda$")
    axes[2].set_ylabel("Expectation value")
    axes[2].set_title("Order parameters")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return csv_path, fig_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRL 127.250601 single-qubit benchmark scan.")
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
    parser.add_argument("--output-prefix", type=str, default="prl127_qubit")
    return parser


def run_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, Path, Path]:
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
    df = run_scan(config)
    csv_path, fig_path = write_outputs(df, config)
    return df, csv_path, fig_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    df, csv_path, fig_path = run_from_args(args)
    print("PRL 127.250601 qubit benchmark complete.")
    print(f"Rows: {len(df)}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
