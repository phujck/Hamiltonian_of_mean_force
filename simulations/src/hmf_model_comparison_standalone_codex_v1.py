"""
Standalone HMF comparison with running renormalization for compact v5 channels.

This script is intentionally self-contained and writes only suffixed output files.
It does not modify any pre-existing files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh, logm


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass
class BenchmarkConfig:
    beta: float
    omega_q: float
    theta: float
    n_modes: int
    n_cut: int
    omega_min: float
    omega_max: float
    q_strength: float
    tau_c: float


@dataclass
class RenormConfig:
    # Multiplicative channel prefactor (fit close to unity from sweeps)
    scale: float = 1.04
    # Running cap relative to a = beta * omega_q / 2
    kappa: float = 0.94
    eps: float = 1e-10


@dataclass
class ChannelSet:
    sigma_plus: float
    sigma_minus: float
    delta_z: float
    chi_raw: float
    chi_eff: float
    run_factor: float
    gamma: float


def _discrete_bath(config: BenchmarkConfig) -> tuple[np.ndarray, float, np.ndarray]:
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)], dtype=float)
        delta_omega = float(config.omega_max - config.omega_min)
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes, dtype=float)
        delta_omega = float(omegas[1] - omegas[0])

    j0 = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    return omegas, delta_omega, j0


def _kernel_profile(config: BenchmarkConfig, u_abs: np.ndarray) -> np.ndarray:
    beta = config.beta
    omegas, delta_omega, j0 = _discrete_bath(config)
    g2 = np.maximum(j0, 0.0) * delta_omega

    kernel = np.zeros_like(u_abs, dtype=float)
    for omega_k, g2_k in zip(omegas, g2):
        denom = np.sinh(0.5 * beta * omega_k)
        if abs(denom) < 1e-14:
            kernel += (2.0 * g2_k) / max(beta * omega_k, 1e-14)
        else:
            kernel += g2_k * np.cosh(omega_k * (0.5 * beta - u_abs)) / denom
    return kernel


def _integration_grid(beta: float, n_points: int) -> np.ndarray:
    return np.linspace(0.0, beta, int(n_points), dtype=float)


def laplace_k0(config: BenchmarkConfig, omega: float, n_grid: int = 1001) -> float:
    u = _integration_grid(config.beta, n_grid)
    k0 = _kernel_profile(config, u)
    return float(np.trapezoid(k0 * np.exp(omega * u), u))


def resonant_r0(config: BenchmarkConfig, omega: float, n_grid: int = 1001) -> float:
    u = _integration_grid(config.beta, n_grid)
    k0 = _kernel_profile(config, u)
    return float(np.trapezoid((config.beta - u) * k0 * np.exp(omega * u), u))


def _project_density(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    evals, evecs = eigh(rho)
    evals = np.clip(np.real(evals), 1e-15, None)
    rho = (evecs * evals) @ evecs.conj().T
    rho /= np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


def _x_tilde_operator(config: BenchmarkConfig, tau: float) -> np.ndarray:
    c = np.cos(config.theta)
    s = np.sin(config.theta)
    w = config.omega_q
    return c * SIGMA_Z - s * (np.cosh(w * tau) * SIGMA_X + 1.0j * np.sinh(w * tau) * SIGMA_Y)


def _scaled_expm(matrix: np.ndarray) -> np.ndarray:
    evals = np.linalg.eigvals(matrix)
    shift = float(np.max(np.real(evals)))
    from scipy.linalg import expm

    return expm(matrix - shift * IDENTITY_2)


def ordered_gaussian_state(config: BenchmarkConfig, g: float) -> np.ndarray:
    n_slices = 40
    kl_rank = 4
    gh_order = 4

    t = np.linspace(0.0, config.beta, int(n_slices), dtype=float)
    dt = float(t[1] - t[0])
    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, config.beta - u)
    cov = _kernel_profile(config, u)

    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = np.clip(evals[idx], 0.0, None)
    evecs = evecs[:, idx]
    eff_rank = min(kl_rank, int(np.count_nonzero(evals > 1e-14)))

    if eff_rank > 0:
        kl_basis = evecs[:, :eff_rank] * np.sqrt(evals[:eff_rank])
    else:
        kl_basis = np.zeros((len(t), 0), dtype=float)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(gh_order)
    eta_1d = np.sqrt(2.0) * gh_x
    weights_1d = gh_w / np.sqrt(np.pi)

    x_grid = np.asarray([_x_tilde_operator(config, tau) for tau in t], dtype=complex)
    w_avg = np.zeros((2, 2), dtype=complex)

    for inds in itertools.product(range(gh_order), repeat=eff_rank):
        ind_arr = np.asarray(inds, dtype=int)
        eta_vec = eta_1d[ind_arr] if eff_rank > 0 else np.zeros(0, dtype=float)
        xi = g * (kl_basis @ eta_vec) if eff_rank > 0 else np.zeros(n_slices, dtype=float)
        weight = float(np.prod(weights_1d[ind_arr])) if eff_rank > 0 else 1.0

        u_op = IDENTITY_2.copy()
        for n in range(n_slices):
            v = dt * xi[n]
            step = np.cosh(v) * IDENTITY_2 - np.sinh(v) * x_grid[n]
            u_op = step @ u_op
        w_avg += weight * u_op

    hs = 0.5 * config.omega_q * SIGMA_Z
    prefactor = _scaled_expm(-config.beta * hs)
    rho = prefactor @ w_avg
    rho = _project_density(rho)
    return rho


def _raw_base_channels(config: BenchmarkConfig, sign_xy: float, sign_z: float) -> tuple[float, float, float]:
    beta = config.beta
    omega_q = config.omega_q
    c = np.cos(config.theta)
    s = np.sin(config.theta)

    k0_zero = laplace_k0(config, 0.0)
    k0_plus = laplace_k0(config, omega_q)
    k0_minus = laplace_k0(config, -omega_q)
    r0_plus = resonant_r0(config, omega_q)
    r0_minus = resonant_r0(config, -omega_q)

    sigma_plus0 = sign_xy * (c * s / omega_q) * ((1.0 + np.exp(beta * omega_q)) * k0_zero - 2.0 * k0_plus)
    sigma_minus0 = sign_xy * (c * s / omega_q) * (
        (1.0 + np.exp(-beta * omega_q)) * k0_zero - 2.0 * k0_minus
    )
    delta_z0 = sign_z * (s**2) * 0.5 * (r0_plus - r0_minus)
    return float(sigma_plus0), float(sigma_minus0), float(delta_z0)


def _build_channels_running(config: BenchmarkConfig, g: float, renorm: RenormConfig) -> ChannelSet:
    sigma_plus0, sigma_minus0, delta_z0 = _raw_base_channels(config, sign_xy=1.0, sign_z=1.0)
    g2 = float(g * g)

    sigma_plus = renorm.scale * g2 * sigma_plus0
    sigma_minus = renorm.scale * g2 * sigma_minus0
    delta_z = renorm.scale * g2 * delta_z0

    chi_raw = float(np.sqrt(max(delta_z * delta_z + sigma_plus * sigma_minus, 0.0)))
    if chi_raw <= renorm.eps:
        return ChannelSet(
            sigma_plus=sigma_plus,
            sigma_minus=sigma_minus,
            delta_z=delta_z,
            chi_raw=chi_raw,
            chi_eff=chi_raw,
            run_factor=1.0,
            gamma=1.0,
        )

    a = 0.5 * config.beta * config.omega_q
    chi_cap = max(renorm.kappa * abs(a), renorm.eps)
    run = 1.0 / (1.0 + chi_raw / chi_cap)

    sigma_plus_eff = run * sigma_plus
    sigma_minus_eff = run * sigma_minus
    delta_z_eff = run * delta_z
    chi_eff = run * chi_raw
    gamma = float(np.tanh(chi_eff) / chi_eff) if chi_eff > renorm.eps else 1.0

    return ChannelSet(
        sigma_plus=float(sigma_plus_eff),
        sigma_minus=float(sigma_minus_eff),
        delta_z=float(delta_z_eff),
        chi_raw=chi_raw,
        chi_eff=float(chi_eff),
        run_factor=float(run),
        gamma=gamma,
    )


def _build_channels_legacy(config: BenchmarkConfig, g: float) -> ChannelSet:
    # Historical standalone settings: flipped transverse sign and fixed 1/4 scaling.
    sigma_plus0, sigma_minus0, delta_z0 = _raw_base_channels(config, sign_xy=-1.0, sign_z=1.0)
    g2 = float(g * g)
    scale = 0.25
    sigma_plus = scale * g2 * sigma_plus0
    sigma_minus = scale * g2 * sigma_minus0
    delta_z = scale * g2 * delta_z0
    chi_eff = float(np.sqrt(max(delta_z * delta_z + sigma_plus * sigma_minus, 0.0)))
    gamma = float(np.tanh(chi_eff) / chi_eff) if chi_eff > 1e-10 else 1.0
    return ChannelSet(
        sigma_plus=float(sigma_plus),
        sigma_minus=float(sigma_minus),
        delta_z=float(delta_z),
        chi_raw=chi_eff,
        chi_eff=chi_eff,
        run_factor=1.0,
        gamma=gamma,
    )


def _state_from_channels(config: BenchmarkConfig, channels: ChannelSet) -> np.ndarray:
    a = 0.5 * config.beta * config.omega_q
    gamma = channels.gamma
    d = channels.delta_z
    sp = channels.sigma_plus
    sm = channels.sigma_minus

    z_q = 2.0 * (np.cosh(a) - gamma * d * np.sinh(a))
    rho = np.array(
        [
            [np.exp(-a) * (1.0 + gamma * d), np.exp(-a) * gamma * sp],
            [np.exp(a) * gamma * sm, np.exp(a) * (1.0 - gamma * d)],
        ],
        dtype=complex,
    )
    rho /= z_q
    return _project_density(rho)


def v5_state_running(config: BenchmarkConfig, g: float, renorm: RenormConfig) -> tuple[np.ndarray, ChannelSet]:
    channels = _build_channels_running(config, g, renorm)
    return _state_from_channels(config, channels), channels


def v5_state_legacy(config: BenchmarkConfig, g: float) -> tuple[np.ndarray, ChannelSet]:
    channels = _build_channels_legacy(config, g)
    return _state_from_channels(config, channels), channels


def extract_density(rho: np.ndarray) -> tuple[float, float, float]:
    p00 = float(np.real(rho[0, 0]))
    p11 = float(np.real(rho[1, 1]))
    coh = float(abs(rho[0, 1]))
    return p00, p11, coh


def get_hmf_fields(rho: np.ndarray, beta: float) -> tuple[float, float]:
    rho = _project_density(rho)
    evals, evecs = eigh(rho)
    evals = np.clip(np.real(evals), 1e-14, 1.0)
    log_rho = evecs @ np.diag(np.log(evals)) @ evecs.conj().T
    hmf = -1.0 / beta * log_rho
    hx = float(np.real(np.trace(hmf @ SIGMA_X)) / 2.0)
    hz = float(np.real(np.trace(hmf @ SIGMA_Z)) / 2.0)
    return hx, hz


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def run_field_scan(
    theta_val: float,
    g_val: float,
    output_csv: Path,
    output_png: Path,
    renorm: RenormConfig,
) -> pd.DataFrame:
    betas = np.linspace(0.1, 5.0, 20)
    rows: list[dict[str, float]] = []

    for beta in betas:
        cfg = BenchmarkConfig(
            beta=beta,
            omega_q=2.0,
            theta=theta_val,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )

        rho_ord = ordered_gaussian_state(cfg, g_val)
        rho_leg, ch_leg = v5_state_legacy(cfg, g_val)
        rho_run, ch_run = v5_state_running(cfg, g_val, renorm)

        ord_hx, ord_hz = get_hmf_fields(rho_ord, beta)
        leg_hx, leg_hz = get_hmf_fields(rho_leg, beta)
        run_hx, run_hz = get_hmf_fields(rho_run, beta)

        rows.append(
            {
                "beta": float(beta),
                "ord_hx": ord_hx,
                "ord_hz": ord_hz,
                "legacy_hx": leg_hx,
                "legacy_hz": leg_hz,
                "running_hx": run_hx,
                "running_hz": run_hz,
                "legacy_abs_err_hx": abs(leg_hx - ord_hx),
                "legacy_abs_err_hz": abs(leg_hz - ord_hz),
                "running_abs_err_hx": abs(run_hx - ord_hx),
                "running_abs_err_hz": abs(run_hz - ord_hz),
                "legacy_chi": ch_leg.chi_eff,
                "running_chi_raw": ch_run.chi_raw,
                "running_chi_eff": ch_run.chi_eff,
                "running_factor": ch_run.run_factor,
            }
        )

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(df["beta"], df["ord_hz"], color="black", linewidth=2.0, label="Ordered")
    ax[0].plot(df["beta"], df["running_hz"], color="#0B6E4F", linewidth=2.0, label="Compact (running)")
    ax[0].plot(df["beta"], df["legacy_hz"], color="#888888", linestyle="--", linewidth=1.5, label="Compact (legacy)")
    ax[0].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$h_z$")
    ax[0].grid(alpha=0.25)
    ax[0].set_title("Longitudinal Field")

    ax[1].plot(df["beta"], df["ord_hx"], color="black", linewidth=2.0, label="Ordered")
    ax[1].plot(df["beta"], df["running_hx"], color="#C84B31", linewidth=2.0, label="Compact (running)")
    ax[1].plot(df["beta"], df["legacy_hx"], color="#888888", linestyle="--", linewidth=1.5, label="Compact (legacy)")
    ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel(r"$h_x$")
    ax[1].grid(alpha=0.25)
    ax[1].set_title("Transverse Field")

    title = rf"HMF Field Scan: $\theta={theta_val:.3f}$, $g={g_val:.2f}$"
    fig.suptitle(title, fontsize=11)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return df


def run_population_sweeps(
    output_csv: Path,
    output_summary_csv: Path,
    output_png: Path,
    renorm: RenormConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_defs = [
        {
            "name": "coupling",
            "param_name": "g",
            "param_values": np.linspace(0.0, 2.5, 30),
            "beta": 2.0,
            "theta_mode": "fixed",
            "theta_fixed": np.pi / 4,
            "g_mode": "param",
            "g_fixed": None,
            "xlabel": r"$g$",
            "caption": r"$\beta=2,\ \theta=\pi/4$",
        },
        {
            "name": "angle",
            "param_name": "theta",
            "param_values": np.linspace(0.0, np.pi / 2, 30),
            "beta": 2.0,
            "theta_mode": "param",
            "theta_fixed": None,
            "g_mode": "fixed",
            "g_fixed": 0.5,
            "xlabel": r"$\theta/\pi$",
            "caption": r"$\beta=2,\ g=0.5$",
        },
        {
            "name": "temperature",
            "param_name": "beta",
            "param_values": np.linspace(0.1, 8.0, 30),
            "beta": None,
            "theta_mode": "fixed",
            "theta_fixed": np.pi / 2,
            "g_mode": "fixed",
            "g_fixed": 0.5,
            "xlabel": r"$\beta$",
            "caption": r"$\theta=\pi/2,\ g=0.5$",
        },
    ]

    rows: list[dict[str, float | str]] = []
    for sweep in sweep_defs:
        for param in sweep["param_values"]:
            beta = float(param) if sweep["param_name"] == "beta" else float(sweep["beta"])
            theta = float(param) if sweep["param_name"] == "theta" else float(sweep["theta_fixed"])
            g = float(param) if sweep["param_name"] == "g" else float(sweep["g_fixed"])

            cfg = BenchmarkConfig(
                beta=beta,
                omega_q=1.0,
                theta=theta,
                n_modes=40,
                n_cut=1,
                omega_min=0.1,
                omega_max=10.0,
                q_strength=10.0,
                tau_c=1.0,
            )

            rho_ord = ordered_gaussian_state(cfg, g)
            rho_leg, ch_leg = v5_state_legacy(cfg, g)
            rho_run, ch_run = v5_state_running(cfg, g, renorm)

            ord_p00, ord_p11, ord_coh = extract_density(rho_ord)
            leg_p00, leg_p11, leg_coh = extract_density(rho_leg)
            run_p00, run_p11, run_coh = extract_density(rho_run)

            rows.append(
                {
                    "sweep": sweep["name"],
                    "param_name": sweep["param_name"],
                    "param": float(param),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "ord_p00": ord_p00,
                    "ord_p11": ord_p11,
                    "ord_coh": ord_coh,
                    "legacy_p00": leg_p00,
                    "legacy_p11": leg_p11,
                    "legacy_coh": leg_coh,
                    "running_p00": run_p00,
                    "running_p11": run_p11,
                    "running_coh": run_coh,
                    "legacy_delta_p00": leg_p00 - ord_p00,
                    "legacy_delta_coh": leg_coh - ord_coh,
                    "running_delta_p00": run_p00 - ord_p00,
                    "running_delta_coh": run_coh - ord_coh,
                    "legacy_chi": ch_leg.chi_eff,
                    "running_chi_raw": ch_run.chi_raw,
                    "running_chi_eff": ch_run.chi_eff,
                    "running_factor": ch_run.run_factor,
                }
            )

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)

    summary_rows: list[dict[str, float | str]] = []
    for sweep_name, grp in df.groupby("sweep"):
        for model in ("legacy", "running"):
            rmse_p = _rmse(grp[f"{model}_delta_p00"].to_numpy())
            rmse_c = _rmse(grp[f"{model}_delta_coh"].to_numpy())
            summary_rows.append(
                {
                    "sweep": sweep_name,
                    "model": model,
                    "rmse_p00": rmse_p,
                    "rmse_coh": rmse_c,
                    "max_abs_p00": float(np.max(np.abs(grp[f"{model}_delta_p00"].to_numpy()))),
                    "max_abs_coh": float(np.max(np.abs(grp[f"{model}_delta_coh"].to_numpy()))),
                }
            )

    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "model"]).reset_index(drop=True)
    summary.to_csv(output_summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for row, sweep in enumerate(sweep_defs):
        grp = df[df["sweep"] == sweep["name"]].sort_values("param")
        x = grp["param"].to_numpy()
        if sweep["param_name"] == "theta":
            x_plot = x / np.pi
        else:
            x_plot = x

        axes[row, 0].plot(x_plot, grp["ord_p00"], color="black", linewidth=2.0, label="Ordered")
        axes[row, 0].plot(x_plot, grp["running_p00"], color="#0B6E4F", linewidth=2.0, label="Compact (running)")
        axes[row, 0].plot(x_plot, grp["legacy_p00"], color="#888888", linestyle="--", linewidth=1.5, label="Compact (legacy)")
        axes[row, 0].set_ylabel(r"$\rho_{00}$")
        axes[row, 0].set_xlabel(sweep["xlabel"])
        axes[row, 0].set_title(f"Population: {sweep['caption']}")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(x_plot, grp["ord_coh"], color="black", linewidth=2.0, label="Ordered")
        axes[row, 1].plot(x_plot, grp["running_coh"], color="#C84B31", linewidth=2.0, label="Compact (running)")
        axes[row, 1].plot(x_plot, grp["legacy_coh"], color="#888888", linestyle="--", linewidth=1.5, label="Compact (legacy)")
        axes[row, 1].set_ylabel(r"$|\rho_{01}|$")
        axes[row, 1].set_xlabel(sweep["xlabel"])
        axes[row, 1].set_title(f"Coherence: {sweep['caption']}")
        axes[row, 1].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("HMF Population/Coherence Sweeps: Ordered vs Compact Models", fontsize=12)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    return df, summary


def write_debug_log(log_path: Path, renorm: RenormConfig, summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# HMF Debug Log (Codex v1)")
    lines.append("")
    lines.append("## Objective")
    lines.append("Align compact-v5 populations/coherences with ordered-model sweeps using sign/scaling diagnostics.")
    lines.append("")
    lines.append("## Changes Tested")
    lines.append("1. Legacy baseline: transverse channel sign flipped and fixed `scale=0.25` (historical standalone behavior).")
    lines.append("2. Sign correction: restore manuscript-consistent signs for `Sigma_+` and `Sigma_-`.")
    lines.append(
        "3. Running renormalization: apply `r = 1 / (1 + chi_raw / chi_cap)`, with `chi_cap = kappa * (beta*omega_q/2)`."
    )
    lines.append(f"4. Final parameters: `scale={renorm.scale:.4f}`, `kappa={renorm.kappa:.4f}`.")
    lines.append("")
    lines.append("## Error Sources Identified")
    lines.append("- Fixed 1/4 scale under-corrects low-to-mid coupling and over-regularizes only one regime.")
    lines.append("- Sign mismatch in transverse channels drives wrong direction in generic-angle sweeps.")
    lines.append("- Unbounded raw `chi ~ g^2` causes diagonal over-polarization at strong coupling.")
    lines.append("")
    lines.append("## Sweep RMSE Summary")
    lines.append("")
    lines.append("| sweep | model | rmse_p00 | rmse_coh | max_abs_p00 | max_abs_coh |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['sweep']} | {row['model']} | {row['rmse_p00']:.6f} | {row['rmse_coh']:.6f} | "
            f"{row['max_abs_p00']:.6f} | {row['max_abs_coh']:.6f} |"
        )
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    renorm = RenormConfig(scale=1.04, kappa=0.94)

    pop_csv = out_dir / "hmf_population_sweeps_codex_v1.csv"
    pop_summary_csv = out_dir / "hmf_population_sweeps_summary_codex_v1.csv"
    pop_png = out_dir / "hmf_population_sweeps_codex_v1.png"
    pop_df, pop_summary = run_population_sweeps(pop_csv, pop_summary_csv, pop_png, renorm)

    generic_csv = out_dir / "hmf_cmp_generic_codex_v1.csv"
    generic_png = out_dir / "hmf_cmp_generic_codex_v1.png"
    run_field_scan(theta_val=0.25, g_val=2.5, output_csv=generic_csv, output_png=generic_png, renorm=renorm)

    pi2_csv = out_dir / "hmf_cmp_pi2_codex_v1.csv"
    pi2_png = out_dir / "hmf_cmp_pi2_codex_v1.png"
    run_field_scan(theta_val=np.pi / 2.0, g_val=2.5, output_csv=pi2_csv, output_png=pi2_png, renorm=renorm)

    log_path = out_dir / "hmf_debug_log_codex_v1.md"
    write_debug_log(log_path, renorm, pop_summary)

    # Console summary for quick confirmation.
    print("Wrote:", pop_csv.name)
    print("Wrote:", pop_summary_csv.name)
    print("Wrote:", pop_png.name)
    print("Wrote:", generic_csv.name)
    print("Wrote:", generic_png.name)
    print("Wrote:", pi2_csv.name)
    print("Wrote:", pi2_png.name)
    print("Wrote:", log_path.name)
    print("")
    print("Population sweep summary (RMSE):")
    print(pop_summary.to_string(index=False))
    print("")
    print(f"Rows generated: {len(pop_df)}")


if __name__ == "__main__":
    main()

