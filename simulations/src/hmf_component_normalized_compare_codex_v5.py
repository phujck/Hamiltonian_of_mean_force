"""
Compare analytically normalized compact density components against ordered model.

This script evaluates compact-model density components directly via bounded formulas:
    m_z = (u - t) / (1 - u t)
    m_x = (q * sech(a)) / (1 - u t)
where
    a = beta * omega_q / 2
    t = tanh(a)
    u = gamma * Delta_z
    q = gamma * sqrt(Sigma_+ Sigma_-)
    gamma = tanh(chi)/chi, chi^2 = Delta_z^2 + Sigma_+ Sigma_-

It writes fresh output files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    RenormConfig,
    ordered_gaussian_state,
    laplace_k0,
    resonant_r0,
    extract_density,
)


@dataclass
class SweepDef:
    name: str
    param_name: str
    param_values: np.ndarray
    beta_fixed: float | None
    theta_fixed: float | None
    g_fixed: float | None
    xlabel: str
    caption: str


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _base_channels(cfg: BenchmarkConfig) -> tuple[float, float, float]:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    c = float(np.cos(cfg.theta))
    s = float(np.sin(cfg.theta))
    k0_0 = laplace_k0(cfg, 0.0, 1001)
    k0_p = laplace_k0(cfg, w, 1001)
    k0_m = laplace_k0(cfg, -w, 1001)
    r_p = resonant_r0(cfg, w, 1001)
    r_m = resonant_r0(cfg, -w, 1001)

    sigma_plus0 = (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sigma_minus0 = (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    delta_z0 = (s**2) * 0.5 * (r_p - r_m)
    return float(sigma_plus0), float(sigma_minus0), float(delta_z0)


def _apply_running(
    sigma_plus: float,
    sigma_minus: float,
    delta_z: float,
    beta: float,
    omega_q: float,
    renorm: RenormConfig,
) -> tuple[float, float, float]:
    chi_raw = float(np.sqrt(max(delta_z * delta_z + sigma_plus * sigma_minus, 0.0)))
    if chi_raw <= renorm.eps:
        return sigma_plus, sigma_minus, delta_z
    a = 0.5 * beta * omega_q
    chi_cap = max(renorm.kappa * abs(a), renorm.eps)
    run = 1.0 / (1.0 + chi_raw / chi_cap)
    return run * sigma_plus, run * sigma_minus, run * delta_z


def _normalized_components(
    beta: float,
    omega_q: float,
    sigma_plus: float,
    sigma_minus: float,
    delta_z: float,
) -> tuple[float, float]:
    """
    Returns Bloch components (m_x, m_z) in a stable, analytically normalized form.
    """
    chi_sq = float(delta_z * delta_z + sigma_plus * sigma_minus)
    chi = float(np.sqrt(max(chi_sq, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-12 else 1.0

    u = gamma * delta_z
    q = gamma * np.sqrt(max(sigma_plus * sigma_minus, 0.0))

    a = 0.5 * beta * omega_q
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-14:
        den = 1e-14 if den >= 0 else -1e-14

    mz = (u - t) / den
    mx = (q / np.cosh(a)) / den
    return float(mx), float(mz)


def _density_from_components(mx: float, mz: float) -> tuple[float, float, float, float]:
    # rho = 0.5 * (I + mx sigma_x + mz sigma_z)
    p00 = 0.5 * (1.0 + mz)
    p11 = 0.5 * (1.0 - mz)
    coh = 0.5 * abs(mx)
    # Clamp tiny numerical drift
    p00 = float(np.clip(p00, 0.0, 1.0))
    p11 = float(np.clip(p11, 0.0, 1.0))
    z = p00 + p11
    if z <= 0.0:
        p00, p11 = 0.5, 0.5
    else:
        p00, p11 = p00 / z, p11 / z
    ratio = float(p00 / max(p11, 1e-15))
    return float(p00), float(p11), float(coh), ratio


def _compact_components(
    cfg: BenchmarkConfig,
    g: float,
    use_running: bool,
    renorm: RenormConfig,
) -> tuple[float, float, float, float]:
    sp0, sm0, dz0 = _base_channels(cfg)
    g2 = float(g * g)
    sp = g2 * sp0
    sm = g2 * sm0
    dz = g2 * dz0

    if use_running:
        sp, sm, dz = _apply_running(sp, sm, dz, cfg.beta, cfg.omega_q, renorm)

    mx, mz = _normalized_components(cfg.beta, cfg.omega_q, sp, sm, dz)
    return _density_from_components(mx, mz)


def run_compare() -> tuple[pd.DataFrame, pd.DataFrame]:
    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.linspace(0.0, 2.0, 11),
            beta_fixed=2.0,
            theta_fixed=np.pi / 4,
            g_fixed=None,
            xlabel=r"$g$",
            caption=r"$\beta=2,\ \theta=\pi/4$",
        ),
        SweepDef(
            name="angle",
            param_name="theta",
            param_values=np.linspace(0.0, np.pi / 2, 11),
            beta_fixed=2.0,
            theta_fixed=None,
            g_fixed=0.5,
            xlabel=r"$\theta/\pi$",
            caption=r"$\beta=2,\ g=0.5$",
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.linspace(0.2, 6.0, 11),
            beta_fixed=None,
            theta_fixed=np.pi / 2,
            g_fixed=0.5,
            xlabel=r"$\beta$",
            caption=r"$\theta=\pi/2,\ g=0.5$",
        ),
    ]

    renorm = RenormConfig(scale=1.04, kappa=0.94)
    rows: list[dict[str, float | str]] = []

    for sweep in sweeps:
        for param in sweep.param_values:
            beta = float(param) if sweep.param_name == "beta" else float(sweep.beta_fixed)
            theta = float(param) if sweep.param_name == "theta" else float(sweep.theta_fixed)
            g = float(param) if sweep.param_name == "g" else float(sweep.g_fixed)
            cfg = BenchmarkConfig(
                beta=beta,
                omega_q=2.0,
                theta=theta,
                n_modes=40,
                n_cut=1,
                omega_min=0.1,
                omega_max=10.0,
                q_strength=5.0,
                tau_c=0.5,
            )

            ord_p00, ord_p11, ord_coh = extract_density(ordered_gaussian_state(cfg, g))
            ord_ratio = float(ord_p00 / max(ord_p11, 1e-15))

            raw_p00, raw_p11, raw_coh, raw_ratio = _compact_components(cfg, g, use_running=False, renorm=renorm)
            run_p00, run_p11, run_coh, run_ratio = _compact_components(cfg, g, use_running=True, renorm=renorm)

            rows.append(
                {
                    "sweep": sweep.name,
                    "param_name": sweep.param_name,
                    "param": float(param),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "ordered_p00": ord_p00,
                    "ordered_p11": ord_p11,
                    "ordered_coh": ord_coh,
                    "ordered_ratio": ord_ratio,
                    "compact_raw_p00": raw_p00,
                    "compact_raw_p11": raw_p11,
                    "compact_raw_coh": raw_coh,
                    "compact_raw_ratio": raw_ratio,
                    "compact_running_p00": run_p00,
                    "compact_running_p11": run_p11,
                    "compact_running_coh": run_coh,
                    "compact_running_ratio": run_ratio,
                    "raw_dp00": raw_p00 - ord_p00,
                    "raw_dp11": raw_p11 - ord_p11,
                    "raw_dcoh": raw_coh - ord_coh,
                    "raw_ratio_over_ordered": raw_ratio / max(ord_ratio, 1e-15),
                    "running_dp00": run_p00 - ord_p00,
                    "running_dp11": run_p11 - ord_p11,
                    "running_dcoh": run_coh - ord_coh,
                    "running_ratio_over_ordered": run_ratio / max(ord_ratio, 1e-15),
                }
            )

    df = pd.DataFrame.from_records(rows)
    summary_rows = []
    for sweep in df["sweep"].unique():
        g = df[df["sweep"] == sweep]
        for model in ("compact_raw", "compact_running"):
            if model == "compact_raw":
                dp00 = g["raw_dp00"]
                dcoh = g["raw_dcoh"]
                ratio_rel = g["raw_ratio_over_ordered"]
            else:
                dp00 = g["running_dp00"]
                dcoh = g["running_dcoh"]
                ratio_rel = g["running_ratio_over_ordered"]
            summary_rows.append(
                {
                    "sweep": sweep,
                    "model": model,
                    "rmse_p00": _rmse(dp00),
                    "rmse_coh": _rmse(dcoh),
                    "ratio_over_ordered_median": float(np.median(ratio_rel)),
                    "ratio_over_ordered_min": float(np.min(ratio_rel)),
                    "ratio_over_ordered_max": float(np.max(ratio_rel)),
                }
            )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "model"]).reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_component_normalized_scan_codex_v5.csv"
    summary_csv = out_dir / "hmf_component_normalized_summary_codex_v5.csv"
    fig_png = out_dir / "hmf_component_normalized_compare_codex_v5.png"
    log_md = out_dir / "hmf_component_normalized_log_codex_v5.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), constrained_layout=True)
    for row, sweep in enumerate(["coupling", "angle", "temperature"]):
        g = df[df["sweep"] == sweep].sort_values("param")
        x = g["param"].to_numpy()
        if sweep == "angle":
            x = x / np.pi

        axes[row, 0].plot(x, g["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
        axes[row, 0].plot(x, g["compact_raw_p00"], color="#888888", linestyle="--", linewidth=1.5, label="Compact raw")
        axes[row, 0].plot(x, g["compact_running_p00"], color="#0B6E4F", linewidth=2.0, label="Compact running")
        axes[row, 0].set_ylabel(r"$\rho_{00}$")
        axes[row, 0].set_title(f"{sweep.capitalize()} population")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(x, g["ordered_ratio"], color="black", linewidth=2.0, label="Ordered ratio")
        axes[row, 1].plot(x, g["compact_raw_ratio"], color="#888888", linestyle="--", linewidth=1.5, label="Compact raw ratio")
        axes[row, 1].plot(x, g["compact_running_ratio"], color="#C84B31", linewidth=2.0, label="Compact running ratio")
        axes[row, 1].set_ylabel(r"$\rho_{00}/\rho_{11}$")
        axes[row, 1].set_title(f"{sweep.capitalize()} ratio")
        axes[row, 1].grid(alpha=0.25)

        axes[row, 0].set_xlabel(g["param_name"].iloc[0] if sweep != "angle" else "theta/pi")
        axes[row, 1].set_xlabel(g["param_name"].iloc[0] if sweep != "angle" else "theta/pi")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Normalized Component Compact Model vs Ordered", fontsize=12)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Normalized-Component Comparison (Codex v5)")
    lines.append("")
    lines.append("This run compares direct analytically normalized compact components against ordered outputs.")
    lines.append("")
    lines.append("| sweep | model | rmse_p00 | rmse_coh | ratio_over_ordered_median | ratio_over_ordered_min | ratio_over_ordered_max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['model']} | {r['rmse_p00']:.6f} | {r['rmse_coh']:.6f} | "
            f"{r['ratio_over_ordered_median']:.6f} | {r['ratio_over_ordered_min']:.6f} | {r['ratio_over_ordered_max']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


def main() -> None:
    df, summary = run_compare()
    write_outputs(df, summary)


if __name__ == "__main__":
    main()
