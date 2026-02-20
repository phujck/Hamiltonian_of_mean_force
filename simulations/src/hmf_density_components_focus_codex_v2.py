"""
Density-component-focused HMF diagnostics (Codex v2).

Outputs only density-based comparisons:
- rho_00, rho_11, |rho_01|
for ordered, legacy compact, and running-renormalized compact models.
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
    extract_density,
    ordered_gaussian_state,
    v5_state_legacy,
    v5_state_running,
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


def fit_renorm_density_only(
    sweeps: list[SweepDef],
    scale_grid: np.ndarray,
    kappa_grid: np.ndarray,
) -> RenormConfig:
    rows = []
    for sweep in sweeps:
        for param in sweep.param_values:
            beta = float(param) if sweep.param_name == "beta" else float(sweep.beta_fixed)
            theta = float(param) if sweep.param_name == "theta" else float(sweep.theta_fixed)
            g = float(param) if sweep.param_name == "g" else float(sweep.g_fixed)
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
            p_ord, p11_ord, c_ord = extract_density(ordered_gaussian_state(cfg, g))
            rows.append((cfg, g, p_ord, p11_ord, c_ord))

    best_score = np.inf
    best = RenormConfig()

    for scale in scale_grid:
        for kappa in kappa_grid:
            renorm = RenormConfig(scale=float(scale), kappa=float(kappa))
            sq_err = []
            for cfg, g, p_ord, p11_ord, c_ord in rows:
                rho_run, _ = v5_state_running(cfg, g, renorm)
                p_run, p11_run, c_run = extract_density(rho_run)
                sq_err.append((p_run - p_ord) ** 2)
                sq_err.append((p11_run - p11_ord) ** 2)
                sq_err.append((c_run - c_ord) ** 2)
            score = float(np.sqrt(np.mean(np.asarray(sq_err, dtype=float))))
            if score < best_score:
                best_score = score
                best = renorm

    return best


def run_density_component_sweeps(
    output_csv: Path,
    summary_csv: Path,
    output_png: Path,
    output_log: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, RenormConfig]:
    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.linspace(0.0, 2.5, 30),
            beta_fixed=2.0,
            theta_fixed=np.pi / 4,
            g_fixed=None,
            xlabel=r"$g$",
            caption=r"$\beta=2,\ \theta=\pi/4$",
        ),
        SweepDef(
            name="angle",
            param_name="theta",
            param_values=np.linspace(0.0, np.pi / 2, 30),
            beta_fixed=2.0,
            theta_fixed=None,
            g_fixed=0.5,
            xlabel=r"$\theta/\pi$",
            caption=r"$\beta=2,\ g=0.5$",
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.linspace(0.1, 8.0, 30),
            beta_fixed=None,
            theta_fixed=np.pi / 2,
            g_fixed=0.5,
            xlabel=r"$\beta$",
            caption=r"$\theta=\pi/2,\ g=0.5$",
        ),
    ]

    # Density-only calibration for running parameters.
    scale_grid = np.linspace(0.8, 1.2, 41)
    kappa_grid = np.linspace(0.7, 1.2, 51)
    best = fit_renorm_density_only(sweeps, scale_grid, kappa_grid)

    rows: list[dict[str, float | str]] = []
    for sweep in sweeps:
        for param in sweep.param_values:
            beta = float(param) if sweep.param_name == "beta" else float(sweep.beta_fixed)
            theta = float(param) if sweep.param_name == "theta" else float(sweep.theta_fixed)
            g = float(param) if sweep.param_name == "g" else float(sweep.g_fixed)
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
            rho_leg, _ = v5_state_legacy(cfg, g)
            rho_run, _ = v5_state_running(cfg, g, best)

            ord_p00, ord_p11, ord_coh = extract_density(rho_ord)
            leg_p00, leg_p11, leg_coh = extract_density(rho_leg)
            run_p00, run_p11, run_coh = extract_density(rho_run)

            rows.append(
                {
                    "sweep": sweep.name,
                    "param_name": sweep.param_name,
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
                    "legacy_dp00": leg_p00 - ord_p00,
                    "legacy_dp11": leg_p11 - ord_p11,
                    "legacy_dcoh": leg_coh - ord_coh,
                    "running_dp00": run_p00 - ord_p00,
                    "running_dp11": run_p11 - ord_p11,
                    "running_dcoh": run_coh - ord_coh,
                }
            )

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)

    summary_rows = []
    for sweep_name, grp in df.groupby("sweep"):
        for model, prefix in (("legacy", "legacy"), ("running", "running")):
            summary_rows.append(
                {
                    "sweep": sweep_name,
                    "model": model,
                    "rmse_p00": _rmse(grp[f"{prefix}_dp00"]),
                    "rmse_p11": _rmse(grp[f"{prefix}_dp11"]),
                    "rmse_coh": _rmse(grp[f"{prefix}_dcoh"]),
                    "max_abs_p00": float(np.max(np.abs(grp[f"{prefix}_dp00"]))),
                    "max_abs_p11": float(np.max(np.abs(grp[f"{prefix}_dp11"]))),
                    "max_abs_coh": float(np.max(np.abs(grp[f"{prefix}_dcoh"]))),
                }
            )

    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "model"]).reset_index(drop=True)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for row, sweep in enumerate(sweeps):
        grp = df[df["sweep"] == sweep.name].sort_values("param")
        x = grp["param"].to_numpy()
        x_plot = x / np.pi if sweep.param_name == "theta" else x

        def draw(col_ord: str, col_leg: str, col_run: str, ax, title: str, ylabel: str) -> None:
            ax.plot(x_plot, grp[col_ord], color="black", linewidth=2.0, label="Ordered")
            ax.plot(x_plot, grp[col_run], color="#0B6E4F", linewidth=2.0, label="Compact (running)")
            ax.plot(x_plot, grp[col_leg], color="#888888", linestyle="--", linewidth=1.5, label="Compact (legacy)")
            ax.set_title(title)
            ax.set_xlabel(sweep.xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)

        draw("ord_p00", "legacy_p00", "running_p00", axes[row, 0], f"rho_00: {sweep.caption}", r"$\rho_{00}$")
        draw("ord_p11", "legacy_p11", "running_p11", axes[row, 1], f"rho_11: {sweep.caption}", r"$\rho_{11}$")
        draw("ord_coh", "legacy_coh", "running_coh", axes[row, 2], f"|rho_01|: {sweep.caption}", r"$|\rho_{01}|$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Density Components Across Sweeps: Ordered vs Compact Models", fontsize=13)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Density-Focused Debug Log (Codex v2)")
    lines.append("")
    lines.append("## Scope")
    lines.append("Only density components were used for calibration and validation: rho_00, rho_11, |rho_01|.")
    lines.append("")
    lines.append("## Fitted Running Parameters")
    lines.append(f"- scale = {best.scale:.6f}")
    lines.append(f"- kappa = {best.kappa:.6f}")
    lines.append("- model: chi_eff = chi_raw / (1 + chi_raw / (kappa * beta*omega_q/2))")
    lines.append("")
    lines.append("## RMSE Summary")
    lines.append("")
    lines.append("| sweep | model | rmse_p00 | rmse_p11 | rmse_coh | max_abs_p00 | max_abs_p11 | max_abs_coh |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['sweep']} | {row['model']} | {row['rmse_p00']:.6f} | {row['rmse_p11']:.6f} | "
            f"{row['rmse_coh']:.6f} | {row['max_abs_p00']:.6f} | {row['max_abs_p11']:.6f} | {row['max_abs_coh']:.6f} |"
        )
    output_log.write_text("\n".join(lines), encoding="utf-8")

    return df, summary, best


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "hmf_density_components_codex_v2.csv"
    summary_path = out_dir / "hmf_density_components_summary_codex_v2.csv"
    png_path = out_dir / "hmf_density_components_codex_v2.png"
    log_path = out_dir / "hmf_density_debug_log_codex_v2.md"

    _, summary, best = run_density_component_sweeps(
        output_csv=csv_path,
        summary_csv=summary_path,
        output_png=png_path,
        output_log=log_path,
    )

    print("Wrote:", csv_path.name)
    print("Wrote:", summary_path.name)
    print("Wrote:", png_path.name)
    print("Wrote:", log_path.name)
    print("")
    print(f"Best density-fit renorm: scale={best.scale:.6f}, kappa={best.kappa:.6f}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

