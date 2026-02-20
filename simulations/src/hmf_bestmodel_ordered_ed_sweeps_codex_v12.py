"""
Lightweight sweeps: Best analytic model vs Ordered vs ED (Codex v12).

Produces:
- population/coherence sweep plots (3x2 panel)
- pointwise CSV for all sweeps
- RMSE summary table

Designed for safe execution with run_safe.ps1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
    ordered_gaussian_state,
)
from hmf_component_normalized_compare_codex_v5 import _compact_components
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state


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


def _build_configs(beta: float, theta: float) -> tuple[LiteConfig, EDConfig]:
    # Keep computational load conservative.
    lite = LiteConfig(
        beta=beta,
        omega_q=2.0,
        theta=theta,
        n_modes=24,
        n_cut=1,
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
    )
    # ED finite-bath config: intentionally small Hilbert space for safe runtime.
    ed = EDConfig(
        beta=beta,
        omega_q=2.0,
        theta=theta,
        n_modes=2,
        n_cut=3,
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=2.0,
        lambda_points=3,
        output_prefix="hmf_bestmodel_ordered_ed_codex_v12",
    )
    return lite, ed


def run_sweeps() -> tuple[pd.DataFrame, pd.DataFrame]:
    ren = RenormConfig(scale=1.04, kappa=0.94)
    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.linspace(0.0, 2.0, 17),
            beta_fixed=2.0,
            theta_fixed=float(np.pi / 4.0),
            g_fixed=None,
            xlabel="g",
            caption=r"$\beta=2,\ \theta=\pi/4$",
        ),
        SweepDef(
            name="angle",
            param_name="theta",
            param_values=np.linspace(0.0, float(np.pi / 2.0), 17),
            beta_fixed=2.0,
            theta_fixed=None,
            g_fixed=0.5,
            xlabel=r"$\theta/\pi$",
            caption=r"$\beta=2,\ g=0.5$",
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.linspace(0.2, 6.0, 17),
            beta_fixed=None,
            theta_fixed=float(np.pi / 2.0),
            g_fixed=0.5,
            xlabel=r"$\beta$",
            caption=r"$\theta=\pi/2,\ g=0.5$",
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for sweep in sweeps:
        for p in sweep.param_values:
            beta = float(p) if sweep.param_name == "beta" else float(sweep.beta_fixed)
            theta = float(p) if sweep.param_name == "theta" else float(sweep.theta_fixed)
            g = float(p) if sweep.param_name == "g" else float(sweep.g_fixed)

            lite_cfg, ed_cfg = _build_configs(beta, theta)

            # Ordered
            rho_ord = ordered_gaussian_state(lite_cfg, g)
            ord_p00, ord_p11, ord_coh = extract_density(rho_ord)

            # Best analytic (running normalized compact)
            best_p00, best_p11, best_coh, best_ratio = _compact_components(
                lite_cfg, g, use_running=True, renorm=ren
            )

            # ED (light settings)
            ed_ctx = build_ed_context(ed_cfg)
            rho_ed = exact_reduced_state(ed_ctx, g)
            ed_p00, ed_p11, ed_coh = extract_density(rho_ed)

            rows.append(
                {
                    "sweep": sweep.name,
                    "param_name": sweep.param_name,
                    "param": float(p),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "ordered_p00": float(ord_p00),
                    "ordered_p11": float(ord_p11),
                    "ordered_coh": float(ord_coh),
                    "best_p00": float(best_p00),
                    "best_p11": float(best_p11),
                    "best_coh": float(best_coh),
                    "ed_p00": float(ed_p00),
                    "ed_p11": float(ed_p11),
                    "ed_coh": float(ed_coh),
                    "d_best_vs_ordered_p00": float(best_p00 - ord_p00),
                    "d_best_vs_ordered_coh": float(best_coh - ord_coh),
                    "d_best_vs_ed_p00": float(best_p00 - ed_p00),
                    "d_best_vs_ed_coh": float(best_coh - ed_coh),
                    "d_ordered_vs_ed_p00": float(ord_p00 - ed_p00),
                    "d_ordered_vs_ed_coh": float(ord_coh - ed_coh),
                }
            )

    df = pd.DataFrame.from_records(rows)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": sweep,
                "comparison": "best_vs_ordered",
                "rmse_p00": _rmse(grp["d_best_vs_ordered_p00"]),
                "rmse_coh": _rmse(grp["d_best_vs_ordered_coh"]),
            }
        )
        summary_rows.append(
            {
                "sweep": sweep,
                "comparison": "best_vs_ed",
                "rmse_p00": _rmse(grp["d_best_vs_ed_p00"]),
                "rmse_coh": _rmse(grp["d_best_vs_ed_coh"]),
            }
        )
        summary_rows.append(
            {
                "sweep": sweep,
                "comparison": "ordered_vs_ed",
                "rmse_p00": _rmse(grp["d_ordered_vs_ed_p00"]),
                "rmse_coh": _rmse(grp["d_ordered_vs_ed_coh"]),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "comparison"]).reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_bestmodel_ordered_ed_scan_codex_v12.csv"
    summary_csv = out_dir / "hmf_bestmodel_ordered_ed_summary_codex_v12.csv"
    fig_png = out_dir / "hmf_bestmodel_ordered_ed_sweeps_codex_v12.png"
    log_md = out_dir / "hmf_bestmodel_ordered_ed_log_codex_v12.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    sweep_order = ["coupling", "angle", "temperature"]
    for row, name in enumerate(sweep_order):
        g = df[df["sweep"] == name].sort_values("param")
        x = g["param"].to_numpy(dtype=float)
        xlabel = g["param_name"].iloc[0]
        if name == "angle":
            x = x / np.pi
            xlabel = "theta/pi"

        axes[row, 0].plot(x, g["ed_p00"], color="black", linewidth=2.2, label="ED (light)")
        axes[row, 0].plot(x, g["best_p00"], color="#0B6E4F", linewidth=2.0, label="Best analytic")
        axes[row, 0].plot(x, g["ordered_p00"], color="#C84B31", linestyle="--", linewidth=1.8, label="Ordered")
        axes[row, 0].set_ylabel(r"$\rho_{00}$")
        axes[row, 0].set_xlabel(xlabel)
        axes[row, 0].set_title(f"{name.capitalize()} population")
        axes[row, 0].grid(alpha=0.25)

        axes[row, 1].plot(x, g["ed_coh"], color="black", linewidth=2.2, label="ED (light)")
        axes[row, 1].plot(x, g["best_coh"], color="#0B6E4F", linewidth=2.0, label="Best analytic")
        axes[row, 1].plot(x, g["ordered_coh"], color="#C84B31", linestyle="--", linewidth=1.8, label="Ordered")
        axes[row, 1].set_ylabel(r"$|\rho_{01}|$")
        axes[row, 1].set_xlabel(xlabel)
        axes[row, 1].set_title(f"{name.capitalize()} coherence")
        axes[row, 1].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Best Analytic vs Ordered vs ED (Light) : Population and Coherence Sweeps", fontsize=12)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Best Model vs Ordered vs ED (Codex v12)")
    lines.append("")
    lines.append("ED is computed with lightweight settings: n_modes=2, n_cut=3.")
    lines.append("Ordered uses lightweight kernel settings from standalone implementation.")
    lines.append("")
    lines.append("| sweep | comparison | rmse_p00 | rmse_coh |")
    lines.append("|---|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['comparison']} | {r['rmse_p00']:.6f} | {r['rmse_coh']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


def main() -> None:
    df, summary = run_sweeps()
    write_outputs(df, summary)


if __name__ == "__main__":
    main()

