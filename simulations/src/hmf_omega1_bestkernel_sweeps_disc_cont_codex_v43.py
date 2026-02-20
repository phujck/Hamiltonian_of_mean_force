"""
Parameter sweeps for the selected fair-kernel setup:
  omega_q=1, n_modes=2, n_cut=6, omega in [0.2, 1.8]

Compares:
- ED (exact reduced state)
- Analytic solution with matching discrete kernel (n_modes=2)
- Analytic solution with continuum-like kernel (n_modes=40)

Sweeps:
- coupling g
- inverse temperature beta
- angle theta
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
)
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig


OUT_STEM = "hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43"


@dataclass(frozen=True)
class SweepDef:
    name: str
    param_name: str
    param_values: np.ndarray
    beta_fixed: float | None
    theta_fixed: float | None
    g_fixed: float | None


def _rmse(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else np.nan


def _build_lite(beta: float, theta: float, n_modes: int) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=1.0,
        theta=float(theta),
        n_modes=int(n_modes),
        n_cut=1,
        omega_min=0.2,
        omega_max=1.8,
        q_strength=5.0,
        tau_c=0.5,
    )


def _build_ed(beta: float, theta: float) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=1.0,
        theta=float(theta),
        n_modes=2,
        n_cut=6,
        omega_min=0.2,
        omega_max=1.8,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix=OUT_STEM,
    )


def run_sweeps() -> tuple[pd.DataFrame, pd.DataFrame]:
    ren = RenormConfig(scale=1.04, kappa=0.94)

    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.linspace(0.0, 2.0, 21),
            beta_fixed=2.0,
            theta_fixed=float(np.pi / 4.0),
            g_fixed=None,
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.linspace(0.2, 10.0, 25),
            beta_fixed=None,
            theta_fixed=float(np.pi / 2.0),
            g_fixed=0.5,
        ),
        SweepDef(
            name="angle",
            param_name="theta",
            param_values=np.linspace(0.0, float(np.pi / 2.0), 21),
            beta_fixed=2.0,
            theta_fixed=None,
            g_fixed=0.5,
        ),
    ]

    rows: list[dict[str, float | str]] = []
    total = sum(len(s.param_values) for s in sweeps)
    k = 0

    for sw in sweeps:
        for p in sw.param_values:
            beta = float(p) if sw.param_name == "beta" else float(sw.beta_fixed)
            theta = float(p) if sw.param_name == "theta" else float(sw.theta_fixed)
            g = float(p) if sw.param_name == "g" else float(sw.g_fixed)

            lite_disc = _build_lite(beta=beta, theta=theta, n_modes=2)
            disc_p00, disc_p11, disc_coh, _disc_ratio = _compact_components(
                lite_disc, g, use_running=True, renorm=ren
            )

            lite_cont = _build_lite(beta=beta, theta=theta, n_modes=40)
            cont_p00, cont_p11, cont_coh, _cont_ratio = _compact_components(
                lite_cont, g, use_running=True, renorm=ren
            )

            ed_cfg = _build_ed(beta=beta, theta=theta)
            ed_ctx = build_ed_context(ed_cfg)
            rho_ed = exact_reduced_state(ed_ctx, g)
            ed_p00, ed_p11, ed_coh = extract_density(rho_ed)

            rows.append(
                {
                    "sweep": sw.name,
                    "param_name": sw.param_name,
                    "param": float(p),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "ed_p00": float(ed_p00),
                    "ed_p11": float(ed_p11),
                    "ed_coh": float(ed_coh),
                    "analytic_disc_p00": float(disc_p00),
                    "analytic_disc_p11": float(disc_p11),
                    "analytic_disc_coh": float(disc_coh),
                    "analytic_cont_p00": float(cont_p00),
                    "analytic_cont_p11": float(cont_p11),
                    "analytic_cont_coh": float(cont_coh),
                    "d_disc_p00": float(ed_p00 - disc_p00),
                    "d_disc_coh": float(ed_coh - disc_coh),
                    "d_cont_p00": float(ed_p00 - cont_p00),
                    "d_cont_coh": float(ed_coh - cont_coh),
                }
            )
            k += 1
            if k % 20 == 0:
                print(f"[PROGRESS] {k}/{total}")

    df = pd.DataFrame.from_records(rows).sort_values(["sweep", "param"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "rmse_disc_p00": _rmse(grp["d_disc_p00"]),
                "rmse_cont_p00": _rmse(grp["d_cont_p00"]),
                "rmse_disc_coh": _rmse(grp["d_disc_coh"]),
                "rmse_cont_coh": _rmse(grp["d_cont_coh"]),
                "max_abs_disc_dp00": float(np.max(np.abs(grp["d_disc_p00"]))),
                "max_abs_cont_dp00": float(np.max(np.abs(grp["d_cont_p00"]))),
                "max_abs_disc_dcoh": float(np.max(np.abs(grp["d_disc_coh"]))),
                "max_abs_cont_dcoh": float(np.max(np.abs(grp["d_cont_coh"]))),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("sweep").reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / f"{OUT_STEM}_scan.csv"
    summary_csv = out_dir / f"{OUT_STEM}_summary.csv"
    fig_png = out_dir / f"{OUT_STEM}.png"
    log_md = out_dir / f"{OUT_STEM}_log.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.2), constrained_layout=True)
    order = ["coupling", "temperature", "angle"]
    for i, name in enumerate(order):
        g = df[df["sweep"] == name].sort_values("param").copy()
        x = g["param"].to_numpy(dtype=float)
        xlabel = g["param_name"].iloc[0]
        if name == "angle":
            x = x / np.pi
            xlabel = "theta/pi"

        row = summary[summary["sweep"] == name].iloc[0]
        rmse_disc_p = float(row["rmse_disc_p00"])
        rmse_cont_p = float(row["rmse_cont_p00"])
        rmse_disc_c = float(row["rmse_disc_coh"])
        rmse_cont_c = float(row["rmse_cont_coh"])

        axes[i, 0].plot(x, g["ed_p00"], color="black", linewidth=2.0, label="ED (m=2,c=6)")
        axes[i, 0].plot(
            x,
            g["analytic_disc_p00"],
            color="#0B6E4F",
            linestyle="--",
            linewidth=2.0,
            label="Analytic discrete",
        )
        axes[i, 0].plot(
            x,
            g["analytic_cont_p00"],
            color="#1F4E79",
            linestyle=":",
            linewidth=2.2,
            label="Analytic continuous",
        )
        axes[i, 0].set_title(
            f"{name.capitalize()} population (RMSE disc={rmse_disc_p:.4f}, cont={rmse_cont_p:.4f})"
        )
        axes[i, 0].set_xlabel(xlabel)
        axes[i, 0].set_ylabel("rho_00")
        axes[i, 0].grid(alpha=0.25)

        axes[i, 1].plot(x, g["ed_coh"], color="black", linewidth=2.0, label="ED (m=2,c=6)")
        axes[i, 1].plot(
            x,
            g["analytic_disc_coh"],
            color="#0B6E4F",
            linestyle="--",
            linewidth=2.0,
            label="Analytic discrete",
        )
        axes[i, 1].plot(
            x,
            g["analytic_cont_coh"],
            color="#1F4E79",
            linestyle=":",
            linewidth=2.2,
            label="Analytic continuous",
        )
        axes[i, 1].set_title(
            f"{name.capitalize()} coherence (RMSE disc={rmse_disc_c:.2e}, cont={rmse_cont_c:.2e})"
        )
        axes[i, 1].set_xlabel(xlabel)
        axes[i, 1].set_ylabel("|rho_01|")
        axes[i, 1].grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Omega_q=1 Best-Kernel Sweeps: Discrete vs Continuous Analytic (Codex v43)")
    lines.append("")
    lines.append("Setup fixed for fairness:")
    lines.append("- omega_q=1, n_modes=2, n_cut=6, omega window=[0.2,1.8]")
    lines.append("- ED uses the same 2-mode window.")
    lines.append("- Analytic discrete: n_modes=2 on this same window.")
    lines.append("- Analytic continuous: n_modes=40 on this same window.")
    lines.append("")
    lines.append(
        "| sweep | rmse_disc_p00 | rmse_cont_p00 | rmse_disc_coh | rmse_cont_coh | max_abs_disc_dp00 | max_abs_cont_dp00 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['rmse_disc_p00']:.6f} | {r['rmse_cont_p00']:.6f} | "
            f"{r['rmse_disc_coh']:.6e} | {r['rmse_cont_coh']:.6e} | "
            f"{r['max_abs_disc_dp00']:.6f} | {r['max_abs_cont_dp00']:.6f} |"
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
