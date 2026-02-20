"""
Characterize which spectral window works best when ED and analytic share the same window.

Outputs:
- scan CSV with per-point ED/ordered/analytic-best comparisons
- summary CSV with RMSE vs window width and resonance alignment
- diagnostic plot
- markdown log

Run with:
  powershell -ExecutionPolicy Bypass -File run_safe.ps1 simulations/src/hmf_spectral_window_characterization_codex_v34.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
    ordered_gaussian_state,
)
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig


@dataclass(frozen=True)
class EDCase:
    label: str
    n_modes: int
    n_cut: int


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def _tail_fraction(omega_max: float, tau_c: float) -> float:
    # For J(w) ~ tau*w*exp(-tau*w): fraction of area above omega_max.
    x = tau_c * omega_max
    return float(np.exp(-x) * (1.0 + x))


def _nearest_mode_detune(omega_min: float, omega_max: float, omega_q: float, n_modes: int) -> tuple[float, float]:
    if n_modes <= 1:
        w = 0.5 * (omega_min + omega_max)
        return abs(w - omega_q), omega_max - omega_min
    grid = np.linspace(omega_min, omega_max, n_modes, dtype=float)
    dw = float(grid[1] - grid[0])
    detune = float(np.min(np.abs(grid - omega_q)))
    return detune, dw


def _build_lite(
    beta: float,
    theta: float,
    omega_q: float,
    omega_min: float,
    omega_max: float,
) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=40,
        n_cut=1,
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        q_strength=5.0,
        tau_c=0.5,
    )


def _build_ed(
    beta: float,
    theta: float,
    omega_q: float,
    omega_min: float,
    omega_max: float,
    case: EDCase,
) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=int(case.n_modes),
        n_cut=int(case.n_cut),
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_spectral_window_characterization_codex_v34",
    )


def run_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Fixed physics for this characterization.
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0
    omega_min = 0.1
    tau_c = 0.5
    betas = np.linspace(0.6, 10.0, 11, dtype=float)
    omega_max_vals = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0], dtype=float)
    ren = RenormConfig(scale=1.04, kappa=0.94)

    cases = [
        EDCase(label="m3c5", n_modes=3, n_cut=5),
        EDCase(label="m4c4", n_modes=4, n_cut=4),
    ]

    rows: list[dict[str, float | int | str]] = []
    ref_cache: dict[tuple[float, float], tuple[float, float, float, float]] = {}

    for omega_max in omega_max_vals:
        for beta in betas:
            ref_key = (float(beta), float(omega_max))
            if ref_key not in ref_cache:
                lite = _build_lite(beta, theta, omega_q, omega_min, omega_max)
                rho_ord = ordered_gaussian_state(lite, g)
                ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)
                best_p00, _best_p11, best_coh, _best_ratio = _compact_components(
                    lite, g, use_running=True, renorm=ren
                )
                ref_cache[ref_key] = (float(ord_p00), float(ord_coh), float(best_p00), float(best_coh))

            ord_p00, ord_coh, best_p00, best_coh = ref_cache[ref_key]

            for case in cases:
                detune, dw = _nearest_mode_detune(
                    omega_min=omega_min,
                    omega_max=float(omega_max),
                    omega_q=omega_q,
                    n_modes=case.n_modes,
                )
                cfg = _build_ed(
                    beta=float(beta),
                    theta=theta,
                    omega_q=omega_q,
                    omega_min=omega_min,
                    omega_max=float(omega_max),
                    case=case,
                )
                ctx = build_ed_context(cfg)
                rho_ed = exact_reduced_state(ctx, g)
                ed_p00, _ed_p11, ed_coh = extract_density(rho_ed)

                rows.append(
                    {
                        "case": case.label,
                        "n_modes": int(case.n_modes),
                        "n_cut": int(case.n_cut),
                        "beta": float(beta),
                        "g": g,
                        "theta": theta,
                        "omega_q": omega_q,
                        "omega_min": omega_min,
                        "omega_max": float(omega_max),
                        "window_ratio_min": float(omega_min / omega_q),
                        "window_ratio_max": float(omega_max / omega_q),
                        "detune_abs": float(detune),
                        "detune_over_omega_q": float(detune / omega_q),
                        "detune_over_dw": float(detune / max(dw, 1e-12)),
                        "dw": float(dw),
                        "tail_fraction": _tail_fraction(float(omega_max), tau_c),
                        "ordered_p00": ord_p00,
                        "ordered_coh": ord_coh,
                        "best_p00": best_p00,
                        "best_coh": best_coh,
                        "ed_p00": float(ed_p00),
                        "ed_coh": float(ed_coh),
                        "d_ed_ord_p00": float(ed_p00 - ord_p00),
                        "d_ed_best_p00": float(ed_p00 - best_p00),
                        "d_ed_ord_coh": float(ed_coh - ord_coh),
                        "d_ed_best_coh": float(ed_coh - best_coh),
                    }
                )

    df = pd.DataFrame.from_records(rows).sort_values(["case", "omega_max", "beta"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for (case, n_modes, n_cut, omega_max), grp in df.groupby(["case", "n_modes", "n_cut", "omega_max"]):
        summary_rows.append(
            {
                "case": str(case),
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "omega_q": float(grp["omega_q"].iloc[0]),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(omega_max),
                "window_ratio_min": float(grp["window_ratio_min"].iloc[0]),
                "window_ratio_max": float(grp["window_ratio_max"].iloc[0]),
                "dw": float(grp["dw"].iloc[0]),
                "detune_abs": float(grp["detune_abs"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "detune_over_dw": float(grp["detune_over_dw"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_ord_p00"].to_numpy(float)),
                "rmse_ed_vs_best_p00": _rmse(grp["d_ed_best_p00"].to_numpy(float)),
                "rmse_ed_vs_ordered_coh": _rmse(grp["d_ed_ord_coh"].to_numpy(float)),
                "rmse_ed_vs_best_coh": _rmse(grp["d_ed_best_coh"].to_numpy(float)),
                "p00_at_beta2": float(
                    np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
                "p00_at_beta8": float(
                    np.interp(8.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
            }
        )
    summary = (
        pd.DataFrame.from_records(summary_rows)
        .sort_values(["case", "omega_max"])
        .reset_index(drop=True)
    )
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_spectral_window_characterization_scan_codex_v34.csv"
    summary_csv = out_dir / "hmf_spectral_window_characterization_summary_codex_v34.csv"
    fig_png = out_dir / "hmf_spectral_window_characterization_codex_v34.png"
    log_md = out_dir / "hmf_spectral_window_characterization_log_codex_v34.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axes[0, 0]
    for case, grp in summary.groupby("case"):
        g = grp.sort_values("omega_max")
        ax.plot(g["omega_max"], g["rmse_ed_vs_best_p00"], "o-", linewidth=1.8, label=f"{case} vs best")
        ax.plot(g["omega_max"], g["rmse_ed_vs_ordered_p00"], "s--", linewidth=1.4, label=f"{case} vs ordered")
    ax.set_title("Population RMSE vs omega_max")
    ax.set_xlabel("omega_max")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    for case, grp in summary.groupby("case"):
        g = grp.sort_values("omega_max")
        ax.plot(g["omega_max"], g["detune_over_omega_q"], "o-", linewidth=1.8, label=f"{case}: detune/omega_q")
        ax.plot(g["omega_max"], g["tail_fraction"], "s--", linewidth=1.4, label=f"{case}: tail frac")
    ax.set_title("Resonance alignment and tail")
    ax.set_xlabel("omega_max")
    ax.set_ylabel("dimensionless")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 0]
    for case, grp in summary.groupby("case"):
        g = grp.sort_values("detune_over_omega_q")
        ax.plot(g["detune_over_omega_q"], g["rmse_ed_vs_best_p00"], "o-", linewidth=1.8, label=case)
    ax.set_title("Population RMSE vs resonance detune")
    ax.set_xlabel("min|omega_k-omega_q| / omega_q")
    ax.set_ylabel("RMSE(ED-best)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for case, grp in df.groupby("case"):
        best_row = (
            summary[summary["case"] == case]
            .sort_values("rmse_ed_vs_best_p00")
            .iloc[0]
        )
        best_omax = float(best_row["omega_max"])
        sub = grp[grp["omega_max"] == best_omax].sort_values("beta")
        ax.plot(sub["beta"], sub["ed_p00"], linewidth=1.8, label=f"ED {case} (best omax={best_omax:g})")
        ax.plot(sub["beta"], sub["best_p00"], linestyle="--", linewidth=1.8, label=f"Best analytic {case}")
        ax.plot(sub["beta"], sub["ordered_p00"], linestyle=":", linewidth=1.8, label=f"Ordered {case}")
    ax.set_title("Best-window beta sweep per case")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Spectral Window Characterization (Codex v34)")
    lines.append("")
    lines.append("ED and analytic are evaluated on the same spectral window in each run.")
    lines.append("Fixed setup: theta=pi/2, g=0.5, omega_q=2.0, omega_min=0.1, tau_c=0.5.")
    lines.append("")
    lines.append(
        "| case | n_modes | n_cut | omega_max | window_ratio_max | detune_over_omega_q | detune_over_dw | tail_fraction | rmse_ed_vs_best_p00 | rmse_ed_vs_ordered_p00 | p00_at_beta2 | p00_at_beta8 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.sort_values(["case", "rmse_ed_vs_best_p00"]).iterrows():
        lines.append(
            f"| {r['case']} | {int(r['n_modes'])} | {int(r['n_cut'])} | {r['omega_max']:.3f} | "
            f"{r['window_ratio_max']:.3f} | {r['detune_over_omega_q']:.3f} | {r['detune_over_dw']:.3f} | "
            f"{r['tail_fraction']:.4f} | {r['rmse_ed_vs_best_p00']:.6f} | {r['rmse_ed_vs_ordered_p00']:.6f} | "
            f"{r['p00_at_beta2']:.6f} | {r['p00_at_beta8']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)


def main() -> None:
    df, summary = run_scan()
    write_outputs(df, summary)


if __name__ == "__main__":
    main()

