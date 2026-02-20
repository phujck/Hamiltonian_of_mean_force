"""
Scan window ratios against omega_q to identify a characteristic spectral window.

ED and analytic use the same (omega_min, omega_max) in each run.
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
    x = tau_c * omega_max
    return float(np.exp(-x) * (1.0 + x))


def _detune_ratio(omega_min: float, omega_max: float, omega_q: float, n_modes: int) -> float:
    if n_modes <= 1:
        return float(abs(0.5 * (omega_min + omega_max) - omega_q) / omega_q)
    grid = np.linspace(omega_min, omega_max, n_modes, dtype=float)
    return float(np.min(np.abs(grid - omega_q)) / omega_q)


def _build_lite(beta: float, theta: float, omega_q: float, omega_min: float, omega_max: float) -> LiteConfig:
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
        output_prefix="hmf_omegaq_scaled_window_scan_codex_v35",
    )


def run_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    theta = float(np.pi / 2.0)
    g = 0.5
    tau_c = 0.5
    ren = RenormConfig(scale=1.04, kappa=0.94)

    omega_q_vals = np.array([1.5, 2.0, 2.5, 3.0], dtype=float)
    ratio_min = 0.05
    ratio_max_vals = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    betas = np.linspace(0.6, 8.0, 9, dtype=float)

    cases = [
        EDCase(label="m3c5", n_modes=3, n_cut=5),
        EDCase(label="m4c4", n_modes=4, n_cut=4),
    ]

    ref_cache: dict[tuple[float, float, float], tuple[float, float, float, float]] = {}
    rows: list[dict[str, float | int | str]] = []

    total = len(omega_q_vals) * len(ratio_max_vals) * len(cases) * len(betas)
    done = 0
    for omega_q in omega_q_vals:
        for ratio_max in ratio_max_vals:
            omega_min = ratio_min * omega_q
            omega_max = ratio_max * omega_q
            for beta in betas:
                key = (float(omega_q), float(ratio_max), float(beta))
                if key not in ref_cache:
                    lite = _build_lite(beta, theta, omega_q, omega_min, omega_max)
                    rho_ord = ordered_gaussian_state(lite, g)
                    ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)
                    best_p00, _best_p11, best_coh, _best_ratio = _compact_components(
                        lite, g, use_running=True, renorm=ren
                    )
                    ref_cache[key] = (float(ord_p00), float(ord_coh), float(best_p00), float(best_coh))

                ord_p00, ord_coh, best_p00, best_coh = ref_cache[key]
                for case in cases:
                    cfg = _build_ed(beta, theta, omega_q, omega_min, omega_max, case)
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
                            "omega_q": float(omega_q),
                            "ratio_min": float(ratio_min),
                            "ratio_max": float(ratio_max),
                            "omega_min": float(omega_min),
                            "omega_max": float(omega_max),
                            "detune_over_omega_q": _detune_ratio(
                                omega_min=omega_min,
                                omega_max=omega_max,
                                omega_q=omega_q,
                                n_modes=case.n_modes,
                            ),
                            "tail_fraction": _tail_fraction(omega_max, tau_c),
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
                    done += 1
                    if done % 40 == 0:
                        print(f"[PROGRESS] {done}/{total}")

    df = pd.DataFrame.from_records(rows).sort_values(
        ["case", "omega_q", "ratio_max", "beta"]
    ).reset_index(drop=True)

    summary_rows: list[dict[str, float | int | str]] = []
    for (case, n_modes, n_cut, omega_q, ratio_max), grp in df.groupby(
        ["case", "n_modes", "n_cut", "omega_q", "ratio_max"]
    ):
        summary_rows.append(
            {
                "case": str(case),
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "omega_q": float(omega_q),
                "ratio_min": float(grp["ratio_min"].iloc[0]),
                "ratio_max": float(ratio_max),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(grp["omega_max"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_ed_vs_best_p00": _rmse(grp["d_ed_best_p00"].to_numpy(float)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_ord_p00"].to_numpy(float)),
                "rmse_ed_vs_best_coh": _rmse(grp["d_ed_best_coh"].to_numpy(float)),
                "rmse_ed_vs_ordered_coh": _rmse(grp["d_ed_ord_coh"].to_numpy(float)),
                "p00_at_beta2": float(
                    np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
                "p00_at_beta6": float(
                    np.interp(6.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["case", "omega_q", "ratio_max"]
    ).reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_omegaq_scaled_window_scan_codex_v35.csv"
    summary_csv = out_dir / "hmf_omegaq_scaled_window_summary_codex_v35.csv"
    fig_png = out_dir / "hmf_omegaq_scaled_window_codex_v35.png"
    log_md = out_dir / "hmf_omegaq_scaled_window_log_codex_v35.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    ax = axes[0]
    for (case, omega_q), grp in summary.groupby(["case", "omega_q"]):
        g = grp.sort_values("ratio_max")
        ax.plot(
            g["ratio_max"],
            g["rmse_ed_vs_best_p00"],
            "o-",
            linewidth=1.6,
            label=f"{case}, omega_q={omega_q:g}",
        )
    ax.set_title("RMSE(ED-best) vs ratio_max")
    ax.set_xlabel("ratio_max = omega_max / omega_q")
    ax.set_ylabel("population RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for case, grp in summary.groupby("case"):
        best = (
            grp.sort_values(["omega_q", "rmse_ed_vs_best_p00"])
            .groupby("omega_q", as_index=False)
            .first()
        )
        ax.plot(
            best["omega_q"],
            best["ratio_max"],
            "o-",
            linewidth=2.0,
            label=f"{case} best ratio_max",
        )
    ax.set_title("Best ratio_max vs omega_q")
    ax.set_xlabel("omega_q")
    ax.set_ylabel("argmin ratio_max")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Omega_q Scaled Window Scan (Codex v35)")
    lines.append("")
    lines.append("All runs use same spectral window for ED and analytic within each point.")
    lines.append("Fixed: theta=pi/2, g=0.5, ratio_min=0.05, tau_c=0.5.")
    lines.append("")
    lines.append(
        "| case | omega_q | ratio_max | detune_over_omega_q | tail_fraction | rmse_ed_vs_best_p00 | rmse_ed_vs_ordered_p00 | p00_at_beta2 | p00_at_beta6 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.sort_values(["case", "omega_q", "rmse_ed_vs_best_p00"]).iterrows():
        lines.append(
            f"| {r['case']} | {r['omega_q']:.3f} | {r['ratio_max']:.3f} | "
            f"{r['detune_over_omega_q']:.3f} | {r['tail_fraction']:.4f} | "
            f"{r['rmse_ed_vs_best_p00']:.6f} | {r['rmse_ed_vs_ordered_p00']:.6f} | "
            f"{r['p00_at_beta2']:.6f} | {r['p00_at_beta6']:.6f} |"
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

