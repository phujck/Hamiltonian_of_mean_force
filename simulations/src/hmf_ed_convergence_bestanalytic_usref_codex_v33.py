"""
High-beta ED overlay using the previously converged analytic branch
plus ultrastrong-limit reference lines.

No new ED points are computed.
Input:
  hmf_ed_convergence_highbeta_lomodes_codex_v30_scan.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import BenchmarkConfig as LiteConfig, RenormConfig
from prl127_qubit_benchmark import SIGMA_X, SIGMA_Z, ultrastrong_projected_state


def _rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr)))) if len(arr) else np.nan


def _ultrastrong_components(beta: float, theta: float, omega_q: float) -> tuple[float, float]:
    h_s = 0.5 * omega_q * SIGMA_Z
    x_op = np.cos(theta) * SIGMA_Z - np.sin(theta) * SIGMA_X
    rho_us = ultrastrong_projected_state(h_s, x_op, beta)
    p00 = float(np.real(rho_us[0, 0]))
    coh = float(abs(rho_us[0, 1]))
    return p00, coh


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src_csv = out_dir / "hmf_ed_convergence_highbeta_lomodes_codex_v30_scan.csv"
    out_scan_csv = out_dir / "hmf_ed_convergence_bestanalytic_usref_scan_codex_v33.csv"
    out_summary_csv = out_dir / "hmf_ed_convergence_bestanalytic_usref_summary_codex_v33.csv"
    out_fig_png = out_dir / "hmf_ed_convergence_bestanalytic_usref_codex_v33.png"
    out_log_md = out_dir / "hmf_ed_convergence_bestanalytic_usref_log_codex_v33.md"

    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source scan: {src_csv}")

    df = pd.read_csv(src_csv).copy()
    df = df[df["status"] == "ok"].copy()
    df = df.sort_values(["n_modes", "n_cut", "beta"]).reset_index(drop=True)

    theta = float(df["theta"].iloc[0])
    g = float(df["g"].iloc[0])
    omega_q = 2.0

    # Previously converged analytic branch used in v12/v30.
    ren = RenormConfig(scale=1.04, kappa=0.94)
    beta_vals = sorted(df["beta"].unique().tolist())
    ref_rows: list[dict[str, float]] = []
    for beta in beta_vals:
        lite = LiteConfig(
            beta=float(beta),
            omega_q=omega_q,
            theta=theta,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        p00_best, _p11_best, coh_best, _ratio_best = _compact_components(
            lite, g, use_running=True, renorm=ren
        )
        p00_us, coh_us = _ultrastrong_components(beta=float(beta), theta=theta, omega_q=omega_q)
        ref_rows.append(
            {
                "beta": float(beta),
                "analytic_best_p00": float(p00_best),
                "analytic_best_coh": float(coh_best),
                "ultrastrong_p00": float(p00_us),
                "ultrastrong_coh": float(coh_us),
            }
        )
    ref = pd.DataFrame.from_records(ref_rows)
    df = df.merge(ref, on="beta", how="left")

    summary_rows: list[dict[str, float | int]] = []
    for (n_modes, n_cut), grp in df.groupby(["n_modes", "n_cut"]):
        gdf = grp.sort_values("beta")
        summary_rows.append(
            {
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "n_points": int(len(gdf)),
                "rmse_ed_vs_ordered_p00": _rmse(
                    gdf["ed_p00"].to_numpy(float) - gdf["ordered_p00"].to_numpy(float)
                ),
                "rmse_ed_vs_best_p00": _rmse(
                    gdf["ed_p00"].to_numpy(float) - gdf["analytic_best_p00"].to_numpy(float)
                ),
                "rmse_ed_vs_us_p00": _rmse(
                    gdf["ed_p00"].to_numpy(float) - gdf["ultrastrong_p00"].to_numpy(float)
                ),
                "rmse_ed_vs_ordered_coh": _rmse(
                    gdf["ed_coh"].to_numpy(float) - gdf["ordered_coh"].to_numpy(float)
                ),
                "rmse_ed_vs_best_coh": _rmse(
                    gdf["ed_coh"].to_numpy(float) - gdf["analytic_best_coh"].to_numpy(float)
                ),
                "rmse_ed_vs_us_coh": _rmse(
                    gdf["ed_coh"].to_numpy(float) - gdf["ultrastrong_coh"].to_numpy(float)
                ),
                "p00_at_beta2": float(
                    np.interp(2.0, gdf["beta"].to_numpy(float), gdf["ed_p00"].to_numpy(float))
                ),
                "p00_at_beta10": float(
                    np.interp(10.0, gdf["beta"].to_numpy(float), gdf["ed_p00"].to_numpy(float))
                ),
            }
        )

    summary = (
        pd.DataFrame.from_records(summary_rows)
        .sort_values(["n_modes", "n_cut"])
        .reset_index(drop=True)
    )

    df.to_csv(out_scan_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)

    ax = axes[0]
    for (n_modes, n_cut), grp in df.groupby(["n_modes", "n_cut"]):
        gdf = grp.sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_p00"], linewidth=1.2, label=f"ED m{int(n_modes)}/c{int(n_cut)}")
    ref_line = df.sort_values("beta").drop_duplicates(subset=["beta"])
    ax.plot(ref_line["beta"], ref_line["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(
        ref_line["beta"],
        ref_line["analytic_best_p00"],
        color="#0B6E4F",
        linestyle="--",
        linewidth=2.0,
        label="Analytic best (running)",
    )
    ax.plot(
        ref_line["beta"],
        ref_line["ultrastrong_p00"],
        color="#AA3377",
        linestyle=":",
        linewidth=2.0,
        label="Ultrastrong limit",
    )
    ax.set_title("Population with ultrastrong reference")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for (n_modes, n_cut), grp in df.groupby(["n_modes", "n_cut"]):
        gdf = grp.sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_coh"], linewidth=1.2, label=f"ED m{int(n_modes)}/c{int(n_cut)}")
    ax.plot(ref_line["beta"], ref_line["ordered_coh"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(
        ref_line["beta"],
        ref_line["analytic_best_coh"],
        color="#0B6E4F",
        linestyle="--",
        linewidth=2.0,
        label="Analytic best (running)",
    )
    ax.plot(
        ref_line["beta"],
        ref_line["ultrastrong_coh"],
        color="#AA3377",
        linestyle=":",
        linewidth=2.0,
        label="Ultrastrong limit",
    )
    ax.set_title("Coherence with ultrastrong reference")
    ax.set_xlabel("beta")
    ax.set_ylabel("|rho_01|")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.savefig(out_fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# High-Beta ED vs Best Analytic + Ultrastrong (Codex v33)")
    lines.append("")
    lines.append("Analytic branch used: `_compact_components(..., use_running=True, RenormConfig(scale=1.04, kappa=0.94))`.")
    lines.append("Ultrastrong reference from PRL projected state is included as an additional line.")
    lines.append("")
    lines.append(
        "| n_modes | n_cut | n_points | rmse_ed_vs_ordered_p00 | rmse_ed_vs_best_p00 | rmse_ed_vs_us_p00 | rmse_ed_vs_ordered_coh | rmse_ed_vs_best_coh | rmse_ed_vs_us_coh | p00_at_beta2 | p00_at_beta10 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_modes'])} | {int(r['n_cut'])} | {int(r['n_points'])} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {r['rmse_ed_vs_best_p00']:.6f} | {r['rmse_ed_vs_us_p00']:.6f} | "
            f"{r['rmse_ed_vs_ordered_coh']:.6e} | {r['rmse_ed_vs_best_coh']:.6e} | {r['rmse_ed_vs_us_coh']:.6e} | "
            f"{r['p00_at_beta2']:.6f} | {r['p00_at_beta10']:.6f} |"
        )
    out_log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_scan_csv.name)
    print("Wrote:", out_summary_csv.name)
    print("Wrote:", out_fig_png.name)
    print("Wrote:", out_log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

