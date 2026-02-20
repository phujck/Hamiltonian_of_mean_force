"""
Overlay pure scale-bounded analytic components on existing high-beta ED scan.

No new ED points are computed.
Input:
  hmf_ed_convergence_highbeta_lomodes_codex_v30_scan.csv

Analytic used here:
  _compact_components(..., use_running=False)
which is the direct analytically normalized-component form with no running cap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import BenchmarkConfig as LiteConfig, RenormConfig


def _rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr)))) if len(arr) else np.nan


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src_csv = out_dir / "hmf_ed_convergence_highbeta_lomodes_codex_v30_scan.csv"
    out_scan_csv = out_dir / "hmf_ed_convergence_highbeta_scalebound_overlay_scan_codex_v31.csv"
    out_summary_csv = out_dir / "hmf_ed_convergence_highbeta_scalebound_overlay_summary_codex_v31.csv"
    out_fig_png = out_dir / "hmf_ed_convergence_highbeta_scalebound_overlay_codex_v31.png"
    out_log_md = out_dir / "hmf_ed_convergence_highbeta_scalebound_overlay_log_codex_v31.md"

    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source scan: {src_csv}")

    df = pd.read_csv(src_csv).copy()
    df = df[df["status"] == "ok"].copy()
    df = df.sort_values(["n_modes", "n_cut", "beta"]).reset_index(drop=True)

    theta = float(df["theta"].iloc[0])
    g = float(df["g"].iloc[0])
    omega_q = 2.0

    # Note: renorm params are ignored when use_running=False, kept explicit for clarity.
    ren = RenormConfig(scale=1.0, kappa=1.0)
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
        p00, _p11, coh, _ratio = _compact_components(lite, g, use_running=False, renorm=ren)
        ref_rows.append(
            {
                "beta": float(beta),
                "analytic_scalebound_p00": float(p00),
                "analytic_scalebound_coh": float(coh),
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
                "rmse_ed_vs_scalebound_p00": _rmse(
                    gdf["ed_p00"].to_numpy(float)
                    - gdf["analytic_scalebound_p00"].to_numpy(float)
                ),
                "rmse_ed_vs_ordered_coh": _rmse(
                    gdf["ed_coh"].to_numpy(float) - gdf["ordered_coh"].to_numpy(float)
                ),
                "rmse_ed_vs_scalebound_coh": _rmse(
                    gdf["ed_coh"].to_numpy(float)
                    - gdf["analytic_scalebound_coh"].to_numpy(float)
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    ax = axes[0]
    for (n_modes, n_cut), grp in df.groupby(["n_modes", "n_cut"]):
        gdf = grp.sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_p00"], linewidth=1.2, label=f"ED m{int(n_modes)}/c{int(n_cut)}")
    ref_line = df.sort_values("beta").drop_duplicates(subset=["beta"])
    ax.plot(ref_line["beta"], ref_line["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(
        ref_line["beta"],
        ref_line["analytic_scalebound_p00"],
        color="#0B6E4F",
        linestyle="--",
        linewidth=2.0,
        label="Analytic scale-bound",
    )
    ax.set_title("Population: ED vs Ordered vs Scale-bound Analytic")
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
        ref_line["analytic_scalebound_coh"],
        color="#0B6E4F",
        linestyle="--",
        linewidth=2.0,
        label="Analytic scale-bound",
    )
    ax.set_title("Coherence: ED vs Ordered vs Scale-bound Analytic")
    ax.set_xlabel("beta")
    ax.set_ylabel("|rho_01|")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.savefig(out_fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# High-Beta Scale-Bound Analytic Overlay (Codex v31)")
    lines.append("")
    lines.append(
        "Analytic curve is computed with `_compact_components(..., use_running=False)` only."
    )
    lines.append("")
    lines.append(
        "| n_modes | n_cut | n_points | rmse_ed_vs_ordered_p00 | rmse_ed_vs_scalebound_p00 | rmse_ed_vs_ordered_coh | rmse_ed_vs_scalebound_coh | p00_at_beta2 | p00_at_beta10 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_modes'])} | {int(r['n_cut'])} | {int(r['n_points'])} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {r['rmse_ed_vs_scalebound_p00']:.6f} | "
            f"{r['rmse_ed_vs_ordered_coh']:.6f} | {r['rmse_ed_vs_scalebound_coh']:.6f} | "
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

