"""
ED beta-rate convergence with analytic overlays (v12-style and raw compact).

Based on v21, but adds:
- analytic_v12: _compact_components(..., use_running=True, RenormConfig from v12)
- analytic_raw: _compact_components(..., use_running=False)
"""

from __future__ import annotations

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


def _d_from_p00(beta: float, omega_q: float, p00: float) -> float:
    a = 0.5 * beta * omega_q
    p = float(np.clip(p00, 1e-15, 1.0 - 1e-15))
    return float(a + 0.5 * np.log(p / (1.0 - p)))


def _build_ed(beta: float, theta: float, n_cut: int) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=2,
        n_cut=int(n_cut),
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_ed_beta_rate_with_analytic_codex_v26",
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_ed_beta_rate_with_analytic_scan_codex_v26.csv"
    summary_csv = out_dir / "hmf_ed_beta_rate_with_analytic_summary_codex_v26.csv"
    fig_png = out_dir / "hmf_ed_beta_rate_with_analytic_codex_v26.png"
    log_md = out_dir / "hmf_ed_beta_rate_with_analytic_log_codex_v26.md"

    betas = np.linspace(0.4, 6.0, 15)
    n_cuts = [3, 4, 5, 6, 8, 10]
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0
    ren = RenormConfig(scale=1.04, kappa=0.94)

    rows: list[dict[str, float | int | str]] = []

    # References on same beta grid.
    ref_rows: dict[float, dict[str, float]] = {}
    for beta in betas:
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
        rho_o = ordered_gaussian_state(lite, g)
        p00_o, _, coh_o = extract_density(rho_o)
        p00_v12, _p11_v12, coh_v12, _ratio_v12 = _compact_components(
            lite, g, use_running=True, renorm=ren
        )
        p00_raw, _p11_raw, coh_raw, _ratio_raw = _compact_components(
            lite, g, use_running=False, renorm=ren
        )
        ref_rows[float(beta)] = {
            "ordered_p00": float(p00_o),
            "ordered_coh": float(coh_o),
            "analytic_v12_p00": float(p00_v12),
            "analytic_v12_coh": float(coh_v12),
            "analytic_raw_p00": float(p00_raw),
            "analytic_raw_coh": float(coh_raw),
        }

    for n_cut in n_cuts:
        for beta in betas:
            cfg = _build_ed(beta=beta, theta=theta, n_cut=n_cut)
            ctx = build_ed_context(cfg)
            rho = exact_reduced_state(ctx, g)
            p00, _p11, coh = extract_density(rho)

            ref = ref_rows[float(beta)]
            rows.append(
                {
                    "beta": float(beta),
                    "n_cut": int(n_cut),
                    "ed_p00": float(p00),
                    "ed_coh": float(coh),
                    "ordered_p00": ref["ordered_p00"],
                    "ordered_coh": ref["ordered_coh"],
                    "analytic_v12_p00": ref["analytic_v12_p00"],
                    "analytic_v12_coh": ref["analytic_v12_coh"],
                    "analytic_raw_p00": ref["analytic_raw_p00"],
                    "analytic_raw_coh": ref["analytic_raw_coh"],
                    "d_ed_minus_ordered_p00": float(p00 - ref["ordered_p00"]),
                    "d_ed_minus_v12_p00": float(p00 - ref["analytic_v12_p00"]),
                    "d_ed_minus_raw_p00": float(p00 - ref["analytic_raw_p00"]),
                    "d_ed": _d_from_p00(beta, omega_q, p00),
                    "d_ordered": _d_from_p00(beta, omega_q, ref["ordered_p00"]),
                    "d_v12": _d_from_p00(beta, omega_q, ref["analytic_v12_p00"]),
                    "d_raw": _d_from_p00(beta, omega_q, ref["analytic_raw_p00"]),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["n_cut", "beta"]).reset_index(drop=True)

    # Derivative diagnostics by n_cut
    deriv_rows: list[dict[str, float | int]] = []
    for n_cut, grp in df.groupby("n_cut"):
        gdf = grp.sort_values("beta")
        b = gdf["beta"].to_numpy(dtype=float)
        h = float(b[1] - b[0])
        dp = np.gradient(gdf["ed_p00"].to_numpy(dtype=float), h)
        dd = np.gradient(gdf["d_ed"].to_numpy(dtype=float), h)
        margin = dd - 0.5 * omega_q
        deriv_rows.append(
            {
                "n_cut": int(n_cut),
                "ed_p00_beta2": float(np.interp(2.0, b, gdf["ed_p00"].to_numpy(dtype=float))),
                "dp00_max": float(np.max(dp)),
                "dp00_min": float(np.min(dp)),
                "margin_max": float(np.max(margin)),
                "margin_min": float(np.min(margin)),
                "rmse_vs_ordered_p00": float(np.sqrt(np.mean(np.square(gdf["d_ed_minus_ordered_p00"].to_numpy(dtype=float))))),
                "rmse_vs_v12_p00": float(np.sqrt(np.mean(np.square(gdf["d_ed_minus_v12_p00"].to_numpy(dtype=float))))),
                "rmse_vs_raw_p00": float(np.sqrt(np.mean(np.square(gdf["d_ed_minus_raw_p00"].to_numpy(dtype=float))))),
            }
        )
        df.loc[gdf.index, "dp00_dbeta"] = dp
        df.loc[gdf.index, "dprime_minus_w2"] = margin

    summary = pd.DataFrame.from_records(deriv_rows).sort_values("n_cut").reset_index(drop=True)

    # Analytic derivative curves from one n_cut slice references
    ref_df = df[df["n_cut"] == n_cuts[0]].sort_values("beta")
    b = ref_df["beta"].to_numpy(dtype=float)
    h = float(b[1] - b[0])
    dp_ordered = np.gradient(ref_df["ordered_p00"].to_numpy(dtype=float), h)
    dp_v12 = np.gradient(ref_df["analytic_v12_p00"].to_numpy(dtype=float), h)
    dp_raw = np.gradient(ref_df["analytic_raw_p00"].to_numpy(dtype=float), h)
    margin_ordered = np.gradient(ref_df["d_ordered"].to_numpy(dtype=float), h) - 0.5 * omega_q
    margin_v12 = np.gradient(ref_df["d_v12"].to_numpy(dtype=float), h) - 0.5 * omega_q
    margin_raw = np.gradient(ref_df["d_raw"].to_numpy(dtype=float), h) - 0.5 * omega_q

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_p00"], linewidth=1.4, label=f"ED n_cut={n_cut}")
    ax.plot(b, ref_df["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(b, ref_df["analytic_v12_p00"], color="#0B6E4F", linewidth=2.0, linestyle="--", label="Analytic v12")
    ax.plot(b, ref_df["analytic_raw_p00"], color="#AA3377", linewidth=1.8, linestyle=":", label="Analytic raw")
    ax.set_title("Population vs beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["dp00_dbeta"], linewidth=1.4, label=f"ED n_cut={n_cut}")
    ax.plot(b, dp_ordered, color="black", linewidth=2.0, label="Ordered")
    ax.plot(b, dp_v12, color="#0B6E4F", linewidth=2.0, linestyle="--", label="Analytic v12")
    ax.plot(b, dp_raw, color="#AA3377", linewidth=1.8, linestyle=":", label="Analytic raw")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("d rho_00 / d beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("derivative")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["dprime_minus_w2"], linewidth=1.4, label=f"ED n_cut={n_cut}")
    ax.plot(b, margin_ordered, color="black", linewidth=2.0, label="Ordered")
    ax.plot(b, margin_v12, color="#0B6E4F", linewidth=2.0, linestyle="--", label="Analytic v12")
    ax.plot(b, margin_raw, color="#AA3377", linewidth=1.8, linestyle=":", label="Analytic raw")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("d' - omega_q/2")
    ax.set_xlabel("beta")
    ax.set_ylabel("margin")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(summary["n_cut"], summary["ed_p00_beta2"], "o-", color="#1F4E79", linewidth=2.0, label="ED p00 @ beta=2")
    ax.axhline(float(np.interp(2.0, b, ref_df["ordered_p00"].to_numpy(dtype=float))), color="black", linewidth=1.8, label="Ordered @ beta=2")
    ax.axhline(float(np.interp(2.0, b, ref_df["analytic_v12_p00"].to_numpy(dtype=float))), color="#0B6E4F", linestyle="--", linewidth=1.8, label="Analytic v12 @ beta=2")
    ax.axhline(float(np.interp(2.0, b, ref_df["analytic_raw_p00"].to_numpy(dtype=float))), color="#AA3377", linestyle=":", linewidth=1.8, label="Analytic raw @ beta=2")
    ax.set_title("Convergence at beta=2")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# ED Beta-Rate Convergence with Analytic Overlays (Codex v26)")
    lines.append("")
    lines.append("| n_cut | ed_p00_beta2 | rmse_vs_ordered_p00 | rmse_vs_v12_p00 | rmse_vs_raw_p00 | dp00_max | dp00_min | margin_max | margin_min |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_cut'])} | {r['ed_p00_beta2']:.6f} | {r['rmse_vs_ordered_p00']:.6f} | {r['rmse_vs_v12_p00']:.6f} | {r['rmse_vs_raw_p00']:.6f} | "
            f"{r['dp00_max']:.6f} | {r['dp00_min']:.6f} | {r['margin_max']:.6f} | {r['margin_min']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
