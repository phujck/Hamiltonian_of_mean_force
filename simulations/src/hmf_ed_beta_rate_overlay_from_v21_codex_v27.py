"""
Overlay analytic curves onto existing v21 ED convergence data.

This script does NOT compute new ED points.
It reads:
  hmf_ed_beta_rate_convergence_scan_codex_v21.csv
and adds analytic references from the v12 pipeline:
  - analytic_v12: _compact_components(..., use_running=True)
  - analytic_raw: _compact_components(..., use_running=False)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import BenchmarkConfig as LiteConfig, RenormConfig


def _d_from_p00(beta: float, omega_q: float, p00: float) -> float:
    a = 0.5 * beta * omega_q
    p = float(np.clip(p00, 1e-15, 1.0 - 1e-15))
    return float(a + 0.5 * np.log(p / (1.0 - p)))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    src_csv = out_dir / "hmf_ed_beta_rate_convergence_scan_codex_v21.csv"
    out_scan_csv = out_dir / "hmf_ed_beta_rate_overlay_from_v21_scan_codex_v27.csv"
    out_summary_csv = out_dir / "hmf_ed_beta_rate_overlay_from_v21_summary_codex_v27.csv"
    out_fig_png = out_dir / "hmf_ed_beta_rate_overlay_from_v21_codex_v27.png"
    out_log_md = out_dir / "hmf_ed_beta_rate_overlay_from_v21_log_codex_v27.md"

    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source scan: {src_csv}")

    df = pd.read_csv(src_csv).copy()
    df = df.sort_values(["n_cut", "beta"]).reset_index(drop=True)

    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0
    ren = RenormConfig(scale=1.04, kappa=0.94)

    # Build analytic references once per beta.
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
        p00_v12, _p11_v12, coh_v12, _ratio_v12 = _compact_components(
            lite, g, use_running=True, renorm=ren
        )
        p00_raw, _p11_raw, coh_raw, _ratio_raw = _compact_components(
            lite, g, use_running=False, renorm=ren
        )
        ref_rows.append(
            {
                "beta": float(beta),
                "analytic_v12_p00": float(p00_v12),
                "analytic_v12_coh": float(coh_v12),
                "analytic_raw_p00": float(p00_raw),
                "analytic_raw_coh": float(coh_raw),
                "d_v12": _d_from_p00(beta, omega_q, p00_v12),
                "d_raw": _d_from_p00(beta, omega_q, p00_raw),
            }
        )

    ref = pd.DataFrame.from_records(ref_rows)
    df = df.merge(ref, on="beta", how="left")

    # Ensure derivatives are available/recomputed consistently.
    n_cuts = sorted(df["n_cut"].unique().tolist())
    deriv_rows: list[dict[str, float | int]] = []
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        b = gdf["beta"].to_numpy(dtype=float)
        h = float(b[1] - b[0])

        if "dp00_dbeta" not in df.columns:
            dp_ed = np.gradient(gdf["ed_p00"].to_numpy(dtype=float), h)
            df.loc[gdf.index, "dp00_dbeta"] = dp_ed
        if "dprime_minus_w2" not in df.columns:
            dprime = np.gradient(gdf["d_ed"].to_numpy(dtype=float), h) - 0.5 * omega_q
            df.loc[gdf.index, "dprime_minus_w2"] = dprime

        diff_ord = gdf["ed_p00"].to_numpy(dtype=float) - gdf["ordered_p00"].to_numpy(dtype=float)
        diff_v12 = gdf["ed_p00"].to_numpy(dtype=float) - gdf["analytic_v12_p00"].to_numpy(dtype=float)
        diff_raw = gdf["ed_p00"].to_numpy(dtype=float) - gdf["analytic_raw_p00"].to_numpy(dtype=float)

        deriv_rows.append(
            {
                "n_cut": int(n_cut),
                "ed_p00_beta2": float(np.interp(2.0, b, gdf["ed_p00"].to_numpy(dtype=float))),
                "rmse_vs_ordered_p00": float(np.sqrt(np.mean(np.square(diff_ord)))),
                "rmse_vs_v12_p00": float(np.sqrt(np.mean(np.square(diff_v12)))),
                "rmse_vs_raw_p00": float(np.sqrt(np.mean(np.square(diff_raw)))),
            }
        )

    summary = pd.DataFrame.from_records(deriv_rows).sort_values("n_cut").reset_index(drop=True)

    # Analytic derivative overlays from beta-only reference.
    ref_line = ref.sort_values("beta")
    b = ref_line["beta"].to_numpy(dtype=float)
    h = float(b[1] - b[0])
    dp_ordered = np.gradient(
        df[df["n_cut"] == n_cuts[0]].sort_values("beta")["ordered_p00"].to_numpy(dtype=float), h
    )
    dp_v12 = np.gradient(ref_line["analytic_v12_p00"].to_numpy(dtype=float), h)
    dp_raw = np.gradient(ref_line["analytic_raw_p00"].to_numpy(dtype=float), h)
    margin_ordered = (
        np.gradient(
            df[df["n_cut"] == n_cuts[0]].sort_values("beta")["d_ordered"].to_numpy(dtype=float), h
        )
        - 0.5 * omega_q
    )
    margin_v12 = np.gradient(ref_line["d_v12"].to_numpy(dtype=float), h) - 0.5 * omega_q
    margin_raw = np.gradient(ref_line["d_raw"].to_numpy(dtype=float), h) - 0.5 * omega_q

    df.to_csv(out_scan_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_p00"], linewidth=1.4, label=f"ED n_cut={n_cut}")
    ordered_line = df[df["n_cut"] == n_cuts[0]].sort_values("beta")
    ax.plot(ordered_line["beta"], ordered_line["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(ref_line["beta"], ref_line["analytic_v12_p00"], color="#0B6E4F", linewidth=2.0, linestyle="--", label="Analytic v12")
    ax.plot(ref_line["beta"], ref_line["analytic_raw_p00"], color="#AA3377", linewidth=1.8, linestyle=":", label="Analytic raw")
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
    ax.axhline(float(np.interp(2.0, b, ordered_line["ordered_p00"].to_numpy(dtype=float))), color="black", linewidth=1.8, label="Ordered @ beta=2")
    ax.axhline(float(np.interp(2.0, b, ref_line["analytic_v12_p00"].to_numpy(dtype=float))), color="#0B6E4F", linestyle="--", linewidth=1.8, label="Analytic v12 @ beta=2")
    ax.axhline(float(np.interp(2.0, b, ref_line["analytic_raw_p00"].to_numpy(dtype=float))), color="#AA3377", linestyle=":", linewidth=1.8, label="Analytic raw @ beta=2")
    ax.set_title("Convergence at beta=2")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# ED Overlay From v21 (Codex v27)")
    lines.append("")
    lines.append("No new ED was computed; this overlays analytic curves on existing v21 scan.")
    lines.append("")
    lines.append("| n_cut | ed_p00_beta2 | rmse_vs_ordered_p00 | rmse_vs_v12_p00 | rmse_vs_raw_p00 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_cut'])} | {r['ed_p00_beta2']:.6f} | {r['rmse_vs_ordered_p00']:.6f} | "
            f"{r['rmse_vs_v12_p00']:.6f} | {r['rmse_vs_raw_p00']:.6f} |"
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
