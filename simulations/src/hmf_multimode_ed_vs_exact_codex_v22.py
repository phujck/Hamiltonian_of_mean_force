"""
Safe multimode ED comparison against exact-raw and ordered models.

Focus regime:
  theta = pi/2, g = 0.5, beta sweep

Purpose:
  Check whether adding bath modes in ED pushes behavior toward exact-raw.
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
    extract_density,
    ordered_gaussian_state,
)
from hmf_population_sweep_stable_compare_codex_v17 import _channels, _stable_p00
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig


@dataclass
class EDCase:
    label: str
    n_modes: int
    n_cut: int


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _build_ed(beta: float, theta: float, n_modes: int, n_cut: int) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=int(n_modes),
        n_cut=int(n_cut),
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_multimode_ed_vs_exact_codex_v22",
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_multimode_ed_vs_exact_scan_codex_v22.csv"
    summary_csv = out_dir / "hmf_multimode_ed_vs_exact_summary_codex_v22.csv"
    fig_png = out_dir / "hmf_multimode_ed_vs_exact_codex_v22.png"
    log_md = out_dir / "hmf_multimode_ed_vs_exact_log_codex_v22.md"

    betas = np.linspace(0.4, 6.0, 11)
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0

    # Safe-but-stronger ED cases: add modes while keeping matrix sizes manageable.
    cases = [
        EDCase("ed_m2_c10", n_modes=2, n_cut=10),  # dim = 2*10^2 = 200
        EDCase("ed_m3_c6", n_modes=3, n_cut=6),    # dim = 2*6^3 = 432
        EDCase("ed_m4_c4", n_modes=4, n_cut=4),    # dim = 2*4^4 = 512
    ]

    rows: list[dict[str, float | str | int]] = []
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

        sp, sm, dz = _channels(lite, g)
        p00_exact = _stable_p00(beta, omega_q, sp, sm, dz)
        p00_ord, _p11_ord, coh_ord = extract_density(ordered_gaussian_state(lite, g))

        for case in cases:
            ecfg = _build_ed(beta, theta, case.n_modes, case.n_cut)
            ed_ctx = build_ed_context(ecfg)
            rho_ed = exact_reduced_state(ed_ctx, g)
            p00_ed, _p11_ed, coh_ed = extract_density(rho_ed)

            rows.append(
                {
                    "beta": float(beta),
                    "theta": theta,
                    "g": g,
                    "case": case.label,
                    "n_modes": case.n_modes,
                    "n_cut": case.n_cut,
                    "ed_p00": float(p00_ed),
                    "ed_coh": float(coh_ed),
                    "ordered_p00": float(p00_ord),
                    "ordered_coh": float(coh_ord),
                    "exact_raw_p00": float(p00_exact),
                    "d_ed_minus_exact_p00": float(p00_ed - p00_exact),
                    "d_ed_minus_ordered_p00": float(p00_ed - p00_ord),
                    "d_ordered_minus_exact_p00": float(p00_ord - p00_exact),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["case", "beta"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str | int]] = []
    for case, grp in df.groupby("case"):
        summary_rows.append(
            {
                "case": str(case),
                "n_modes": int(grp["n_modes"].iloc[0]),
                "n_cut": int(grp["n_cut"].iloc[0]),
                "rmse_ed_vs_exact_p00": _rmse(grp["d_ed_minus_exact_p00"]),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_minus_ordered_p00"]),
                "rmse_ordered_vs_exact_p00": _rmse(grp["d_ordered_minus_exact_p00"]),
                "p00_beta2_ed": float(np.interp(2.0, grp["beta"], grp["ed_p00"])),
                "p00_beta2_ordered": float(np.interp(2.0, grp["beta"], grp["ordered_p00"])),
                "p00_beta2_exact": float(np.interp(2.0, grp["beta"], grp["exact_raw_p00"])),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["n_modes", "n_cut"]).reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    for case, grp in df.groupby("case"):
        ax.plot(grp["beta"], grp["ed_p00"], linewidth=1.8, label=f"{case} (m={int(grp['n_modes'].iloc[0])},c={int(grp['n_cut'].iloc[0])})")
    ref = df[df["case"] == summary["case"].iloc[0]].sort_values("beta")
    ax.plot(ref["beta"], ref["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(ref["beta"], ref["exact_raw_p00"], color="#0B6E4F", linewidth=2.0, linestyle="--", label="Exact raw")
    ax.set_title("Population vs beta (theta=pi/2, g=0.5)")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(summary["n_modes"] + 0.02 * summary["n_cut"], summary["rmse_ed_vs_exact_p00"], "o-", color="#0B6E4F", linewidth=2.0, label="ED vs exact")
    ax.plot(summary["n_modes"] + 0.02 * summary["n_cut"], summary["rmse_ed_vs_ordered_p00"], "o-", color="#C84B31", linewidth=2.0, label="ED vs ordered")
    ax.set_title("RMSE trend by ED case")
    ax.set_xlabel("n_modes + 0.02*n_cut")
    ax.set_ylabel("RMSE rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for case, grp in df.groupby("case"):
        ax.plot(grp["beta"], grp["d_ed_minus_ordered_p00"], linewidth=1.8, label=f"{case}: ED-ordered")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("ED - Ordered mismatch")
    ax.set_xlabel("beta")
    ax.set_ylabel("delta rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for case, grp in df.groupby("case"):
        ax.plot(grp["beta"], grp["d_ed_minus_exact_p00"], linewidth=1.8, label=f"{case}: ED-exact")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("ED - Exact mismatch")
    ax.set_xlabel("beta")
    ax.set_ylabel("delta rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Multimode ED vs Exact/Ordered (Codex v22)")
    lines.append("")
    lines.append("| case | n_modes | n_cut | rmse_ed_vs_exact_p00 | rmse_ed_vs_ordered_p00 | rmse_ordered_vs_exact_p00 | p00_beta2_ed | p00_beta2_ordered | p00_beta2_exact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['case']} | {int(r['n_modes'])} | {int(r['n_cut'])} | {r['rmse_ed_vs_exact_p00']:.6f} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {r['rmse_ordered_vs_exact_p00']:.6f} | "
            f"{r['p00_beta2_ed']:.6f} | {r['p00_beta2_ordered']:.6f} | {r['p00_beta2_exact']:.6f} |"
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
