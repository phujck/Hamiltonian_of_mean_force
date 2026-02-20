"""
Extended safe multimode ED comparison against exact-raw and ordered.

Compared to v22, this adds larger truncations per mode while keeping runtime safe.
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
        output_prefix="hmf_multimode_ed_vs_exact_extended_codex_v22b",
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_multimode_ed_vs_exact_extended_scan_codex_v22b.csv"
    summary_csv = out_dir / "hmf_multimode_ed_vs_exact_extended_summary_codex_v22b.csv"
    fig_png = out_dir / "hmf_multimode_ed_vs_exact_extended_codex_v22b.png"
    log_md = out_dir / "hmf_multimode_ed_vs_exact_extended_log_codex_v22b.md"

    betas = np.linspace(0.4, 6.0, 9)
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0

    cases = [
        EDCase("ed_m2_c10", n_modes=2, n_cut=10),  # dim=200
        EDCase("ed_m3_c8", n_modes=3, n_cut=8),    # dim=1024
        EDCase("ed_m3_c10", n_modes=3, n_cut=10),  # dim=2000
        EDCase("ed_m4_c5", n_modes=4, n_cut=5),    # dim=1250
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
        p00_ord, _p11_ord, _coh_ord = extract_density(ordered_gaussian_state(lite, g))

        for case in cases:
            ecfg = _build_ed(beta, theta, case.n_modes, case.n_cut)
            ed_ctx = build_ed_context(ecfg)
            rho_ed = exact_reduced_state(ed_ctx, g)
            p00_ed, _p11_ed, _coh_ed = extract_density(rho_ed)

            rows.append(
                {
                    "beta": float(beta),
                    "case": case.label,
                    "n_modes": case.n_modes,
                    "n_cut": case.n_cut,
                    "ed_p00": float(p00_ed),
                    "ordered_p00": float(p00_ord),
                    "exact_raw_p00": float(p00_exact),
                    "d_ed_minus_exact": float(p00_ed - p00_exact),
                    "d_ed_minus_ordered": float(p00_ed - p00_ord),
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
                "rmse_ed_vs_exact": _rmse(grp["d_ed_minus_exact"]),
                "rmse_ed_vs_ordered": _rmse(grp["d_ed_minus_ordered"]),
                "p00_beta2_ed": float(np.interp(2.0, grp["beta"], grp["ed_p00"])),
                "p00_beta2_ordered": float(np.interp(2.0, grp["beta"], grp["ordered_p00"])),
                "p00_beta2_exact": float(np.interp(2.0, grp["beta"], grp["exact_raw_p00"])),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["n_modes", "n_cut"]).reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax = axes[0]
    for case, grp in df.groupby("case"):
        ax.plot(grp["beta"], grp["ed_p00"], linewidth=1.8, label=f"{case}")
    ref = df[df["case"] == summary["case"].iloc[0]].sort_values("beta")
    ax.plot(ref["beta"], ref["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(ref["beta"], ref["exact_raw_p00"], color="#0B6E4F", linewidth=2.0, linestyle="--", label="Exact raw")
    ax.set_title("Population vs beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(summary["case"], summary["rmse_ed_vs_exact"], "o-", color="#0B6E4F", linewidth=2.0, label="ED vs exact")
    ax.plot(summary["case"], summary["rmse_ed_vs_ordered"], "o-", color="#C84B31", linewidth=2.0, label="ED vs ordered")
    ax.set_title("RMSE by ED case")
    ax.set_ylabel("RMSE rho_00")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(frameon=False)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Extended Multimode ED vs Exact/Ordered (Codex v22b)")
    lines.append("")
    lines.append("| case | n_modes | n_cut | rmse_ed_vs_exact | rmse_ed_vs_ordered | p00_beta2_ed | p00_beta2_ordered | p00_beta2_exact |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['case']} | {int(r['n_modes'])} | {int(r['n_cut'])} | {r['rmse_ed_vs_exact']:.6f} | "
            f"{r['rmse_ed_vs_ordered']:.6f} | {r['p00_beta2_ed']:.6f} | {r['p00_beta2_ordered']:.6f} | {r['p00_beta2_exact']:.6f} |"
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
