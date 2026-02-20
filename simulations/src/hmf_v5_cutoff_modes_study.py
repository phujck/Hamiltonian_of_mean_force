"""
Convergence study for ED truncation choices (mode count and Fock cutoff)
against the ordered-kernel finite-model comparator.

Outputs:
- simulations/results/data/hmf_v5_cutoff_modes_scan.csv
- simulations/results/data/hmf_v5_cutoff_modes_summary.csv
- simulations/results/figures/hmf_v5_cutoff_modes_convergence.png
- manuscript/figures/hmf_v5_cutoff_modes_convergence.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state,
)
from prl127_qubit_benchmark import (
    BenchmarkConfig,
    build_static_operators,
    partial_trace_bath,
    thermal_state,
    trace_distance,
)


def _ed_state(config: BenchmarkConfig, g: float) -> np.ndarray:
    h_static, h_int, _, h_s, _ = build_static_operators(config)
    dim_system = h_s.shape[0]
    dim_bath = h_static.shape[0] // dim_system
    # Keep ED definition consistent with v5 scans: no explicit counterterm.
    # In this qubit model X^2 = I, so lambda^2 Q X^2 would be a scalar shift.
    h_tot = h_static + g * h_int
    rho_tot = thermal_state(h_tot, config.beta)
    return partial_trace_bath(rho_tot, dim_system, dim_bath)


def _ordered_state(
    config: BenchmarkConfig,
    g: float,
    time_slices: int,
    kl_rank: int,
    gh_order: int,
    max_nodes: int,
) -> np.ndarray:
    ordered_ctx = _build_ordered_quadrature_context(
        config,
        n_time_slices=time_slices,
        kl_rank=kl_rank,
        gh_order=gh_order,
        max_nodes=max_nodes,
    )
    return finite_hmf_ordered_gaussian_state(g, ordered_ctx)


def _parse_float_list(text: str) -> List[float]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("Expected at least one value.")
    return out


def _parse_int_list(text: str) -> List[int]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("Expected at least one value.")
    return out


def run_study(args: argparse.Namespace) -> Dict[str, Path]:
    beta_omega_list = _parse_float_list(args.beta_omega_list)
    g_probe_list = _parse_float_list(args.g_probe_list)
    if len(beta_omega_list) != len(g_probe_list):
        raise ValueError("beta-omega-list and g-probe-list must have the same length.")

    n_cut_scan = _parse_int_list(args.n_cut_scan)
    n_modes_scan = _parse_int_list(args.n_modes_scan)
    n_cut_for_modes = _parse_int_list(args.n_cut_for_modes)

    rows = []
    for beta_omega, g_probe in zip(beta_omega_list, g_probe_list):
        beta = beta_omega / args.omega_q

        # Scan A: vary n_cut at fixed n_modes.
        for n_cut in n_cut_scan:
            cfg = BenchmarkConfig(
                beta=beta,
                omega_q=args.omega_q,
                theta=args.theta,
                n_modes=args.n_modes_fixed,
                n_cut=n_cut,
                omega_min=args.omega_min,
                omega_max=args.omega_max,
                q_strength=args.q_strength,
                tau_c=args.tau_c,
            )
            rho_ed = _ed_state(cfg, g_probe)
            rho_ord = _ordered_state(
                cfg,
                g_probe,
                time_slices=args.ordered_time_slices,
                kl_rank=args.ordered_kl_rank,
                gh_order=args.ordered_gh_order,
                max_nodes=args.ordered_max_nodes,
            )
            rows.append(
                {
                    "scan_type": "n_cut",
                    "beta_omega_q": beta_omega,
                    "beta": beta,
                    "g_probe": g_probe,
                    "n_modes": args.n_modes_fixed,
                    "n_cut": n_cut,
                    "trace_distance": trace_distance(rho_ed, rho_ord),
                }
            )

        # Scan B: vary n_modes at fixed n_cut values.
        for n_cut in n_cut_for_modes:
            for n_modes in n_modes_scan:
                cfg = BenchmarkConfig(
                    beta=beta,
                    omega_q=args.omega_q,
                    theta=args.theta,
                    n_modes=n_modes,
                    n_cut=n_cut,
                    omega_min=args.omega_min,
                    omega_max=args.omega_max,
                    q_strength=args.q_strength,
                    tau_c=args.tau_c,
                )
                rho_ed = _ed_state(cfg, g_probe)
                rho_ord = _ordered_state(
                    cfg,
                    g_probe,
                    time_slices=args.ordered_time_slices,
                    kl_rank=args.ordered_kl_rank,
                    gh_order=args.ordered_gh_order,
                    max_nodes=args.ordered_max_nodes,
                )
                rows.append(
                    {
                        "scan_type": "n_modes",
                        "beta_omega_q": beta_omega,
                        "beta": beta,
                        "g_probe": g_probe,
                        "n_modes": n_modes,
                        "n_cut": n_cut,
                        "trace_distance": trace_distance(rho_ed, rho_ord),
                    }
                )

    df = pd.DataFrame.from_records(rows)

    # Summary: report baseline and best value per beta for both scans.
    summary_rows = []
    for beta_omega in sorted(df["beta_omega_q"].unique()):
        sub_cut = df[(df["beta_omega_q"] == beta_omega) & (df["scan_type"] == "n_cut")].copy()
        sub_modes = df[(df["beta_omega_q"] == beta_omega) & (df["scan_type"] == "n_modes")].copy()
        sub_cut.sort_values("n_cut", inplace=True)
        baseline_cut_row = sub_cut.iloc[0]
        best_cut_row = sub_cut.loc[sub_cut["trace_distance"].idxmin()]

        # Baseline mode row at n_modes=min(n_modes_scan), n_cut=min(n_cut_for_modes)
        base_mode_row = sub_modes[
            (sub_modes["n_modes"] == min(n_modes_scan)) & (sub_modes["n_cut"] == min(n_cut_for_modes))
        ].iloc[0]
        best_mode_row = sub_modes.loc[sub_modes["trace_distance"].idxmin()]
        summary_rows.append(
            {
                "beta_omega_q": beta_omega,
                "g_probe": float(sub_cut.iloc[0]["g_probe"]),
                "baseline_n_cut": int(baseline_cut_row["n_cut"]),
                "baseline_n_cut_distance": float(baseline_cut_row["trace_distance"]),
                "best_n_cut": int(best_cut_row["n_cut"]),
                "best_n_cut_distance": float(best_cut_row["trace_distance"]),
                "baseline_n_modes": int(base_mode_row["n_modes"]),
                "baseline_modes_n_cut": int(base_mode_row["n_cut"]),
                "baseline_n_modes_distance": float(base_mode_row["trace_distance"]),
                "best_n_modes": int(best_mode_row["n_modes"]),
                "best_modes_n_cut": int(best_mode_row["n_cut"]),
                "best_n_modes_distance": float(best_mode_row["trace_distance"]),
            }
        )
    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("beta_omega_q")

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "simulations" / "results" / "data"
    fig_dir = project_root / "simulations" / "results" / "figures"
    ms_fig_dir = project_root / "manuscript" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ms_fig_dir.mkdir(parents=True, exist_ok=True)

    scan_path = data_dir / f"{args.output_prefix}_scan.csv"
    summary_path = data_dir / f"{args.output_prefix}_summary.csv"
    fig_path = fig_dir / f"{args.output_prefix}_convergence.png"
    ms_fig_path = ms_fig_dir / f"{args.output_prefix}_convergence.png"

    df.to_csv(scan_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    _plot(df, fig_path, ms_fig_path, n_cut_for_modes)
    return {
        "scan_csv": scan_path,
        "summary_csv": summary_path,
        "figure": fig_path,
        "manuscript_figure": ms_fig_path,
    }


def _plot(df: pd.DataFrame, fig_path: Path, ms_fig_path: Path, n_cut_for_modes: List[int]) -> None:
    colors = {0.5: "#5B2A86", 2.0: "#D6457B", 8.0: "#F2A900"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    sub_cut = df[df["scan_type"] == "n_cut"].copy()
    for beta_omega in sorted(sub_cut["beta_omega_q"].unique()):
        s = sub_cut[sub_cut["beta_omega_q"] == beta_omega].sort_values("n_cut")
        ax.plot(
            s["n_cut"],
            s["trace_distance"],
            marker="o",
            lw=2.0,
            color=colors.get(beta_omega, None),
            label=rf"$\beta\omega_q={beta_omega:g}$",
        )
    ax.set_xlabel(r"Fock cutoff $n_{\max}$ (fixed $N_\omega=2$)")
    ax.set_ylabel(r"$D(\rho_Q^{\rm ED},\rho_Q^{\rm ord})$")
    ax.set_title("(a) Cutoff Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    sub_modes = df[df["scan_type"] == "n_modes"].copy()
    linestyle_map = ["-", "--", ":", "-."]
    for beta_omega in sorted(sub_modes["beta_omega_q"].unique()):
        for idx, n_cut in enumerate(sorted(n_cut_for_modes)):
            s = sub_modes[(sub_modes["beta_omega_q"] == beta_omega) & (sub_modes["n_cut"] == n_cut)].sort_values(
                "n_modes"
            )
            ax.plot(
                s["n_modes"],
                s["trace_distance"],
                marker="o",
                lw=1.8,
                linestyle=linestyle_map[idx % len(linestyle_map)],
                color=colors.get(beta_omega, None),
                label=rf"$\beta\omega_q={beta_omega:g},\,n_{{\max}}={n_cut}$",
            )
    ax.set_xlabel(r"Bath mode count $N_\omega$")
    ax.set_ylabel(r"$D(\rho_Q^{\rm ED},\rho_Q^{\rm ord})$")
    ax.set_title("(b) Mode-Count Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    fig.savefig(ms_fig_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run n_cut/n_modes convergence study for v5 qubit benchmark.")
    parser.add_argument("--omega-q", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=float(np.pi / 4.0))
    parser.add_argument("--beta-omega-list", type=str, default="0.5,2,8")
    parser.add_argument(
        "--g-probe-list",
        type=str,
        default="0.329393,0.142668,0.026443",
        help="Representative g values, one per beta-omega entry.",
    )
    parser.add_argument("--omega-min", type=float, default=0.5)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--q-strength", type=float, default=10.0)
    parser.add_argument("--tau-c", type=float, default=1.0)

    parser.add_argument("--n-modes-fixed", type=int, default=2)
    parser.add_argument("--n-cut-scan", type=str, default="4,6,8,10,12")
    parser.add_argument("--n-modes-scan", type=str, default="1,2,3,4")
    parser.add_argument("--n-cut-for-modes", type=str, default="4,6")

    parser.add_argument("--ordered-time-slices", type=int, default=80)
    parser.add_argument("--ordered-kl-rank", type=int, default=5)
    parser.add_argument("--ordered-gh-order", type=int, default=3)
    parser.add_argument("--ordered-max-nodes", type=int, default=20000)

    parser.add_argument("--output-prefix", type=str, default="hmf_v5_cutoff_modes")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_paths = run_study(args)
    print("Cutoff/mode convergence study complete.")
    for key, path in out_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
