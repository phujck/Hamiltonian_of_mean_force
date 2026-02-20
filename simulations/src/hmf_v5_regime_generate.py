"""
Generate v5 regime-scan data (ED vs ordered/compact theory) without plotting.

Outputs:
- simulations/results/data/<prefix>_scan.csv
- simulations/results/data/<prefix>_summary.csv
- simulations/results/data/<prefix>_halfstep_validation.csv

Run (safe thread-limited):
  powershell -ExecutionPolicy Bypass -File .\run_safe.ps1 simulations/src/hmf_v5_regime_generate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from prl127_qubit_benchmark import BenchmarkConfig
from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state,
)

from hmf_v5_qubit_core import (
    build_ed_context,
    central_difference,
    compute_v5_base_channels,
    coupling_channels,
    exact_reduced_state,
    state_observables,
    state_trace_distance,
    v5_theory_state,
)


def _parse_beta_omega_list(text: str) -> List[float]:
    values = []
    for token in text.split(","):
        clean = token.strip()
        if clean:
            values.append(float(clean))
    if not values:
        raise ValueError("beta-omega-list must contain at least one numeric value.")
    return values


def _build_config(args: argparse.Namespace, beta: float) -> BenchmarkConfig:
    return BenchmarkConfig(
        beta=beta,
        omega_q=args.omega_q,
        theta=args.theta,
        lambda_min=args.g_min,
        lambda_max=args.g_max_cap,
        lambda_points=args.g_points,
        n_modes=args.n_modes,
        n_cut=args.n_cut,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        q_strength=args.q_strength,
        tau_c=args.tau_c,
        output_prefix=args.output_prefix,
    )


def _derive_g_range(
    g_min: float,
    g_max_cap: float,
    g_star_factor: float,
    chi0: float,
) -> Tuple[float, bool, bool]:
    if chi0 > 1e-14:
        g_star = float(chi0 ** (-0.5))
        target = float(g_star_factor * g_star)
        hit_cap = target > g_max_cap
        g_max = float(max(g_min, min(g_max_cap, target)))
        return g_max, hit_cap, True
    return float(max(g_min, g_max_cap)), False, False


def _run_single_beta(args: argparse.Namespace, beta_omega_q: float) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    beta = float(beta_omega_q / args.omega_q)
    config = _build_config(args, beta)
    base = compute_v5_base_channels(config, n_kernel_grid=args.n_kernel_grid)
    ordered_ctx = None
    if args.theory_model == "ordered":
        ordered_ctx = _build_ordered_quadrature_context(
            config,
            n_time_slices=args.ordered_time_slices,
            kl_rank=args.ordered_kl_rank,
            gh_order=args.ordered_gh_order,
            max_nodes=args.ordered_max_nodes,
        )

    g_max, hit_cap, has_g_star = _derive_g_range(
        g_min=args.g_min,
        g_max_cap=args.g_max_cap,
        g_star_factor=args.g_star_factor,
        chi0=base.chi0,
    )
    g_star = float(base.chi0 ** (-0.5)) if has_g_star else float("nan")

    g_values = np.linspace(args.g_min, g_max, int(args.g_points), dtype=float)
    ctx = build_ed_context(config)

    def _theory_state_from_g(g_value: float):
        channels_local = coupling_channels(base, float(g_value))
        if args.theory_model == "compact":
            rho_local = v5_theory_state(config, channels_local)
        else:
            rho_local = finite_hmf_ordered_gaussian_state(float(g_value), ordered_ctx)
        return rho_local, channels_local

    records: List[Dict[str, float]] = []
    for g in g_values:
        rho_ed = exact_reduced_state(ctx, float(g))
        rho_th, channels = _theory_state_from_g(float(g))

        obs_ed = state_observables(rho_ed)
        obs_th = state_observables(rho_th)
        d_state = state_trace_distance(rho_ed, rho_th)

        records.append(
            {
                "beta": beta,
                "omega_q": args.omega_q,
                "beta_omega_q": beta_omega_q,
                "theta": args.theta,
                "g": float(g),
                "g2": float(g * g),
                "theory_model": args.theory_model,
                "g_star": g_star,
                "g_max_used": g_max,
                "g_range_hit_cap": float(hit_cap),
                "chi0": base.chi0,
                "sigma_plus0": base.sigma_plus0,
                "sigma_minus0": base.sigma_minus0,
                "delta_z0": base.delta_z0,
                "kms_residual_base": base.kms_residual,
                "sigma_plus": channels.sigma_plus,
                "sigma_minus": channels.sigma_minus,
                "delta_z": channels.delta_z,
                "chi": channels.chi,
                "gamma": channels.gamma,
                "phi_ed": obs_ed["phi"],
                "r_ed": obs_ed["r"],
                "coherence_ed": obs_ed["coherence"],
                "rho01_abs_ed": obs_ed["rho01_abs"],
                "mx_ed": obs_ed["mx"],
                "my_ed": obs_ed["my"],
                "mz_ed": obs_ed["mz"],
                "gauge_phase_ed": obs_ed["gauge_phase"],
                "phi_th": obs_th["phi"],
                "r_th": obs_th["r"],
                "coherence_th": obs_th["coherence"],
                "rho01_abs_th": obs_th["rho01_abs"],
                "mx_th": obs_th["mx"],
                "my_th": obs_th["my"],
                "mz_th": obs_th["mz"],
                "gauge_phase_th": obs_th["gauge_phase"],
                "trace_distance": d_state,
            }
        )

    df = pd.DataFrame.from_records(records).sort_values("g").reset_index(drop=True)
    # phi is an orientation angle defined modulo pi; unwrap on 2*phi to avoid branch spikes.
    phi_ed_unwrapped = np.unwrap(2.0 * df["phi_ed"].to_numpy()) / 2.0
    phi_th_unwrapped = np.unwrap(2.0 * df["phi_th"].to_numpy()) / 2.0
    df["phi_ed_unwrapped"] = phi_ed_unwrapped
    df["phi_th_unwrapped"] = phi_th_unwrapped
    df["xi_phi_ed"] = central_difference(df["g"].to_numpy(), phi_ed_unwrapped)
    df["xi_phi_th"] = central_difference(df["g"].to_numpy(), phi_th_unwrapped)

    half_df = pd.DataFrame(
        columns=[
            "beta",
            "beta_omega_q",
            "method",
            "g",
            "xi_base",
            "xi_half_interp",
            "abs_diff",
            "max_abs_diff_beta_method",
            "median_abs_diff_beta_method",
        ]
    )
    half_ed_max = float("nan")
    half_th_max = float("nan")
    if args.halfstep_check and len(df) >= 3:
        g_dense = np.linspace(args.g_min, g_max, 2 * (len(df) - 1) + 1, dtype=float)
        phi_ed_dense = []
        phi_th_dense = []
        for g in g_dense:
            rho_ed = exact_reduced_state(ctx, float(g))
            rho_th, _ = _theory_state_from_g(float(g))
            phi_ed_dense.append(state_observables(rho_ed)["phi"])
            phi_th_dense.append(state_observables(rho_th)["phi"])
        phi_ed_dense = np.asarray(phi_ed_dense, dtype=float)
        phi_th_dense = np.asarray(phi_th_dense, dtype=float)
        phi_ed_dense_unwrapped = np.unwrap(2.0 * phi_ed_dense) / 2.0
        phi_th_dense_unwrapped = np.unwrap(2.0 * phi_th_dense) / 2.0

        xi_ed_dense = central_difference(g_dense, phi_ed_dense_unwrapped)
        xi_th_dense = central_difference(g_dense, phi_th_dense_unwrapped)
        xi_ed_half_interp = np.interp(df["g"].to_numpy(), g_dense, xi_ed_dense)
        xi_th_half_interp = np.interp(df["g"].to_numpy(), g_dense, xi_th_dense)

        ed_abs_diff = np.abs(df["xi_phi_ed"].to_numpy() - xi_ed_half_interp)
        th_abs_diff = np.abs(df["xi_phi_th"].to_numpy() - xi_th_half_interp)
        half_ed_max = float(np.max(ed_abs_diff))
        half_th_max = float(np.max(th_abs_diff))
        half_ed_med = float(np.median(ed_abs_diff))
        half_th_med = float(np.median(th_abs_diff))

        half_records = []
        for idx, g in enumerate(df["g"].to_numpy()):
            half_records.append(
                {
                    "beta": beta,
                    "beta_omega_q": beta_omega_q,
                    "method": "ed",
                    "g": float(g),
                    "xi_base": float(df.loc[idx, "xi_phi_ed"]),
                    "xi_half_interp": float(xi_ed_half_interp[idx]),
                    "abs_diff": float(ed_abs_diff[idx]),
                    "max_abs_diff_beta_method": half_ed_max,
                    "median_abs_diff_beta_method": half_ed_med,
                }
            )
            half_records.append(
                {
                    "beta": beta,
                    "beta_omega_q": beta_omega_q,
                    "method": "th",
                    "g": float(g),
                    "xi_base": float(df.loc[idx, "xi_phi_th"]),
                    "xi_half_interp": float(xi_th_half_interp[idx]),
                    "abs_diff": float(th_abs_diff[idx]),
                    "max_abs_diff_beta_method": half_th_max,
                    "median_abs_diff_beta_method": half_th_med,
                }
            )
        half_df = pd.DataFrame.from_records(half_records)

    row0 = df.iloc[0]
    i_peak_ed = int(np.argmax(np.abs(df["xi_phi_ed"].to_numpy())))
    i_peak_th = int(np.argmax(np.abs(df["xi_phi_th"].to_numpy())))
    i_max_d = int(np.argmax(df["trace_distance"].to_numpy()))

    summary = {
        "beta": beta,
        "omega_q": args.omega_q,
        "beta_omega_q": beta_omega_q,
        "theta": args.theta,
        "theory_model": args.theory_model,
        "n_modes": args.n_modes,
        "n_cut": args.n_cut,
        "omega_min": args.omega_min,
        "omega_max": args.omega_max,
        "q_strength": args.q_strength,
        "tau_c": args.tau_c,
        "n_kernel_grid": args.n_kernel_grid,
        "g_points": args.g_points,
        "g_min": args.g_min,
        "g_max_used": g_max,
        "g_max_cap": args.g_max_cap,
        "g_star_factor": args.g_star_factor,
        "has_finite_g_star": float(has_g_star),
        "g_star": g_star,
        "g_range_hit_cap": float(hit_cap),
        "chi0": base.chi0,
        "sigma_plus0": base.sigma_plus0,
        "sigma_minus0": base.sigma_minus0,
        "delta_z0": base.delta_z0,
        "kms_residual_base": base.kms_residual,
        "trace_distance_at_gmin": float(row0["trace_distance"]),
        "trace_distance_max": float(df.loc[i_max_d, "trace_distance"]),
        "g_at_trace_distance_max": float(df.loc[i_max_d, "g"]),
        "my_ed_abs_max": float(np.max(np.abs(df["my_ed"].to_numpy()))),
        "my_th_abs_max": float(np.max(np.abs(df["my_th"].to_numpy()))),
        "phi_peak_abs_xi_ed": float(np.max(np.abs(df["xi_phi_ed"].to_numpy()))),
        "g_at_phi_peak_abs_xi_ed": float(df.loc[i_peak_ed, "g"]),
        "phi_peak_abs_xi_th": float(np.max(np.abs(df["xi_phi_th"].to_numpy()))),
        "g_at_phi_peak_abs_xi_th": float(df.loc[i_peak_th, "g"]),
        "halfstep_checked": float(args.halfstep_check),
        "halfstep_xi_max_abs_diff_ed": float(half_ed_max),
        "halfstep_xi_max_abs_diff_th": float(half_th_max),
    }
    return df, summary, half_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate v5 regime ED-vs-theory data.")
    parser.add_argument("--omega-q", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=float(np.pi / 4.0))
    parser.add_argument("--beta-omega-list", type=str, default="0.5,2,8")

    parser.add_argument("--n-modes", type=int, default=2)
    parser.add_argument("--n-cut", type=int, default=4)
    parser.add_argument("--omega-min", type=float, default=0.5)
    parser.add_argument("--omega-max", type=float, default=8.0)
    parser.add_argument("--q-strength", type=float, default=10.0)
    parser.add_argument("--tau-c", type=float, default=1.0)

    parser.add_argument("--g-min", type=float, default=0.0)
    parser.add_argument("--g-max-cap", type=float, default=6.0)
    parser.add_argument("--g-points", type=int, default=121)
    parser.add_argument("--g-star-factor", type=float, default=2.5)

    parser.add_argument("--n-kernel-grid", type=int, default=4001)
    parser.add_argument("--halfstep-check", dest="halfstep_check", action="store_true")
    parser.add_argument("--no-halfstep-check", dest="halfstep_check", action="store_false")
    parser.set_defaults(halfstep_check=True)

    parser.add_argument(
        "--theory-model",
        type=str,
        choices=["ordered", "compact"],
        default="ordered",
        help="Comparator state model used for theory curves.",
    )
    parser.add_argument("--ordered-time-slices", type=int, default=80)
    parser.add_argument("--ordered-kl-rank", type=int, default=5)
    parser.add_argument("--ordered-gh-order", type=int, default=3)
    parser.add_argument("--ordered-max-nodes", type=int, default=20000)

    parser.add_argument("--output-prefix", type=str, default="hmf_v5_regime")
    return parser


def run_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    beta_omega_values = _parse_beta_omega_list(args.beta_omega_list)

    scan_frames = []
    summary_rows = []
    halfstep_frames = []
    for beta_omega_q in beta_omega_values:
        df_scan, summary_row, df_half = _run_single_beta(args, beta_omega_q)
        scan_frames.append(df_scan)
        summary_rows.append(summary_row)
        if not df_half.empty:
            halfstep_frames.append(df_half)

    scan_df = pd.concat(scan_frames, ignore_index=True).sort_values(["beta_omega_q", "g"]).reset_index(drop=True)
    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("beta_omega_q").reset_index(drop=True)
    if halfstep_frames:
        halfstep_df = pd.concat(halfstep_frames, ignore_index=True).sort_values(["beta_omega_q", "method", "g"])
    else:
        halfstep_df = pd.DataFrame(
            columns=[
                "beta",
                "beta_omega_q",
                "method",
                "g",
                "xi_base",
                "xi_half_interp",
                "abs_diff",
                "max_abs_diff_beta_method",
                "median_abs_diff_beta_method",
            ]
        )

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "simulations" / "results" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_path = out_dir / f"{args.output_prefix}_scan.csv"
    summary_path = out_dir / f"{args.output_prefix}_summary.csv"
    halfstep_path = out_dir / f"{args.output_prefix}_halfstep_validation.csv"

    scan_df.to_csv(scan_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    halfstep_df.to_csv(halfstep_path, index=False)
    return scan_df, summary_df, halfstep_df, scan_path, summary_path, halfstep_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scan_df, summary_df, halfstep_df, scan_path, summary_path, halfstep_path = run_from_args(args)

    print("v5 regime data generation complete.")
    print(f"Rows (scan): {len(scan_df)}")
    print(f"Rows (summary): {len(summary_df)}")
    print(f"Rows (halfstep): {len(halfstep_df)}")
    print(f"Scan CSV: {scan_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Half-step CSV: {halfstep_path}")


if __name__ == "__main__":
    main()
