"""
Checkpointed ED convergence scan based on the v12 comparison stack.

Features:
- scans ED across (n_modes, n_cut) for temperature sweep (and optional coupling sweep)
- compares ED against ordered and best-analytic (v12 style)
- checkpoint CSV is updated regularly so crashes can resume where they stopped
- resumable: rerun the same command and completed points are skipped

Run via:
  powershell -ExecutionPolicy Bypass -File run_safe.ps1 simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass
class SweepPoint:
    sweep: str
    beta: float
    theta: float
    g: float
    param_name: str
    param_value: float


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for token in s.split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("Expected a non-empty integer list.")
    return out


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else np.nan


def _key_of(
    sweep: str,
    beta: float,
    theta: float,
    g: float,
    n_modes: int,
    n_cut: int,
) -> str:
    return f"{sweep}|{beta:.8f}|{theta:.8f}|{g:.8f}|{n_modes}|{n_cut}"


def _build_lite(beta: float, theta: float) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=24,
        n_cut=1,
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
    )


def _build_ed(beta: float, theta: float, n_modes: int, n_cut: int, output_prefix: str) -> EDConfig:
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
        output_prefix=output_prefix,
    )


def _build_points(args: argparse.Namespace) -> list[SweepPoint]:
    points: list[SweepPoint] = []

    betas = np.linspace(args.beta_min, args.beta_max, args.beta_points, dtype=float)
    for beta in betas:
        points.append(
            SweepPoint(
                sweep="temperature_pi2_g05",
                beta=float(beta),
                theta=float(np.pi / 2.0),
                g=0.5,
                param_name="beta",
                param_value=float(beta),
            )
        )

    if args.include_coupling:
        gs = np.linspace(0.0, 2.0, args.g_points, dtype=float)
        for g in gs:
            points.append(
                SweepPoint(
                    sweep="coupling_beta2_pi4",
                    beta=2.0,
                    theta=float(np.pi / 4.0),
                    g=float(g),
                    param_name="g",
                    param_value=float(g),
                )
            )

    return points


def _write_summary_and_plots(
    df: pd.DataFrame,
    summary_csv: Path,
    fig_png: Path,
    log_md: Path,
) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        pd.DataFrame().to_csv(summary_csv, index=False)
        log_md.write_text("# No successful rows yet.\n", encoding="utf-8")
        return

    summary_rows: list[dict[str, float | int | str]] = []
    for (sweep, n_modes, n_cut), grp in ok.groupby(["sweep", "n_modes", "n_cut"]):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "n_points": int(len(grp)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["ed_p00"] - grp["ordered_p00"]),
                "rmse_ed_vs_best_p00": _rmse(grp["ed_p00"] - grp["best_p00"]),
                "rmse_ed_vs_ordered_coh": _rmse(grp["ed_coh"] - grp["ordered_coh"]),
                "rmse_ed_vs_best_coh": _rmse(grp["ed_coh"] - grp["best_coh"]),
                "p00_at_beta2": float(
                    np.interp(
                        2.0,
                        np.asarray(grp["beta"], dtype=float),
                        np.asarray(grp["ed_p00"], dtype=float),
                    )
                )
                if (grp["sweep"].iloc[0] == "temperature_pi2_g05")
                else np.nan,
            }
        )

    summary = (
        pd.DataFrame.from_records(summary_rows)
        .sort_values(["sweep", "n_modes", "n_cut"])
        .reset_index(drop=True)
    )
    summary.to_csv(summary_csv, index=False)

    # Plot only temperature sweep for readability.
    temp = ok[ok["sweep"] == "temperature_pi2_g05"].copy()
    if not temp.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)

        ax = axes[0, 0]
        for (n_modes, n_cut), grp in temp.groupby(["n_modes", "n_cut"]):
            g = grp.sort_values("beta")
            ax.plot(
                g["beta"],
                g["ed_p00"],
                linewidth=1.3,
                label=f"ED m{int(n_modes)}/c{int(n_cut)}",
            )
        ref = temp.sort_values("beta").drop_duplicates(subset=["beta"])
        ax.plot(ref["beta"], ref["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
        ax.plot(ref["beta"], ref["best_p00"], color="#0B6E4F", linewidth=2.0, linestyle="--", label="Best v12")
        ax.set_title("Temperature Sweep: Population")
        ax.set_xlabel("beta")
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)

        ax = axes[0, 1]
        temp_summary = summary[summary["sweep"] == "temperature_pi2_g05"].copy()
        for n_modes, grp_m in temp_summary.groupby("n_modes"):
            g = grp_m.sort_values("n_cut")
            ax.plot(
                g["n_cut"],
                g["p00_at_beta2"],
                "o-",
                linewidth=1.8,
                label=f"n_modes={int(n_modes)}",
            )
        if not ref.empty:
            p_ord_beta2 = float(np.interp(2.0, ref["beta"], ref["ordered_p00"]))
            p_best_beta2 = float(np.interp(2.0, ref["beta"], ref["best_p00"]))
            ax.axhline(p_ord_beta2, color="black", linewidth=1.5, label="Ordered @ beta=2")
            ax.axhline(p_best_beta2, color="#0B6E4F", linestyle="--", linewidth=1.5, label="Best v12 @ beta=2")
        ax.set_title("ED cutoff convergence at beta=2")
        ax.set_xlabel("n_cut")
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

        ax = axes[1, 0]
        for n_modes, grp_m in temp.groupby("n_modes"):
            s = (
                summary[
                    (summary["sweep"] == "temperature_pi2_g05")
                    & (summary["n_modes"] == int(n_modes))
                ]
                .sort_values("n_cut")
                .copy()
            )
            ax.plot(
                s["n_cut"],
                s["rmse_ed_vs_ordered_p00"],
                "o-",
                linewidth=1.8,
                label=f"vs ordered (m={int(n_modes)})",
            )
            ax.plot(
                s["n_cut"],
                s["rmse_ed_vs_best_p00"],
                "s--",
                linewidth=1.5,
                label=f"vs best (m={int(n_modes)})",
            )
        ax.set_title("Population RMSE convergence")
        ax.set_xlabel("n_cut")
        ax.set_ylabel("RMSE rho_00")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)

        ax = axes[1, 1]
        counts = df["status"].value_counts()
        labels = counts.index.tolist()
        vals = counts.values.astype(float)
        ax.bar(labels, vals, color=["#0B6E4F" if x == "ok" else "#C84B31" for x in labels])
        ax.set_title("Checkpoint status counts")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)

        fig.savefig(fig_png, dpi=180)
        plt.close(fig)

    lines: list[str] = []
    lines.append("# v12 ED Convergence Checkpoint Scan (Codex v25)")
    lines.append("")
    lines.append("| sweep | n_modes | n_cut | n_points | rmse_ed_vs_ordered_p00 | rmse_ed_vs_best_p00 | rmse_ed_vs_ordered_coh | rmse_ed_vs_best_coh | p00_at_beta2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {int(r['n_modes'])} | {int(r['n_cut'])} | {int(r['n_points'])} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {r['rmse_ed_vs_best_p00']:.6f} | "
            f"{r['rmse_ed_vs_ordered_coh']:.6f} | {r['rmse_ed_vs_best_coh']:.6f} | "
            f"{r['p00_at_beta2'] if np.isfinite(r['p00_at_beta2']) else np.nan:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed ED mode/cutoff convergence scan (v12 stack).")
    parser.add_argument("--n-modes-list", type=str, default="2,3,4")
    parser.add_argument("--n-cut-list", type=str, default="3,4,5,6,8,10")
    parser.add_argument("--beta-min", type=float, default=0.4)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--beta-points", type=int, default=15)
    parser.add_argument("--include-coupling", action="store_true")
    parser.add_argument("--g-points", type=int, default=11)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--output-prefix", type=str, default="hmf_v12_ed_convergence_checkpoint_codex_v25")
    args = parser.parse_args()

    n_modes_list = _parse_int_list(args.n_modes_list)
    n_cut_list = _parse_int_list(args.n_cut_list)

    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / f"{args.output_prefix}_scan.csv"
    summary_csv = out_dir / f"{args.output_prefix}_summary.csv"
    fig_png = out_dir / f"{args.output_prefix}.png"
    log_md = out_dir / f"{args.output_prefix}_log.md"

    points = _build_points(args)
    tasks = [
        (pt, n_modes, n_cut)
        for pt in points
        for n_modes in n_modes_list
        for n_cut in n_cut_list
    ]

    if scan_csv.exists() and args.resume:
        df = pd.read_csv(scan_csv)
    else:
        df = pd.DataFrame(
            columns=[
                "key",
                "status",
                "error",
                "timestamp",
                "elapsed_s",
                "sweep",
                "param_name",
                "param_value",
                "beta",
                "theta",
                "g",
                "n_modes",
                "n_cut",
                "ordered_p00",
                "ordered_coh",
                "best_p00",
                "best_coh",
                "ed_p00",
                "ed_coh",
            ]
        )

    done = set(df["key"].astype(str).tolist()) if not df.empty else set()

    ren = RenormConfig(scale=1.04, kappa=0.94)

    # Cache point-level ordered/best to avoid repeated work across ED cases.
    point_cache: dict[str, tuple[float, float, float, float]] = {}

    total = len(tasks)
    completed_start = len(done)
    print(f"[START] tasks={total}, existing={completed_start}, remaining={total - completed_start}")

    t_all = time.perf_counter()
    new_rows = 0

    try:
        for idx, (pt, n_modes, n_cut) in enumerate(tasks, start=1):
            key = _key_of(pt.sweep, pt.beta, pt.theta, pt.g, n_modes, n_cut)
            if key in done:
                continue

            t0 = time.perf_counter()
            row: dict[str, object] = {
                "key": key,
                "status": "ok",
                "error": "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sweep": pt.sweep,
                "param_name": pt.param_name,
                "param_value": pt.param_value,
                "beta": pt.beta,
                "theta": pt.theta,
                "g": pt.g,
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
            }

            try:
                point_key = f"{pt.sweep}|{pt.beta:.8f}|{pt.theta:.8f}|{pt.g:.8f}"
                if point_key not in point_cache:
                    lite_cfg = _build_lite(pt.beta, pt.theta)
                    rho_ord = ordered_gaussian_state(lite_cfg, pt.g)
                    ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)
                    best_p00, _best_p11, best_coh, _best_ratio = _compact_components(
                        lite_cfg, pt.g, use_running=True, renorm=ren
                    )
                    point_cache[point_key] = (ord_p00, ord_coh, best_p00, best_coh)

                ord_p00, ord_coh, best_p00, best_coh = point_cache[point_key]
                row["ordered_p00"] = float(ord_p00)
                row["ordered_coh"] = float(ord_coh)
                row["best_p00"] = float(best_p00)
                row["best_coh"] = float(best_coh)

                ed_cfg = _build_ed(
                    beta=pt.beta,
                    theta=pt.theta,
                    n_modes=n_modes,
                    n_cut=n_cut,
                    output_prefix=f"{args.output_prefix}_edtmp",
                )
                ed_ctx = build_ed_context(ed_cfg)
                rho_ed = exact_reduced_state(ed_ctx, pt.g)
                ed_p00, _ed_p11, ed_coh = extract_density(rho_ed)
                row["ed_p00"] = float(ed_p00)
                row["ed_coh"] = float(ed_coh)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                row["ordered_p00"] = np.nan
                row["ordered_coh"] = np.nan
                row["best_p00"] = np.nan
                row["best_coh"] = np.nan
                row["ed_p00"] = np.nan
                row["ed_coh"] = np.nan

            row["elapsed_s"] = float(time.perf_counter() - t0)
            if df.empty:
                df = pd.DataFrame([row])
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            done.add(key)
            new_rows += 1

            # Regular checkpoint writes.
            if new_rows % max(1, args.checkpoint_every) == 0:
                df.to_csv(scan_csv, index=False)
                _write_summary_and_plots(df, summary_csv, fig_png, log_md)

            status = str(row["status"]).upper()
            elapsed_total = time.perf_counter() - t_all
            finished = len(done)
            print(
                f"[{status}] {finished}/{total} "
                f"sweep={pt.sweep} beta={pt.beta:.3f} theta={pt.theta:.3f} g={pt.g:.3f} "
                f"m={n_modes} c={n_cut} dt={row['elapsed_s']:.2f}s T={elapsed_total/60.0:.1f}m"
            )

    except KeyboardInterrupt:
        print("[STOP] KeyboardInterrupt received. Writing checkpoint before exit.")

    # Final checkpoint and summaries.
    df.to_csv(scan_csv, index=False)
    _write_summary_and_plots(df, summary_csv, fig_png, log_md)
    print(f"[DONE] wrote {scan_csv.name}, {summary_csv.name}, {fig_png.name}, {log_md.name}")


if __name__ == "__main__":
    main()
