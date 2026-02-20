"""
Focused three-mode cutoff convergence scan up to n_cut=12 (omega_q=1).

Same setup as v39, but with n_modes fixed to 3.
Includes checkpoint/resume via scan CSV.

Run:
  powershell -ExecutionPolicy Bypass -File run_safe.ps1 simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.py
"""

from __future__ import annotations

from pathlib import Path
import time

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


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def _build_lite(
    beta: float,
    theta: float,
    omega_q: float,
    omega_min: float,
    omega_max: float,
    n_modes: int,
) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=int(n_modes),
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
    n_modes: int,
    n_cut: int,
) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=int(n_modes),
        n_cut=int(n_cut),
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_omega1_threemode_cutoff12_codex_v40",
    )


def run_scan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omega_q = 1.0
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_min = 0.2
    omega_max = 1.8
    n_modes = 3
    n_cuts = list(range(4, 13))  # 4..12 inclusive
    betas = np.linspace(0.6, 10.0, 17, dtype=float)
    ren = RenormConfig(scale=1.04, kappa=0.94)

    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_omega1_threemode_cutoff12_scan_codex_v40.csv"

    cols = [
        "key",
        "status",
        "error",
        "timestamp",
        "elapsed_s",
        "beta",
        "omega_q",
        "theta",
        "g",
        "omega_min",
        "omega_max",
        "n_modes",
        "n_cut",
        "ordered_p00",
        "ordered_coh",
        "analytic_cont_p00",
        "analytic_cont_coh",
        "analytic_disc_p00",
        "analytic_disc_coh",
        "ed_p00",
        "ed_coh",
        "d_ed_ordered_p00",
        "d_ed_cont_p00",
        "d_ed_disc_p00",
        "d_ed_ordered_coh",
        "d_ed_cont_coh",
        "d_ed_disc_coh",
    ]

    if scan_csv.exists():
        df = pd.read_csv(scan_csv)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
    else:
        df = pd.DataFrame(columns=cols)

    done = set(df["key"].astype(str).tolist()) if not df.empty else set()
    total = len(n_cuts) * len(betas)
    print(f"[START] total={total}, done={len(done)}, remaining={total-len(done)}")

    # Refs per beta.
    ref_cache: dict[float, tuple[float, float, float, float, float, float]] = {}
    for beta in betas:
        lite_ordered = _build_lite(beta, theta, omega_q, omega_min, omega_max, n_modes=40)
        rho_ord = ordered_gaussian_state(lite_ordered, g)
        ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)

        cont_p00, _cont_p11, cont_coh, _cont_ratio = _compact_components(
            lite_ordered, g, use_running=True, renorm=ren
        )

        lite_disc = _build_lite(beta, theta, omega_q, omega_min, omega_max, n_modes=n_modes)
        disc_p00, _disc_p11, disc_coh, _disc_ratio = _compact_components(
            lite_disc, g, use_running=True, renorm=ren
        )

        ref_cache[float(beta)] = (
            float(ord_p00),
            float(ord_coh),
            float(cont_p00),
            float(cont_coh),
            float(disc_p00),
            float(disc_coh),
        )

    k = len(done)
    for n_cut in n_cuts:
        for beta in betas:
            key = f"{beta:.8f}|{n_cut}"
            if key in done:
                continue

            t0 = time.perf_counter()
            row: dict[str, object] = {
                "key": key,
                "status": "ok",
                "error": "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "beta": float(beta),
                "omega_q": float(omega_q),
                "theta": float(theta),
                "g": float(g),
                "omega_min": float(omega_min),
                "omega_max": float(omega_max),
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
            }

            ord_p00, ord_coh, cont_p00, cont_coh, disc_p00, disc_coh = ref_cache[float(beta)]
            row["ordered_p00"] = ord_p00
            row["ordered_coh"] = ord_coh
            row["analytic_cont_p00"] = cont_p00
            row["analytic_cont_coh"] = cont_coh
            row["analytic_disc_p00"] = disc_p00
            row["analytic_disc_coh"] = disc_coh

            try:
                cfg = _build_ed(
                    beta=beta,
                    theta=theta,
                    omega_q=omega_q,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    n_modes=n_modes,
                    n_cut=n_cut,
                )
                ctx = build_ed_context(cfg)
                rho_ed = exact_reduced_state(ctx, g)
                ed_p00, _ed_p11, ed_coh = extract_density(rho_ed)
                row["ed_p00"] = float(ed_p00)
                row["ed_coh"] = float(ed_coh)
                row["d_ed_ordered_p00"] = float(ed_p00 - ord_p00)
                row["d_ed_cont_p00"] = float(ed_p00 - cont_p00)
                row["d_ed_disc_p00"] = float(ed_p00 - disc_p00)
                row["d_ed_ordered_coh"] = float(ed_coh - ord_coh)
                row["d_ed_cont_coh"] = float(ed_coh - cont_coh)
                row["d_ed_disc_coh"] = float(ed_coh - disc_coh)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                for c in [
                    "ed_p00",
                    "ed_coh",
                    "d_ed_ordered_p00",
                    "d_ed_cont_p00",
                    "d_ed_disc_p00",
                    "d_ed_ordered_coh",
                    "d_ed_cont_coh",
                    "d_ed_disc_coh",
                ]:
                    row[c] = np.nan

            row["elapsed_s"] = float(time.perf_counter() - t0)
            if df.empty:
                df = pd.DataFrame([row], columns=cols)
            else:
                df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)

            done.add(key)
            k += 1
            df.to_csv(scan_csv, index=False)
            if k % 15 == 0:
                print(f"[PROGRESS] {k}/{total}")

    ok = df[df["status"] == "ok"].copy().sort_values(["n_cut", "beta"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | int]] = []
    for n_cut, grp in ok.groupby("n_cut"):
        summary_rows.append(
            {
                "n_cut": int(n_cut),
                "n_points": int(len(grp)),
                "rmse_ed_vs_disc_p00": _rmse(grp["d_ed_disc_p00"].to_numpy(float)),
                "rmse_ed_vs_cont_p00": _rmse(grp["d_ed_cont_p00"].to_numpy(float)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_ordered_p00"].to_numpy(float)),
                "rmse_ed_vs_disc_coh": _rmse(grp["d_ed_disc_coh"].to_numpy(float)),
                "rmse_ed_vs_cont_coh": _rmse(grp["d_ed_cont_coh"].to_numpy(float)),
                "rmse_ed_vs_ordered_coh": _rmse(grp["d_ed_ordered_coh"].to_numpy(float)),
                "p00_at_beta2": float(np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
                "p00_at_beta8": float(np.interp(8.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("n_cut").reset_index(drop=True)

    ref12 = ok[ok["n_cut"] == 12].sort_values("beta")
    conv_rows: list[dict[str, float | int]] = []
    for n_cut, grp in ok.groupby("n_cut"):
        grp = grp.sort_values("beta")
        conv_rows.append(
            {
                "n_cut": int(n_cut),
                "rmse_cut_vs_12_p00": _rmse(grp["ed_p00"].to_numpy(float) - ref12["ed_p00"].to_numpy(float)),
                "rmse_cut_vs_12_coh": _rmse(grp["ed_coh"].to_numpy(float) - ref12["ed_coh"].to_numpy(float)),
            }
        )
    cutoff_conv = pd.DataFrame.from_records(conv_rows).sort_values("n_cut").reset_index(drop=True)

    return ok, summary, cutoff_conv


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, cutoff_conv: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_omega1_threemode_cutoff12_scan_codex_v40.csv"
    summary_csv = out_dir / "hmf_omega1_threemode_cutoff12_summary_codex_v40.csv"
    conv_csv = out_dir / "hmf_omega1_threemode_cutoff12_conv_codex_v40.csv"
    fig_png = out_dir / "hmf_omega1_threemode_cutoff12_codex_v40.png"
    log_md = out_dir / "hmf_omega1_threemode_cutoff12_log_codex_v40.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    cutoff_conv.to_csv(conv_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut in [4, 6, 8, 10, 12]:
        g = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(g["beta"], g["ed_p00"], linewidth=1.6, label=f"ED n_cut={n_cut}")
    ref = df[df["n_cut"] == 12].sort_values("beta")
    ax.plot(ref["beta"], ref["analytic_disc_p00"], "--", color="#0B6E4F", linewidth=2.0, label="Analytic discrete")
    ax.plot(ref["beta"], ref["analytic_cont_p00"], ":", color="#1F4E79", linewidth=2.0, label="Analytic continuum")
    ax.set_title("Population vs beta (three-mode, cutoff sweep)")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(summary["n_cut"], summary["rmse_ed_vs_disc_p00"], "o-", linewidth=2.0, label="ED vs disc")
    ax.plot(summary["n_cut"], summary["rmse_ed_vs_cont_p00"], "s--", linewidth=1.8, label="ED vs cont")
    ax.plot(summary["n_cut"], summary["rmse_ed_vs_ordered_p00"], "d:", linewidth=1.8, label="ED vs ordered")
    ax.set_title("Population RMSE vs cutoff")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(cutoff_conv["n_cut"], cutoff_conv["rmse_cut_vs_12_p00"], "o-", linewidth=2.0, label="p00")
    ax.plot(cutoff_conv["n_cut"], cutoff_conv["rmse_cut_vs_12_coh"], "s--", linewidth=1.8, label="coh")
    ax.set_title("Convergence to n_cut=12")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("RMSE vs cut=12")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(summary["n_cut"], summary["p00_at_beta2"], "o-", linewidth=1.8, label="ED p00 @ beta=2")
    ax.plot(summary["n_cut"], summary["p00_at_beta8"], "s--", linewidth=1.8, label="ED p00 @ beta=8")
    p_disc2 = float(np.interp(2.0, ref["beta"], ref["analytic_disc_p00"]))
    p_disc8 = float(np.interp(8.0, ref["beta"], ref["analytic_disc_p00"]))
    ax.axhline(p_disc2, color="#0B6E4F", linewidth=1.5, label="Disc analytic @ beta=2")
    ax.axhline(p_disc8, color="#0B6E4F", linestyle=":", linewidth=1.5, label="Disc analytic @ beta=8")
    ax.set_title("Cutoff trend at beta=2 and beta=8")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    best = summary.sort_values("rmse_ed_vs_disc_p00").iloc[0]
    lines: list[str] = []
    lines.append("# Omega_q=1 Three-Mode Cutoff<=12 Scan (Codex v40)")
    lines.append("")
    lines.append("Fixed setup:")
    lines.append("- omega_q=1, theta=pi/2, g=0.5")
    lines.append("- window=[0.2, 1.8] (center_d0.80)")
    lines.append("- n_modes=3, n_cut in [4..12]")
    lines.append("")
    lines.append(
        f"Best cutoff by fair metric (ED vs analytic-discrete p00): n_cut={int(best['n_cut'])}, "
        f"rmse={best['rmse_ed_vs_disc_p00']:.6f}"
    )
    lines.append("")
    lines.append("| n_cut | rmse_ed_vs_disc_p00 | rmse_ed_vs_cont_p00 | rmse_ed_vs_ordered_p00 | rmse_cut_vs_12_p00 |")
    lines.append("|---:|---:|---:|---:|---:|")
    conv_map = {int(r["n_cut"]): float(r["rmse_cut_vs_12_p00"]) for _, r in cutoff_conv.iterrows()}
    for _, r in summary.iterrows():
        c = int(r["n_cut"])
        lines.append(
            f"| {c} | {r['rmse_ed_vs_disc_p00']:.6f} | {r['rmse_ed_vs_cont_p00']:.6f} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {conv_map[c]:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", conv_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


def main() -> None:
    df, summary, cutoff_conv = run_scan()
    write_outputs(df, summary, cutoff_conv)


if __name__ == "__main__":
    main()

