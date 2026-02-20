"""
High-beta, low-mode ED convergence scan with checkpoints.

Design choices from user request:
- "not too many more modes" -> only n_modes in {2, 3}
- "crank beta up" -> beta extended to 12

Checkpoint behavior:
- writes scan CSV after every point
- prints [OK]/[ERR] progress lines
- rerun resumes from existing CSV
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class Case:
    n_modes: int
    n_cut: int


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


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


def _build_ed(beta: float, theta: float, case: Case, prefix: str) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=int(case.n_modes),
        n_cut=int(case.n_cut),
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix=prefix,
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    prefix = "hmf_ed_convergence_highbeta_lomodes_codex_v29"
    scan_csv = out_dir / f"{prefix}_scan.csv"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    fig_png = out_dir / f"{prefix}.png"
    log_md = out_dir / f"{prefix}_log.md"

    # High-beta sweep
    betas = np.linspace(0.6, 12.0, 12, dtype=float)
    theta = float(np.pi / 2.0)
    g = 0.5
    ren = RenormConfig(scale=1.04, kappa=0.94)

    # Low-mode, moderate cutoff ladder
    cases = [
        Case(2, 4), Case(2, 6), Case(2, 8), Case(2, 10), Case(2, 12),
        Case(3, 3), Case(3, 4), Case(3, 5), Case(3, 6),
    ]

    if scan_csv.exists():
        df = pd.read_csv(scan_csv)
    else:
        df = pd.DataFrame(
            columns=[
                "key", "status", "error", "timestamp", "elapsed_s",
                "beta", "theta", "g", "n_modes", "n_cut",
                "ed_p00", "ed_coh",
                "ordered_p00", "ordered_coh",
                "analytic_v12_p00", "analytic_v12_coh",
                "analytic_raw_p00", "analytic_raw_coh",
            ]
        )

    done = set(df["key"].astype(str).tolist()) if not df.empty else set()
    total = len(betas) * len(cases)
    print(f"[START] total={total}, done={len(done)}, remaining={total-len(done)}")

    ref_cache: dict[float, tuple[float, float, float, float, float, float]] = {}
    t_all = time.perf_counter()

    for case in cases:
        for beta in betas:
            key = f"{beta:.8f}|{case.n_modes}|{case.n_cut}"
            if key in done:
                continue

            t0 = time.perf_counter()
            row: dict[str, object] = {
                "key": key,
                "status": "ok",
                "error": "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "beta": float(beta),
                "theta": theta,
                "g": g,
                "n_modes": int(case.n_modes),
                "n_cut": int(case.n_cut),
            }

            try:
                if beta not in ref_cache:
                    lite = _build_lite(beta, theta)
                    rho_o = ordered_gaussian_state(lite, g)
                    p00_o, _p11_o, coh_o = extract_density(rho_o)
                    p00_v12, _p11_v12, coh_v12, _ratio_v12 = _compact_components(
                        lite, g, use_running=True, renorm=ren
                    )
                    p00_raw, _p11_raw, coh_raw, _ratio_raw = _compact_components(
                        lite, g, use_running=False, renorm=ren
                    )
                    ref_cache[beta] = (p00_o, coh_o, p00_v12, coh_v12, p00_raw, coh_raw)

                p00_o, coh_o, p00_v12, coh_v12, p00_raw, coh_raw = ref_cache[beta]
                row["ordered_p00"] = float(p00_o)
                row["ordered_coh"] = float(coh_o)
                row["analytic_v12_p00"] = float(p00_v12)
                row["analytic_v12_coh"] = float(coh_v12)
                row["analytic_raw_p00"] = float(p00_raw)
                row["analytic_raw_coh"] = float(coh_raw)

                ecfg = _build_ed(beta, theta, case, prefix=f"{prefix}_edtmp")
                ed_ctx = build_ed_context(ecfg)
                rho_ed = exact_reduced_state(ed_ctx, g)
                p00_ed, _p11_ed, coh_ed = extract_density(rho_ed)
                row["ed_p00"] = float(p00_ed)
                row["ed_coh"] = float(coh_ed)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                for c in [
                    "ed_p00", "ed_coh",
                    "ordered_p00", "ordered_coh",
                    "analytic_v12_p00", "analytic_v12_coh",
                    "analytic_raw_p00", "analytic_raw_coh",
                ]:
                    row[c] = np.nan

            row["elapsed_s"] = float(time.perf_counter() - t0)
            if df.empty:
                df = pd.DataFrame([row])
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            done.add(key)
            df.to_csv(scan_csv, index=False)

            status = str(row["status"]).upper()
            print(
                f"[{status}] {len(done)}/{total} beta={beta:.3f} m={case.n_modes} c={case.n_cut} "
                f"dt={row['elapsed_s']:.2f}s T={(time.perf_counter()-t_all)/60.0:.1f}m"
            )

    ok = df[df["status"] == "ok"].copy()
    summary_rows: list[dict[str, object]] = []
    for (n_modes, n_cut), grp in ok.groupby(["n_modes", "n_cut"]):
        grp = grp.sort_values("beta")
        summary_rows.append(
            {
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "n_points": int(len(grp)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["ed_p00"].to_numpy(float) - grp["ordered_p00"].to_numpy(float)),
                "rmse_ed_vs_v12_p00": _rmse(grp["ed_p00"].to_numpy(float) - grp["analytic_v12_p00"].to_numpy(float)),
                "rmse_ed_vs_raw_p00": _rmse(grp["ed_p00"].to_numpy(float) - grp["analytic_raw_p00"].to_numpy(float)),
                "p00_at_beta2": float(np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
                "p00_at_beta10": float(np.interp(10.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["n_modes", "n_cut"]).reset_index(drop=True)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    ax = axes[0]
    for (n_modes, n_cut), grp in ok.groupby(["n_modes", "n_cut"]):
        grp = grp.sort_values("beta")
        ax.plot(grp["beta"], grp["ed_p00"], linewidth=1.2, label=f"ED m{int(n_modes)}/c{int(n_cut)}")
    if not ok.empty:
        ref = ok.sort_values("beta").drop_duplicates(subset=["beta"])
        ax.plot(ref["beta"], ref["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
        ax.plot(ref["beta"], ref["analytic_v12_p00"], color="#0B6E4F", linestyle="--", linewidth=1.8, label="Analytic v12")
        ax.plot(ref["beta"], ref["analytic_raw_p00"], color="#AA3377", linestyle=":", linewidth=1.8, label="Analytic raw")
    ax.set_title("High-beta low-mode convergence")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for n_modes, grp_m in summary.groupby("n_modes"):
        grp_m = grp_m.sort_values("n_cut")
        ax.plot(grp_m["n_cut"], grp_m["p00_at_beta2"], "o-", linewidth=1.8, label=f"m={int(n_modes)} @ beta=2")
        ax.plot(grp_m["n_cut"], grp_m["p00_at_beta10"], "s--", linewidth=1.5, label=f"m={int(n_modes)} @ beta=10")
    if not ok.empty:
        ref = ok.sort_values("beta").drop_duplicates(subset=["beta"])
        ax.axhline(float(np.interp(2.0, ref["beta"], ref["ordered_p00"])), color="black", linewidth=1.5, label="Ordered @ beta=2")
        ax.axhline(float(np.interp(10.0, ref["beta"], ref["ordered_p00"])), color="black", linestyle=":", linewidth=1.2, label="Ordered @ beta=10")
    ax.set_title("Cutoff trend at beta=2 and beta=10")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# High-Beta Low-Mode ED Convergence (Codex v29)")
    lines.append("")
    lines.append("| n_modes | n_cut | n_points | rmse_ed_vs_ordered_p00 | rmse_ed_vs_v12_p00 | rmse_ed_vs_raw_p00 | p00_at_beta2 | p00_at_beta10 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_modes'])} | {int(r['n_cut'])} | {int(r['n_points'])} | "
            f"{r['rmse_ed_vs_ordered_p00']:.6f} | {r['rmse_ed_vs_v12_p00']:.6f} | {r['rmse_ed_vs_raw_p00']:.6f} | "
            f"{r['p00_at_beta2']:.6f} | {r['p00_at_beta10']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] wrote {scan_csv.name}, {summary_csv.name}, {fig_png.name}, {log_md.name}")


if __name__ == "__main__":
    main()
