"""
Complete omega_q=1 diagnostic suite for ED vs analytic agreement.

What this does:
- rescales system frequency to omega_q=1
- scans spectral windows, mode count, and cutoff size
- keeps ED and analytic on the same spectral window at each point
- finds the best agreement point (ED vs best analytic population)
- writes a packaged PDF report

Run with:
  powershell -ExecutionPolicy Bypass -File run_safe.ps1 simulations/src/hmf_omega1_diagnostic_suite_codex_v37.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

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
class WindowSpec:
    name: str
    family: str
    omega_min: float
    omega_max: float
    bandwidth: float


@dataclass(frozen=True)
class EDCase:
    n_modes: int
    n_cut: int

    @property
    def label(self) -> str:
        return f"m{self.n_modes}c{self.n_cut}"


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def _tail_fraction(omega_max: float, tau_c: float) -> float:
    # For J(w) ~ tau*w*exp(-tau*w): fraction of area above omega_max.
    x = tau_c * omega_max
    return float(np.exp(-x) * (1.0 + x))


def _detune_and_dw(omega_min: float, omega_max: float, omega_q: float, n_modes: int) -> tuple[float, float]:
    if n_modes <= 1:
        w = 0.5 * (omega_min + omega_max)
        return float(abs(w - omega_q) / omega_q), float(omega_max - omega_min)
    grid = np.linspace(omega_min, omega_max, n_modes, dtype=float)
    dw = float(grid[1] - grid[0])
    detune = float(np.min(np.abs(grid - omega_q)) / omega_q)
    return detune, dw


def _build_windows(omega_q: float) -> list[WindowSpec]:
    windows: list[WindowSpec] = []

    # Centered windows around omega_q.
    for delta in [0.15, 0.30, 0.50, 0.80, 1.20]:
        wmin = max(0.05 * omega_q, omega_q - delta)
        wmax = omega_q + delta
        windows.append(
            WindowSpec(
                name=f"center_d{delta:.2f}",
                family="centered",
                omega_min=float(wmin),
                omega_max=float(wmax),
                bandwidth=float(wmax - wmin),
            )
        )

    # Left-anchored windows (include low frequencies with increasing UV cutoff).
    ratio_min = 0.05
    for ratio_max in [2.0, 3.0, 4.0, 5.0, 6.0]:
        wmin = ratio_min * omega_q
        wmax = ratio_max * omega_q
        windows.append(
            WindowSpec(
                name=f"ratio_r{ratio_max:.1f}",
                family="ratio",
                omega_min=float(wmin),
                omega_max=float(wmax),
                bandwidth=float(wmax - wmin),
            )
        )
    return windows


def _build_lite(beta: float, theta: float, omega_q: float, w: WindowSpec) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=40,
        n_cut=1,
        omega_min=float(w.omega_min),
        omega_max=float(w.omega_max),
        q_strength=5.0,
        tau_c=0.5,
    )


def _build_ed(beta: float, theta: float, omega_q: float, w: WindowSpec, case: EDCase) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=int(case.n_modes),
        n_cut=int(case.n_cut),
        omega_min=float(w.omega_min),
        omega_max=float(w.omega_max),
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_omega1_diagnostic_suite_codex_v37",
    )


def _write_checkpoint(df: pd.DataFrame, scan_csv: Path) -> None:
    df.to_csv(scan_csv, index=False)


def run_suite() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omega_q = 1.0
    theta = float(np.pi / 2.0)
    g = 0.5
    tau_c = 0.5
    ren = RenormConfig(scale=1.04, kappa=0.94)
    betas = np.linspace(0.6, 10.0, 11, dtype=float)
    windows = _build_windows(omega_q)
    cases = [
        EDCase(2, 4), EDCase(2, 5), EDCase(2, 6),
        EDCase(3, 4), EDCase(3, 5), EDCase(3, 6),
        EDCase(4, 4), EDCase(4, 5), EDCase(4, 6),
    ]

    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_omega1_diagnostic_suite_scan_codex_v37.csv"

    cols = [
        "key", "status", "error", "timestamp", "elapsed_s",
        "window_name", "window_family", "bandwidth", "omega_min", "omega_max",
        "beta", "omega_q", "theta", "g", "n_modes", "n_cut", "case",
        "dw", "detune_over_omega_q", "tail_fraction",
        "ordered_p00", "ordered_coh", "best_p00", "best_coh", "ed_p00", "ed_coh",
        "d_ed_ordered_p00", "d_ed_best_p00", "d_ed_ordered_coh", "d_ed_best_coh",
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
    total = len(windows) * len(betas) * len(cases)
    print(f"[START] omega_q=1.0 total={total} done={len(done)} remaining={total-len(done)}")

    ref_cache: dict[tuple[str, float], tuple[float, float, float, float]] = {}
    t_all = time.perf_counter()
    new_rows = 0

    for w in windows:
        for beta in betas:
            ref_key = (w.name, float(beta))
            if ref_key not in ref_cache:
                lite = _build_lite(beta=beta, theta=theta, omega_q=omega_q, w=w)
                rho_ord = ordered_gaussian_state(lite, g)
                ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)
                best_p00, _best_p11, best_coh, _best_ratio = _compact_components(
                    lite, g, use_running=True, renorm=ren
                )
                ref_cache[ref_key] = (float(ord_p00), float(ord_coh), float(best_p00), float(best_coh))

            ord_p00, ord_coh, best_p00, best_coh = ref_cache[ref_key]

            for case in cases:
                key = f"{w.name}|{beta:.8f}|{case.n_modes}|{case.n_cut}"
                if key in done:
                    continue

                t0 = time.perf_counter()
                row: dict[str, object] = {
                    "key": key,
                    "status": "ok",
                    "error": "",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "window_name": w.name,
                    "window_family": w.family,
                    "bandwidth": float(w.bandwidth),
                    "omega_min": float(w.omega_min),
                    "omega_max": float(w.omega_max),
                    "beta": float(beta),
                    "omega_q": float(omega_q),
                    "theta": float(theta),
                    "g": float(g),
                    "n_modes": int(case.n_modes),
                    "n_cut": int(case.n_cut),
                    "case": case.label,
                    "ordered_p00": ord_p00,
                    "ordered_coh": ord_coh,
                    "best_p00": best_p00,
                    "best_coh": best_coh,
                }

                try:
                    detune, dw = _detune_and_dw(
                        omega_min=w.omega_min,
                        omega_max=w.omega_max,
                        omega_q=omega_q,
                        n_modes=case.n_modes,
                    )
                    cfg = _build_ed(beta=beta, theta=theta, omega_q=omega_q, w=w, case=case)
                    ed_ctx = build_ed_context(cfg)
                    rho_ed = exact_reduced_state(ed_ctx, g)
                    ed_p00, _ed_p11, ed_coh = extract_density(rho_ed)

                    row["dw"] = float(dw)
                    row["detune_over_omega_q"] = float(detune)
                    row["tail_fraction"] = _tail_fraction(w.omega_max, tau_c)
                    row["ed_p00"] = float(ed_p00)
                    row["ed_coh"] = float(ed_coh)
                    row["d_ed_ordered_p00"] = float(ed_p00 - ord_p00)
                    row["d_ed_best_p00"] = float(ed_p00 - best_p00)
                    row["d_ed_ordered_coh"] = float(ed_coh - ord_coh)
                    row["d_ed_best_coh"] = float(ed_coh - best_coh)
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)
                    for c in ["dw", "detune_over_omega_q", "tail_fraction", "ed_p00", "ed_coh",
                              "d_ed_ordered_p00", "d_ed_best_p00", "d_ed_ordered_coh", "d_ed_best_coh"]:
                        row[c] = np.nan

                row["elapsed_s"] = float(time.perf_counter() - t0)
                if df.empty:
                    df = pd.DataFrame([row], columns=cols)
                else:
                    df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)
                done.add(key)
                new_rows += 1

                if new_rows % 15 == 0:
                    _write_checkpoint(df, scan_csv)

                if new_rows % 30 == 0:
                    elapsed_m = (time.perf_counter() - t_all) / 60.0
                    print(f"[PROGRESS] {len(done)}/{total} elapsed={elapsed_m:.1f}m")

    _write_checkpoint(df, scan_csv)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful rows in suite scan.")

    # Summary per (window, case)
    summary_rows: list[dict[str, float | int | str]] = []
    for (window_name, case, n_modes, n_cut), grp in ok.groupby(["window_name", "case", "n_modes", "n_cut"]):
        summary_rows.append(
            {
                "window_name": str(window_name),
                "window_family": str(grp["window_family"].iloc[0]),
                "case": str(case),
                "n_modes": int(n_modes),
                "n_cut": int(n_cut),
                "bandwidth": float(grp["bandwidth"].iloc[0]),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(grp["omega_max"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_ed_vs_best_p00": _rmse(grp["d_ed_best_p00"].to_numpy(float)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_ordered_p00"].to_numpy(float)),
                "rmse_ed_vs_best_coh": _rmse(grp["d_ed_best_coh"].to_numpy(float)),
                "rmse_ed_vs_ordered_coh": _rmse(grp["d_ed_ordered_coh"].to_numpy(float)),
                "p00_at_beta2": float(
                    np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
                "p00_at_beta8": float(
                    np.interp(8.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["rmse_ed_vs_best_p00", "n_modes", "n_cut"]
    ).reset_index(drop=True)

    # Best window for each case.
    case_best_rows: list[pd.Series] = []
    for (_n_modes, _n_cut), grp in summary.groupby(["n_modes", "n_cut"]):
        case_best_rows.append(grp.sort_values("rmse_ed_vs_best_p00").iloc[0])
    case_best = pd.DataFrame(case_best_rows).sort_values(["n_modes", "n_cut"]).reset_index(drop=True)

    # Cutoff convergence metric per (window, n_modes): compare cut4/5 against cut6.
    conv_rows: list[dict[str, float | int | str]] = []
    for (window_name, n_modes), grp in ok.groupby(["window_name", "n_modes"]):
        g4 = grp[grp["n_cut"] == 4].sort_values("beta")
        g5 = grp[grp["n_cut"] == 5].sort_values("beta")
        g6 = grp[grp["n_cut"] == 6].sort_values("beta")
        if g6.empty:
            continue
        rmse_46 = _rmse(g4["ed_p00"].to_numpy(float) - g6["ed_p00"].to_numpy(float)) if len(g4) else np.nan
        rmse_56 = _rmse(g5["ed_p00"].to_numpy(float) - g6["ed_p00"].to_numpy(float)) if len(g5) else np.nan
        conv_rows.append(
            {
                "window_name": str(window_name),
                "window_family": str(grp["window_family"].iloc[0]),
                "n_modes": int(n_modes),
                "bandwidth": float(grp["bandwidth"].iloc[0]),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(grp["omega_max"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_cut4_vs_cut6_p00": float(rmse_46),
                "rmse_cut5_vs_cut6_p00": float(rmse_56),
            }
        )
    cutoff_summary = pd.DataFrame.from_records(conv_rows).sort_values(
        ["n_modes", "bandwidth"]
    ).reset_index(drop=True)

    return ok, summary, case_best, cutoff_summary


def _plot_suite(
    ok: pd.DataFrame,
    summary: pd.DataFrame,
    case_best: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    fig_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    # 1) Per-case best agreement by (n_modes,n_cut)
    ax = axes[0, 0]
    x = np.arange(len(case_best))
    labels = [f"m{int(r.n_modes)}c{int(r.n_cut)}" for r in case_best.itertuples(index=False)]
    ax.bar(x, case_best["rmse_ed_vs_best_p00"], color="#0B6E4F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Best RMSE per (n_modes, n_cut)")
    ax.set_ylabel("RMSE(ED-best p00)")
    ax.grid(axis="y", alpha=0.25)

    # 2) Cutoff convergence for n_modes=3
    ax = axes[0, 1]
    c3 = cutoff_summary[cutoff_summary["n_modes"] == 3].sort_values("bandwidth")
    if not c3.empty:
        ax.plot(c3["bandwidth"], c3["rmse_cut4_vs_cut6_p00"], "o-", linewidth=1.8, label="cut4 vs cut6")
        ax.plot(c3["bandwidth"], c3["rmse_cut5_vs_cut6_p00"], "s--", linewidth=1.8, label="cut5 vs cut6")
    ax.set_title("Internal cutoff convergence (n_modes=3)")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("RMSE between ED cutoffs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # 3) Beta sweep at global best point
    ax = axes[1, 0]
    best = summary.iloc[0]
    sub = ok[
        (ok["window_name"] == best["window_name"])
        & (ok["n_modes"] == int(best["n_modes"]))
    ].sort_values(["n_cut", "beta"])
    for n_cut, grp in sub.groupby("n_cut"):
        ax.plot(grp["beta"], grp["ed_p00"], linewidth=1.8, label=f"ED n_cut={int(n_cut)}")
    ref = sub.sort_values("beta").drop_duplicates(subset=["beta"])
    ax.plot(ref["beta"], ref["best_p00"], "--", color="#0B6E4F", linewidth=2.0, label="Best analytic")
    ax.plot(ref["beta"], ref["ordered_p00"], ":", color="black", linewidth=2.0, label="Ordered")
    ax.set_title(
        f"Global-best window: {best['window_name']} (m={int(best['n_modes'])}, c={int(best['n_cut'])})"
    )
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # 4) Window descriptors of top results
    ax = axes[1, 1]
    top = summary.head(12).copy()
    ax.plot(top["bandwidth"], top["rmse_ed_vs_best_p00"], "o-", color="#1F4E79", linewidth=1.8, label="RMSE top12")
    ax2 = ax.twinx()
    ax2.plot(top["bandwidth"], top["tail_fraction"], "s--", color="#AA3377", linewidth=1.5, label="tail fraction")
    ax.set_title("Top-12 windows: RMSE and tail fraction")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("tail fraction")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="best")

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)


def _write_log(
    summary: pd.DataFrame,
    case_best: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    log_md: Path,
) -> None:
    lines: list[str] = []
    best = summary.iloc[0]
    lines.append("# Omega_q=1 Diagnostic Suite (Codex v37)")
    lines.append("")
    lines.append("Global best agreement (ED vs best analytic p00):")
    lines.append(
        f"- window={best['window_name']} ({best['window_family']}), "
        f"omega_min={best['omega_min']:.3f}, omega_max={best['omega_max']:.3f}, bandwidth={best['bandwidth']:.3f}"
    )
    lines.append(
        f"- n_modes={int(best['n_modes'])}, n_cut={int(best['n_cut'])}, "
        f"rmse_ed_vs_best_p00={best['rmse_ed_vs_best_p00']:.6f}"
    )
    lines.append("")
    lines.append("| case | window | omega_min | omega_max | bandwidth | rmse_ed_vs_best_p00 | rmse_ed_vs_ordered_p00 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in case_best.iterrows():
        lines.append(
            f"| m{int(r['n_modes'])}c{int(r['n_cut'])} | {r['window_name']} | "
            f"{r['omega_min']:.3f} | {r['omega_max']:.3f} | {r['bandwidth']:.3f} | "
            f"{r['rmse_ed_vs_best_p00']:.6f} | {r['rmse_ed_vs_ordered_p00']:.6f} |"
        )
    lines.append("")
    lines.append("| n_modes | window | bandwidth | rmse_cut4_vs_cut6_p00 | rmse_cut5_vs_cut6_p00 |")
    lines.append("|---:|---|---:|---:|---:|")
    for _, r in cutoff_summary.iterrows():
        lines.append(
            f"| {int(r['n_modes'])} | {r['window_name']} | {r['bandwidth']:.3f} | "
            f"{r['rmse_cut4_vs_cut6_p00']:.6f} | {r['rmse_cut5_vs_cut6_p00']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")


def _write_tex(
    summary: pd.DataFrame,
    case_best: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    fig_png: Path,
    report_tex: Path,
) -> None:
    best = summary.iloc[0]
    top = summary.head(12).copy()
    top_cols = [
        "window_name", "window_family", "n_modes", "n_cut",
        "omega_min", "omega_max", "bandwidth", "rmse_ed_vs_best_p00",
    ]
    top_ltx = top[top_cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}")

    case_cols = ["n_modes", "n_cut", "window_name", "omega_min", "omega_max", "rmse_ed_vs_best_p00"]
    case_ltx = case_best[case_cols].to_latex(index=False, float_format=lambda x: f"{x:.4f}")

    conv_ltx = cutoff_summary.head(18).to_latex(index=False, float_format=lambda x: f"{x:.4f}")

    tex = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\title{HMF Omega\_q=1 Diagnostic Suite (Codex v37)}")
    tex.append(r"\date{}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append(r"\section*{Global Best Point}")
    tex.append(
        f"Best agreement occurs at window {best['window_name']} "
        f"($\\omega_\\min={best['omega_min']:.3f}$, $\\omega_\\max={best['omega_max']:.3f}$), "
        f"$n_\\mathrm{{modes}}={int(best['n_modes'])}$, $n_\\mathrm{{cut}}={int(best['n_cut'])}$ "
        f"with RMSE$(\\rho_{{00}})$ = {best['rmse_ed_vs_best_p00']:.6f}."
    )
    tex.append(r"\section*{Suite Figure}")
    tex.append(r"\begin{center}")
    tex.append(rf"\includegraphics[width=0.96\textwidth]{{{fig_png.name}}}")
    tex.append(r"\end{center}")
    tex.append(r"\section*{Top Windows}")
    tex.append(top_ltx)
    tex.append(r"\section*{Best Window Per (Modes, Cutoff)}")
    tex.append(case_ltx)
    tex.append(r"\section*{Cutoff Convergence Snapshot}")
    tex.append(conv_ltx)
    tex.append(r"\end{document}")
    report_tex.write_text("\n".join(tex), encoding="utf-8")


def _compile_or_fallback_pdf(
    summary: pd.DataFrame,
    case_best: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    fig_png: Path,
    report_tex: Path,
    report_pdf: Path,
    fallback_pdf: Path,
) -> tuple[str, str]:
    # Try LaTeX first.
    if shutil.which("pdflatex"):
        try:
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", report_tex.name],
                    cwd=report_tex.parent,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if report_pdf.exists():
                return "pdflatex", report_pdf.name
        except Exception:
            pass

    # Fallback: create a simple packaged PDF with matplotlib.
    best = summary.iloc[0]
    with PdfPages(fallback_pdf) as pdf:
        # Page 1: text summary
        fig = plt.figure(figsize=(8.5, 11))
        txt = []
        txt.append("HMF Omega_q=1 Diagnostic Suite (Codex v37)")
        txt.append("")
        txt.append("Global best agreement (ED vs best analytic p00):")
        txt.append(
            f"window={best['window_name']} ({best['window_family']}), "
            f"omega_min={best['omega_min']:.3f}, omega_max={best['omega_max']:.3f}, bandwidth={best['bandwidth']:.3f}"
        )
        txt.append(
            f"n_modes={int(best['n_modes'])}, n_cut={int(best['n_cut'])}, "
            f"rmse_ed_vs_best_p00={best['rmse_ed_vs_best_p00']:.6f}"
        )
        txt.append("")
        txt.append("Best window by case:")
        for _, r in case_best.iterrows():
            txt.append(
                f"m{int(r['n_modes'])}c{int(r['n_cut'])}: {r['window_name']} "
                f"[{r['omega_min']:.3f}, {r['omega_max']:.3f}], rmse={r['rmse_ed_vs_best_p00']:.6f}"
            )
        txt.append("")
        txt.append("Cutoff convergence snapshot (first rows):")
        for _, r in cutoff_summary.head(12).iterrows():
            txt.append(
                f"n_modes={int(r['n_modes'])}, {r['window_name']}: "
                f"c4-c6={r['rmse_cut4_vs_cut6_p00']:.6f}, c5-c6={r['rmse_cut5_vs_cut6_p00']:.6f}"
            )
        fig.text(0.06, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: suite figure
        img = plt.imread(fig_png)
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return "fallback", fallback_pdf.name


def write_outputs(
    ok: pd.DataFrame,
    summary: pd.DataFrame,
    case_best: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
) -> None:
    out_dir = Path(__file__).resolve().parent
    summary_csv = out_dir / "hmf_omega1_diagnostic_suite_summary_codex_v37.csv"
    case_best_csv = out_dir / "hmf_omega1_diagnostic_suite_casebest_codex_v37.csv"
    cutoff_csv = out_dir / "hmf_omega1_diagnostic_suite_cutoff_codex_v37.csv"
    fig_png = out_dir / "hmf_omega1_diagnostic_suite_codex_v37.png"
    log_md = out_dir / "hmf_omega1_diagnostic_suite_log_codex_v37.md"
    report_tex = out_dir / "hmf_omega1_diagnostic_suite_report_codex_v37.tex"
    report_pdf = out_dir / "hmf_omega1_diagnostic_suite_report_codex_v37.pdf"
    fallback_pdf = out_dir / "hmf_omega1_diagnostic_suite_report_fallback_codex_v37.pdf"

    summary.to_csv(summary_csv, index=False)
    case_best.to_csv(case_best_csv, index=False)
    cutoff_summary.to_csv(cutoff_csv, index=False)

    _plot_suite(ok, summary, case_best, cutoff_summary, fig_png)
    _write_log(summary, case_best, cutoff_summary, log_md)
    _write_tex(summary, case_best, cutoff_summary, fig_png, report_tex)
    method, pdf_name = _compile_or_fallback_pdf(
        summary=summary,
        case_best=case_best,
        cutoff_summary=cutoff_summary,
        fig_png=fig_png,
        report_tex=report_tex,
        report_pdf=report_pdf,
        fallback_pdf=fallback_pdf,
    )

    best = summary.iloc[0]
    print("Wrote:", summary_csv.name)
    print("Wrote:", case_best_csv.name)
    print("Wrote:", cutoff_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("Wrote:", report_tex.name)
    print(f"Wrote PDF via {method}: {pdf_name}")
    print("")
    print(
        "BEST:",
        f"window={best['window_name']}",
        f"omega=[{best['omega_min']:.3f},{best['omega_max']:.3f}]",
        f"m={int(best['n_modes'])}",
        f"c={int(best['n_cut'])}",
        f"rmse={best['rmse_ed_vs_best_p00']:.6f}",
    )


def main() -> None:
    ok, summary, case_best, cutoff_summary = run_suite()
    write_outputs(ok, summary, case_best, cutoff_summary)


if __name__ == "__main__":
    main()

