"""
Functional-difference diagnostic for analytic vs ordered density outputs (Codex v9).

Primary outputs:
1) Delta rho_00(g) = rho_00^analytic - rho_00^ordered
2) Relative ratio error(g) = (R_analytic / R_ordered) - 1, where R = rho_00/rho_11

Also fits simple candidate functional forms to each difference:
  - a * g
  - a * g^2
  - a * g^2 / (1 + b g^2)   (b fitted by coarse grid + linear solve for a)

Writes fresh codex_v9 outputs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    RenormConfig,
    ordered_gaussian_state,
    extract_density,
)
from hmf_component_normalized_compare_codex_v5 import _compact_components


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


@dataclass
class FitResult:
    name: str
    a: float
    b: float
    r2: float
    rmse: float
    yhat: np.ndarray


def _fit_linear_basis(y: np.ndarray, u: np.ndarray) -> tuple[float, np.ndarray]:
    den = float(np.dot(u, u))
    if den <= 1e-15:
        return 0.0, np.zeros_like(u)
    a = float(np.dot(y, u) / den)
    return a, a * u


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-15:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _fit_models(g: np.ndarray, y: np.ndarray) -> list[FitResult]:
    fits: list[FitResult] = []

    # Model 1: a g
    u1 = g
    a1, y1 = _fit_linear_basis(y, u1)
    fits.append(FitResult("a*g", a1, np.nan, _r2(y, y1), _rmse(y - y1), y1))

    # Model 2: a g^2
    u2 = g * g
    a2, y2 = _fit_linear_basis(y, u2)
    fits.append(FitResult("a*g^2", a2, np.nan, _r2(y, y2), _rmse(y - y2), y2))

    # Model 3: a g^2 / (1 + b g^2), search b then solve a linearly
    best = None
    for b in np.linspace(0.0, 10.0, 401):
        u = (g * g) / (1.0 + b * g * g)
        a, yhat = _fit_linear_basis(y, u)
        err = float(np.sum((y - yhat) ** 2))
        if best is None or err < best[0]:
            best = (err, a, b, yhat)
    assert best is not None
    _, a3, b3, y3 = best
    fits.append(FitResult("a*g^2/(1+b*g^2)", float(a3), float(b3), _r2(y, y3), _rmse(y - y3), y3))

    return fits


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ren = RenormConfig(scale=1.04, kappa=0.94)

    # Keep this conservative and focused on functional shape vs coupling.
    beta = 2.0
    theta = float(np.pi / 4.0)
    gs = np.linspace(0.0, 2.0, 21)

    rows = []
    for g in gs:
        cfg = BenchmarkConfig(
            beta=beta,
            omega_q=2.0,
            theta=theta,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        p00_ord, p11_ord, coh_ord = extract_density(ordered_gaussian_state(cfg, float(g)))
        ratio_ord = float(p00_ord / max(p11_ord, 1e-15))

        p00_an, p11_an, coh_an, ratio_an = _compact_components(cfg, float(g), use_running=True, renorm=ren)

        rows.append(
            {
                "g": float(g),
                "ordered_p00": float(p00_ord),
                "ordered_coh": float(coh_ord),
                "ordered_ratio": float(ratio_ord),
                "analytic_p00": float(p00_an),
                "analytic_coh": float(coh_an),
                "analytic_ratio": float(ratio_an),
                "delta_p00": float(p00_an - p00_ord),
                "delta_coh": float(coh_an - coh_ord),
                "ratio_rel_err": float(ratio_an / max(ratio_ord, 1e-15) - 1.0),
            }
        )

    df = pd.DataFrame.from_records(rows)

    y_p = df["delta_p00"].to_numpy(dtype=float)
    y_r = df["ratio_rel_err"].to_numpy(dtype=float)
    g = df["g"].to_numpy(dtype=float)

    fits_p = _fit_models(g, y_p)
    fits_r = _fit_models(g, y_r)

    # Save fit curves into scan CSV for easier inspection.
    for fit in fits_p:
        df[f"fit_delta_p00_{fit.name}"] = fit.yhat
    for fit in fits_r:
        df[f"fit_ratio_rel_{fit.name}"] = fit.yhat

    # Summary table
    sum_rows = []
    for target, fits in [("delta_p00", fits_p), ("ratio_rel_err", fits_r)]:
        for fit in fits:
            sum_rows.append(
                {
                    "target": target,
                    "model": fit.name,
                    "a": fit.a,
                    "b": fit.b,
                    "r2": fit.r2,
                    "rmse": fit.rmse,
                }
            )
    summary = pd.DataFrame.from_records(sum_rows).sort_values(["target", "rmse"]).reset_index(drop=True)

    # Write files
    scan_csv = out_dir / "hmf_rho00_ratio_functional_scan_codex_v9.csv"
    summary_csv = out_dir / "hmf_rho00_ratio_functional_summary_codex_v9.csv"
    fig_png = out_dir / "hmf_rho00_ratio_functional_diff_codex_v9.png"
    log_md = out_dir / "hmf_rho00_ratio_functional_log_codex_v9.md"
    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Top: delta rho_00
    ax = axes[0]
    ax.plot(df["g"], df["delta_p00"], "o-", color="black", linewidth=2, label=r"Data: $\Delta\rho_{00}$")
    colors = ["#0B6E4F", "#1F4E79", "#AA3377"]
    for fit, c in zip(fits_p, colors):
        ax.plot(df["g"], fit.yhat, color=c, linestyle="--", linewidth=1.8, label=f"{fit.name} (R2={fit.r2:.3f})")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\rho_{00}^{an} - \rho_{00}^{ord}$")
    ax.set_title(r"Functional shape of $\Delta\rho_{00}(g)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Bottom: ratio relative error
    ax = axes[1]
    ax.plot(df["g"], df["ratio_rel_err"], "o-", color="black", linewidth=2, label=r"Data: $R_{an}/R_{ord}-1$")
    for fit, c in zip(fits_r, colors):
        ax.plot(df["g"], fit.yhat, color=c, linestyle="--", linewidth=1.8, label=f"{fit.name} (R2={fit.r2:.3f})")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\frac{R_{an}}{R_{ord}} - 1$")
    ax.set_title(r"Functional shape of ratio mismatch vs $g$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    # Log
    lines = []
    lines.append("# Functional Difference Diagnostic (Codex v9)")
    lines.append("")
    lines.append(f"Config: beta={beta}, theta=pi/4, omega_q=2, g in [{gs.min():.1f}, {gs.max():.1f}] ({len(gs)} points)")
    lines.append("")
    lines.append("| target | model | a | b | R2 | RMSE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        btxt = f"{r['b']:.6f}" if np.isfinite(r["b"]) else "nan"
        lines.append(
            f"| {r['target']} | {r['model']} | {r['a']:.6f} | {btxt} | {r['r2']:.6f} | {r['rmse']:.6f} |"
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
