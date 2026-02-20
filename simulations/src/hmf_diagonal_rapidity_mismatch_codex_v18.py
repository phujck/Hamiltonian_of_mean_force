"""
Diagonal-channel mismatch in rapidity coordinates.

Idea:
Population depends on u = gamma * Delta_z through
    m_z = (u - t) / (1 - u t),   t = tanh(a), a = beta*omega_q/2.
Hence from any rho_00 we can infer u, then d = atanh(u).

This script compares:
  d_exact = atanh(u_exact)  from analytic channels
  d_ord   = atanh(u_from_ordered_population)

and fits suppression trends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    extract_density,
    laplace_k0,
    ordered_gaussian_state,
    resonant_r0,
)


@dataclass
class Sweep:
    name: str
    param: str
    values: np.ndarray
    beta: float | None
    theta: float | None
    g: float | None


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _clamp_u(u: float) -> float:
    return float(np.clip(u, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)))


def _channels(cfg: BenchmarkConfig, g: float) -> tuple[float, float, float]:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    c = float(np.cos(cfg.theta))
    s = float(np.sin(cfg.theta))
    g2 = float(g * g)

    k0_0 = laplace_k0(cfg, 0.0, 1201)
    k0_p = laplace_k0(cfg, w, 1201)
    k0_m = laplace_k0(cfg, -w, 1201)
    r_p = resonant_r0(cfg, w, 1201)
    r_m = resonant_r0(cfg, -w, 1201)

    sp = g2 * (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sm = g2 * (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    dz = g2 * (s * s) * 0.5 * (r_p - r_m)
    return float(sp), float(sm), float(dz)


def _u_exact(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> float:
    q0 = float(np.sqrt(max(sp * sm, 0.0)))
    chi = float(np.hypot(dz, q0))
    if chi <= 1e-15:
        return float(dz)
    return float(np.tanh(chi) * dz / chi)


def _u_from_p00(p00: float, beta: float, omega_q: float) -> float:
    # p00 = (1 + m_z)/2, m_z = (u - t)/(1 - u t)
    # invert: u = (m_z + t)/(1 + m_z t)
    a = 0.5 * beta * omega_q
    t = float(np.tanh(a))
    mz = float(2.0 * p00 - 1.0)
    den = 1.0 + mz * t
    if abs(den) < 1e-15:
        den = 1e-15 if den >= 0.0 else -1e-15
    return _clamp_u((mz + t) / den)


def _fit_scale_g2(g: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(g) & np.isfinite(y)
    g = g[mask]
    y = y[mask]
    if len(g) == 0:
        return float("nan"), float("nan"), float("nan")
    # fit y ~ a/(1+b g^2), b via grid search, a linear for fixed b
    best = None
    g2 = g * g
    for b in np.linspace(0.0, 20.0, 801):
        u = 1.0 / (1.0 + b * g2)
        den = float(np.dot(u, u))
        if den <= 1e-15:
            continue
        a = float(np.dot(y, u) / den)
        yhat = a * u
        err = float(np.mean((y - yhat) ** 2))
        if best is None or err < best[0]:
            best = (err, a, b, yhat)
    assert best is not None
    err, a, b, yhat = best
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 1e-15 else np.nan
    return float(a), float(b), float(r2)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_diagonal_rapidity_mismatch_scan_codex_v18.csv"
    summary_csv = out_dir / "hmf_diagonal_rapidity_mismatch_summary_codex_v18.csv"
    fig_png = out_dir / "hmf_diagonal_rapidity_mismatch_codex_v18.png"
    log_md = out_dir / "hmf_diagonal_rapidity_mismatch_log_codex_v18.md"

    sweeps = [
        Sweep(
            name="g_beta2_theta_pi2",
            param="g",
            values=np.linspace(0.0, 2.0, 21),
            beta=2.0,
            theta=float(np.pi / 2.0),
            g=None,
        ),
        Sweep(
            name="g_beta2_theta_pi4",
            param="g",
            values=np.linspace(0.0, 2.0, 21),
            beta=2.0,
            theta=float(np.pi / 4.0),
            g=None,
        ),
        Sweep(
            name="beta_theta_pi2_g05",
            param="beta",
            values=np.linspace(0.2, 6.0, 17),
            beta=None,
            theta=float(np.pi / 2.0),
            g=0.5,
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for sw in sweeps:
        for p in sw.values:
            beta = float(p) if sw.param == "beta" else float(sw.beta)
            theta = float(p) if sw.param == "theta" else float(sw.theta)
            g = float(p) if sw.param == "g" else float(sw.g)
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

            sp, sm, dz = _channels(cfg, g)
            u_ex = _clamp_u(_u_exact(beta, cfg.omega_q, sp, sm, dz))
            d_ex = float(np.arctanh(u_ex))

            p00_ord, _p11_ord, _coh_ord = extract_density(ordered_gaussian_state(cfg, g))
            u_ord = _u_from_p00(p00_ord, beta, cfg.omega_q)
            d_ord = float(np.arctanh(u_ord))

            scale = d_ord / d_ex if abs(d_ex) > 1e-15 else np.nan
            rows.append(
                {
                    "sweep": sw.name,
                    "param_name": sw.param,
                    "param": float(p),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "delta_z_exact": dz,
                    "u_exact": u_ex,
                    "u_ordered_inferred": u_ord,
                    "d_exact": d_ex,
                    "d_ordered_inferred": d_ord,
                    "d_scale_ordered_over_exact": scale,
                    "d_diff_ordered_minus_exact": d_ord - d_ex,
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["sweep", "param"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "rmse_u": _rmse(grp["u_ordered_inferred"] - grp["u_exact"]),
                "rmse_d": _rmse(grp["d_diff_ordered_minus_exact"]),
                "scale_median": float(np.nanmedian(grp["d_scale_ordered_over_exact"])),
                "scale_min": float(np.nanmin(grp["d_scale_ordered_over_exact"])),
                "scale_max": float(np.nanmax(grp["d_scale_ordered_over_exact"])),
            }
        )

    # Fit suppression vs g for coupling sweeps
    fit_results: dict[str, tuple[float, float, float]] = {}
    for sweep_name in ("g_beta2_theta_pi2", "g_beta2_theta_pi4"):
        g_grp = df[df["sweep"] == sweep_name].copy()
        g = g_grp["g"].to_numpy(dtype=float)
        y = g_grp["d_scale_ordered_over_exact"].to_numpy(dtype=float)
        a_fit, b_fit, r2_fit = _fit_scale_g2(g, y)
        fit_results[sweep_name] = (a_fit, b_fit, r2_fit)
        summary_rows.append(
            {
                "sweep": f"fit_{sweep_name}",
                "rmse_u": np.nan,
                "rmse_d": np.nan,
                "scale_median": a_fit,
                "scale_min": b_fit,
                "scale_max": r2_fit,
            }
        )

    summary = pd.DataFrame.from_records(summary_rows)
    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # g sweep (theta=pi/2): u
    g2_grp = df[df["sweep"] == "g_beta2_theta_pi2"].copy()
    ax = axes[0, 0]
    ax.plot(g2_grp["g"], g2_grp["u_exact"], color="#0B6E4F", linewidth=2.0, label="u exact")
    ax.plot(g2_grp["g"], g2_grp["u_ordered_inferred"], color="black", linewidth=2.0, label="u ordered inferred")
    ax.set_title("g sweep (theta=pi/2): u")
    ax.set_xlabel("g")
    ax.set_ylabel("u")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # g sweep (theta=pi/2): d-scale + fit
    ax = axes[0, 1]
    g = g2_grp["g"].to_numpy(dtype=float)
    y = g2_grp["d_scale_ordered_over_exact"].to_numpy(dtype=float)
    mask = np.isfinite(g) & np.isfinite(y)
    g_fit = g[mask]
    y_fit = y[mask]
    a_fit, b_fit, r2_fit = fit_results["g_beta2_theta_pi2"]
    yhat = a_fit / (1.0 + b_fit * g_fit * g_fit) if np.isfinite(a_fit) and np.isfinite(b_fit) else np.full_like(g_fit, np.nan)
    ax.plot(g_fit, y_fit, "o-", color="black", linewidth=2.0, label="d_ord / d_exact")
    ax.plot(g_fit, yhat, "--", color="#C84B31", linewidth=1.8, label=f"a/(1+b g^2), R2={r2_fit:.3f}")
    ax.set_title("g sweep (theta=pi/2): d-scale")
    ax.set_xlabel("g")
    ax.set_ylabel("scale")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # beta sweep: u
    b_grp = df[df["sweep"] == "beta_theta_pi2_g05"].copy()
    ax = axes[1, 0]
    ax.plot(b_grp["beta"], b_grp["u_exact"], color="#0B6E4F", linewidth=2.0, label="u exact")
    ax.plot(b_grp["beta"], b_grp["u_ordered_inferred"], color="black", linewidth=2.0, label="u ordered inferred")
    ax.set_title("beta sweep: inferred diagonal variable u")
    ax.set_xlabel("beta")
    ax.set_ylabel("u")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # beta sweep: d scale
    ax = axes[1, 1]
    ax.plot(b_grp["beta"], b_grp["d_scale_ordered_over_exact"], "o-", color="black", linewidth=2.0)
    ax.set_title("beta sweep: d_ord / d_exact")
    ax.set_xlabel("beta")
    ax.set_ylabel("scale")
    ax.grid(alpha=0.25)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Diagonal Rapidity Mismatch (Codex v18)")
    lines.append("")
    lines.append("Population channel reparameterized via u and d=atanh(u).")
    lines.append("")
    lines.append("| sweep | rmse_u | rmse_d | scale_median | scale_min | scale_max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['rmse_u'] if pd.notna(r['rmse_u']) else np.nan:.6f} | "
            f"{r['rmse_d'] if pd.notna(r['rmse_d']) else np.nan:.6f} | "
            f"{r['scale_median']:.6f} | {r['scale_min']:.6f} | {r['scale_max']:.6f} |"
        )
    lines.append("")
    for name in ("g_beta2_theta_pi2", "g_beta2_theta_pi4"):
        a_fit, b_fit, r2_fit = fit_results[name]
        lines.append(
            f"{name} suppression fit: scale(g) ~ a/(1+b g^2), a={a_fit:.6f}, b={b_fit:.6f}, R2={r2_fit:.6f}."
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))
    print("")
    for name in ("g_beta2_theta_pi2", "g_beta2_theta_pi4"):
        a_fit, b_fit, r2_fit = fit_results[name]
        print(f"fit({name}): a={a_fit:.6f}, b={b_fit:.6f}, R2={r2_fit:.6f}")


if __name__ == "__main__":
    main()
