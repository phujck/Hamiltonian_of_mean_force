"""
Numerically stable evaluation of the exact v5 density formula.

This script keeps the analytic model unchanged and compares two evaluators:
1) naive exponential form from Eq. (rho_matrix_v5),
2) stable Mobius/log-domain form for populations and ratio.

Outputs are fresh codex_v16 files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    laplace_k0,
    resonant_r0,
)


@dataclass
class Point:
    beta: float
    theta: float
    g: float


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


def _u_q(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float, float, float]:
    # Robust invariant parameterization:
    # u = gamma * dz = tanh(chi) * dz / chi, q = gamma * sqrt(sp*sm)
    prod = float(sp * sm)
    q0 = float(np.sqrt(max(prod, 0.0)))
    chi = float(np.hypot(dz, q0))
    if chi <= 1e-15:
        tanh_over_chi = 1.0
        tanh_chi = 0.0
    else:
        tanh_chi = float(np.tanh(chi))
        tanh_over_chi = tanh_chi / chi

    u = float(tanh_over_chi * dz)
    q = float(tanh_over_chi * q0)
    a = 0.5 * float(beta) * float(omega_q)
    return u, q, chi, a


def _naive_eval(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float, float]:
    u, q, _chi, a = _u_q(beta, omega_q, sp, sm, dz)
    z = 2.0 * (np.cosh(a) - u * np.sinh(a))
    p00 = float(np.exp(-a) * (1.0 + u) / z)
    p11 = float(np.exp(a) * (1.0 - u) / z)
    ratio = float(p00 / max(p11, 1e-300))
    return p00, p11, ratio


def _stable_eval(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float, float]:
    u, _q, _chi, a = _u_q(beta, omega_q, sp, sm, dz)

    # Exact endpoint handling avoids catastrophic bias when u is saturated.
    if u >= 1.0:
        return 1.0, 0.0, float(np.exp(700))
    if u <= -1.0:
        return 0.0, 1.0, float(np.exp(-700))

    # Strict interior for log1p stability.
    up = min(max(u, np.nextafter(-1.0, 0.0)), np.nextafter(1.0, 0.0))
    l0 = -a + np.log1p(up)   # unnormalized log weight of |0>
    l1 = +a + np.log1p(-up)  # unnormalized log weight of |1>
    m = max(l0, l1)
    w0 = np.exp(l0 - m)
    w1 = np.exp(l1 - m)
    z = w0 + w1
    p00 = float(w0 / z)
    p11 = float(w1 / z)

    # Ratio in log-domain: log R = l0 - l1
    log_ratio = -2.0 * a + np.log1p(up) - np.log1p(-up)
    # avoid overflow while preserving finite diagnostics
    if log_ratio > 700:
        ratio = float(np.exp(700))
    elif log_ratio < -700:
        ratio = float(np.exp(-700))
    else:
        ratio = float(np.exp(log_ratio))

    return p00, p11, ratio


def _points() -> list[Point]:
    betas = [0.2, 0.6, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    thetas = [float(np.pi / 4.0), float(np.pi / 2.0)]
    gs = [0.5, 1.0, 2.0]
    pts: list[Point] = []
    for b in betas:
        for th in thetas:
            for g in gs:
                pts.append(Point(beta=float(b), theta=float(th), g=float(g)))
    return pts


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_stable_exact_eval_scan_codex_v16.csv"
    summary_csv = out_dir / "hmf_stable_exact_eval_summary_codex_v16.csv"
    log_md = out_dir / "hmf_stable_exact_eval_log_codex_v16.md"

    rows: list[dict[str, float]] = []
    for p in _points():
        cfg = BenchmarkConfig(
            beta=p.beta,
            omega_q=2.0,
            theta=p.theta,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )

        sp, sm, dz = _channels(cfg, p.g)
        p00_n, p11_n, r_n = _naive_eval(cfg.beta, cfg.omega_q, sp, sm, dz)
        p00_s, p11_s, r_s = _stable_eval(cfg.beta, cfg.omega_q, sp, sm, dz)

        rows.append(
            {
                "beta": p.beta,
                "theta": p.theta,
                "g": p.g,
                "sigma_plus": sp,
                "sigma_minus": sm,
                "delta_z": dz,
                "naive_p00": p00_n,
                "stable_p00": p00_s,
                "naive_p11": p11_n,
                "stable_p11": p11_s,
                "naive_ratio": r_n,
                "stable_ratio": r_s,
                "abs_dp00_naive_minus_stable": abs(p00_n - p00_s),
                "abs_dp11_naive_minus_stable": abs(p11_n - p11_s),
                "rel_dratio_naive_minus_stable": abs(r_n - r_s) / max(abs(r_s), 1e-300),
                "naive_finite": float(np.isfinite(p00_n) and np.isfinite(p11_n) and np.isfinite(r_n)),
                "stable_finite": float(np.isfinite(p00_s) and np.isfinite(p11_s) and np.isfinite(r_s)),
            }
        )

    df = pd.DataFrame.from_records(rows)
    summary = pd.DataFrame.from_records(
        [
            {
                "n_points": int(len(df)),
                "max_abs_dp00_naive_minus_stable": float(np.max(df["abs_dp00_naive_minus_stable"])),
                "max_abs_dp11_naive_minus_stable": float(np.max(df["abs_dp11_naive_minus_stable"])),
                "max_rel_dratio_naive_minus_stable": float(np.max(df["rel_dratio_naive_minus_stable"])),
                "naive_nonfinite_count": int(np.sum(df["naive_finite"] < 0.5)),
                "stable_nonfinite_count": int(np.sum(df["stable_finite"] < 0.5)),
            }
        ]
    )

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    lines: list[str] = []
    lines.append("# Stable Exact Evaluator Check (Codex v16)")
    lines.append("")
    lines.append("Comparison: naive Eq.(rho_matrix_v5) evaluator vs stable Mobius/log-domain evaluator.")
    lines.append("")
    r = summary.iloc[0]
    lines.append(f"- n_points: {int(r['n_points'])}")
    lines.append(f"- max |delta p00|: {r['max_abs_dp00_naive_minus_stable']:.3e}")
    lines.append(f"- max |delta p11|: {r['max_abs_dp11_naive_minus_stable']:.3e}")
    lines.append(f"- max relative ratio diff: {r['max_rel_dratio_naive_minus_stable']:.3e}")
    lines.append(f"- naive non-finite count: {int(r['naive_nonfinite_count'])}")
    lines.append(f"- stable non-finite count: {int(r['stable_nonfinite_count'])}")
    lines.append("")
    lines.append("Stable form used:")
    lines.append(r"$m_z=(u-\tanh a)/(1-u\tanh a)$, $u=\tanh(\chi)\Delta_z/\chi$, $p_{00}=(1+m_z)/2$.")
    lines.append(r"Ratio from $\log R=-2a+\log(1+u)-\log(1-u)$.")
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
