"""
Test a diagonal-rapidity renormalization inferred from mismatch trends.

Model:
  d_exact = atanh(u_exact), where u_exact = gamma * Delta_z
  d_eff   = s(g) * d_exact, s(g)=a/(1+b g^2)
  u_eff   = tanh(d_eff)

Then rebuild density components with unchanged transverse q channel.
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


def _u_q_exact(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float]:
    q0 = float(np.sqrt(max(sp * sm, 0.0)))
    chi = float(np.hypot(dz, q0))
    if chi <= 1e-15:
        return float(dz), float(q0)
    fac = float(np.tanh(chi) / chi)
    return float(fac * dz), float(fac * q0)


def _p00_coh_from_uq(beta: float, omega_q: float, u: float, q: float) -> tuple[float, float]:
    a = 0.5 * beta * omega_q
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-15:
        den = 1e-15 if den >= 0.0 else -1e-15
    mz = (u - t) / den
    mx = (q / np.cosh(a)) / den
    p00 = float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))
    coh = float(0.5 * abs(mx))
    return p00, coh


def _scale_g_theta(g: float, theta: float, a_fit: float, b_fit: float) -> float:
    # Apply suppression mainly in the longitudinally dominated sector.
    s_g = float(a_fit / (1.0 + b_fit * g * g))
    w = float(np.sin(theta) ** 2)
    return float((1.0 - w) + w * s_g)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_diagonal_rapidity_renorm_scan_codex_v19.csv"
    summary_csv = out_dir / "hmf_diagonal_rapidity_renorm_summary_codex_v19.csv"
    fig_png = out_dir / "hmf_diagonal_rapidity_renorm_compare_codex_v19.png"
    log_md = out_dir / "hmf_diagonal_rapidity_renorm_log_codex_v19.md"

    # Fitted from v18 g-sweep at beta=2, theta=pi/2.
    a_fit = 1.040503
    b_fit = 13.45

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
            name="beta_theta_pi2_g05",
            param="beta",
            values=np.linspace(0.2, 6.0, 17),
            beta=None,
            theta=float(np.pi / 2.0),
            g=0.5,
        ),
        Sweep(
            name="g_beta2_theta_pi4",
            param="g",
            values=np.linspace(0.0, 2.0, 21),
            beta=2.0,
            theta=float(np.pi / 4.0),
            g=None,
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
            u_ex, q_ex = _u_q_exact(beta, cfg.omega_q, sp, sm, dz)
            u_ex = _clamp_u(u_ex)
            p00_ex, coh_ex = _p00_coh_from_uq(beta, cfg.omega_q, u_ex, q_ex)

            d_ex = float(np.arctanh(u_ex))
            s = _scale_g_theta(g, theta, a_fit, b_fit)
            u_eff = float(np.tanh(s * d_ex))
            p00_eff, coh_eff = _p00_coh_from_uq(beta, cfg.omega_q, u_eff, q_ex)

            p00_ord, _p11_ord, coh_ord = extract_density(ordered_gaussian_state(cfg, g))

            rows.append(
                {
                    "sweep": sw.name,
                    "param_name": sw.param,
                    "param": float(p),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "ordered_p00": p00_ord,
                    "ordered_coh": coh_ord,
                    "exact_raw_p00": p00_ex,
                    "exact_raw_coh": coh_ex,
                    "diag_renorm_p00": p00_eff,
                    "diag_renorm_coh": coh_eff,
                    "raw_minus_ordered_p00": p00_ex - p00_ord,
                    "renorm_minus_ordered_p00": p00_eff - p00_ord,
                    "raw_minus_ordered_coh": coh_ex - coh_ord,
                    "renorm_minus_ordered_coh": coh_eff - coh_ord,
                    "diag_scale_sg": s,
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["sweep", "param"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "rmse_raw_p00": _rmse(grp["raw_minus_ordered_p00"]),
                "rmse_renorm_p00": _rmse(grp["renorm_minus_ordered_p00"]),
                "rmse_raw_coh": _rmse(grp["raw_minus_ordered_coh"]),
                "rmse_renorm_coh": _rmse(grp["renorm_minus_ordered_coh"]),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("sweep").reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for r, sweep in enumerate(summary["sweep"].tolist()):
        grp = df[df["sweep"] == sweep].sort_values("param")
        x = grp["param"].to_numpy(dtype=float)
        xlabel = grp["param_name"].iloc[0]

        ax = axes[r, 0]
        ax.plot(x, grp["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
        ax.plot(x, grp["exact_raw_p00"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact raw")
        ax.plot(x, grp["diag_renorm_p00"], color="#0B6E4F", linewidth=2.0, label="Diag-renorm")
        ax.set_title(f"{sweep}: population")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)

        ax = axes[r, 1]
        ax.plot(x, grp["ordered_coh"], color="black", linewidth=2.0, label="Ordered")
        ax.plot(x, grp["exact_raw_coh"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact raw")
        ax.plot(x, grp["diag_renorm_coh"], color="#0B6E4F", linewidth=2.0, label="Diag-renorm")
        ax.set_title(f"{sweep}: coherence")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|rho_01|")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Diagonal Rapidity Renorm Comparison (Codex v19)")
    lines.append("")
    lines.append(f"Using s(g)=a/(1+b g^2) with a={a_fit}, b={b_fit}.")
    lines.append("")
    lines.append("| sweep | rmse_raw_p00 | rmse_renorm_p00 | rmse_raw_coh | rmse_renorm_coh |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['rmse_raw_p00']:.6f} | {r['rmse_renorm_p00']:.6f} | "
            f"{r['rmse_raw_coh']:.6f} | {r['rmse_renorm_coh']:.6f} |"
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
