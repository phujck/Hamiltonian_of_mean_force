"""
Population sweep diagnostic: ordered vs exact-naive vs exact-stable.

Focuses on the temperature sweep where turning behavior is most visible.
Writes only fresh codex_v17 outputs.
"""

from __future__ import annotations

from pathlib import Path

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


def _u_a(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float]:
    q0 = float(np.sqrt(max(sp * sm, 0.0)))
    chi = float(np.hypot(dz, q0))
    if chi <= 1e-15:
        u = float(dz)
    else:
        u = float(np.tanh(chi) * dz / chi)
    a = 0.5 * float(beta) * float(omega_q)
    return u, a


def _naive_p00(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> float:
    u, a = _u_a(beta, omega_q, sp, sm, dz)
    z = 2.0 * (np.cosh(a) - u * np.sinh(a))
    p00 = float(np.exp(-a) * (1.0 + u) / z)
    return p00


def _stable_p00(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> float:
    u, a = _u_a(beta, omega_q, sp, sm, dz)
    if u >= 1.0:
        return 1.0
    if u <= -1.0:
        return 0.0
    up = min(max(u, np.nextafter(-1.0, 0.0)), np.nextafter(1.0, 0.0))
    l0 = -a + np.log1p(up)
    l1 = +a + np.log1p(-up)
    m = max(l0, l1)
    w0 = np.exp(l0 - m)
    w1 = np.exp(l1 - m)
    return float(w0 / (w0 + w1))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_population_sweep_stable_scan_codex_v17.csv"
    summary_csv = out_dir / "hmf_population_sweep_stable_summary_codex_v17.csv"
    fig_png = out_dir / "hmf_population_sweep_stable_compare_codex_v17.png"
    log_md = out_dir / "hmf_population_sweep_stable_log_codex_v17.md"

    # Primary turning-point regime + one oblique-angle regime.
    sweeps = [
        {"name": "beta_theta_pi2_g05", "theta": float(np.pi / 2.0), "g": 0.5, "beta_grid": np.linspace(0.2, 6.0, 17)},
        {"name": "beta_theta_pi4_g05", "theta": float(np.pi / 4.0), "g": 0.5, "beta_grid": np.linspace(0.2, 10.0, 17)},
    ]

    rows: list[dict[str, float | str]] = []
    for s in sweeps:
        for beta in s["beta_grid"]:
            cfg = BenchmarkConfig(
                beta=float(beta),
                omega_q=2.0,
                theta=float(s["theta"]),
                n_modes=40,
                n_cut=1,
                omega_min=0.1,
                omega_max=10.0,
                q_strength=5.0,
                tau_c=0.5,
            )

            sp, sm, dz = _channels(cfg, float(s["g"]))
            p00_naive = _naive_p00(cfg.beta, cfg.omega_q, sp, sm, dz)
            p00_stable = _stable_p00(cfg.beta, cfg.omega_q, sp, sm, dz)
            p00_ord, _p11_ord, _coh_ord = extract_density(ordered_gaussian_state(cfg, float(s["g"])))

            rows.append(
                {
                    "sweep": str(s["name"]),
                    "beta": float(beta),
                    "theta": float(s["theta"]),
                    "g": float(s["g"]),
                    "ordered_p00": float(p00_ord),
                    "exact_naive_p00": float(p00_naive),
                    "exact_stable_p00": float(p00_stable),
                    "naive_minus_ordered": float(p00_naive - p00_ord),
                    "stable_minus_ordered": float(p00_stable - p00_ord),
                    "naive_minus_stable": float(p00_naive - p00_stable) if np.isfinite(p00_naive) else np.nan,
                    "naive_finite": float(np.isfinite(p00_naive)),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["sweep", "beta"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "rmse_naive_vs_ordered": _rmse(grp["naive_minus_ordered"].to_numpy(dtype=float)),
                "rmse_stable_vs_ordered": _rmse(grp["stable_minus_ordered"].to_numpy(dtype=float)),
                "max_abs_naive_minus_stable": float(np.nanmax(np.abs(grp["naive_minus_stable"].to_numpy(dtype=float)))),
                "naive_nonfinite_count": int(np.sum(grp["naive_finite"].to_numpy(dtype=float) < 0.5)),
                "beta_at_ordered_min": float(grp.iloc[np.argmin(grp["ordered_p00"].to_numpy(dtype=float))]["beta"]),
                "beta_at_stable_min": float(grp.iloc[np.argmin(grp["exact_stable_p00"].to_numpy(dtype=float))]["beta"]),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("sweep").reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for r, sweep in enumerate(summary["sweep"].tolist()):
        grp = df[df["sweep"] == sweep].sort_values("beta")
        x = grp["beta"].to_numpy(dtype=float)

        ax = axes[r, 0]
        ax.plot(x, grp["ordered_p00"], color="black", linewidth=2.0, label="Ordered")
        ax.plot(x, grp["exact_stable_p00"], color="#0B6E4F", linewidth=2.0, label="Exact stable")
        ax.plot(x, grp["exact_naive_p00"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact naive")
        ax.set_title(f"{sweep}: population")
        ax.set_xlabel("beta")
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)

        ax = axes[r, 1]
        ax.plot(x, grp["stable_minus_ordered"], color="#0B6E4F", linewidth=2.0, label="Stable - ordered")
        ax.plot(x, grp["naive_minus_ordered"], color="#C84B31", linestyle="--", linewidth=1.4, label="Naive - ordered")
        ax.axhline(0.0, color="#777777", linewidth=1.0)
        ax.set_title(f"{sweep}: mismatch")
        ax.set_xlabel("beta")
        ax.set_ylabel("delta rho_00")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Population Sweep Stable Comparison (Codex v17)")
    lines.append("")
    lines.append("| sweep | rmse_naive_vs_ordered | rmse_stable_vs_ordered | max_abs_naive_minus_stable | naive_nonfinite_count | beta_at_ordered_min | beta_at_stable_min |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['rmse_naive_vs_ordered']:.6f} | {r['rmse_stable_vs_ordered']:.6f} | "
            f"{r['max_abs_naive_minus_stable']:.6e} | {int(r['naive_nonfinite_count'])} | "
            f"{r['beta_at_ordered_min']:.3f} | {r['beta_at_stable_min']:.3f} |"
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
