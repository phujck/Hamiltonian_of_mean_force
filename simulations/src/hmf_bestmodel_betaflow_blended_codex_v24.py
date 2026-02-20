"""
Blended beta-flow best model.

Uses the diagonal flow-shaping branch fitted in v23, but blends it with raw exact
using an angle-selective weight to avoid degrading mixed-angle sectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    extract_density,
    ordered_gaussian_state,
)
from hmf_population_sweep_stable_compare_codex_v17 import _channels
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig


@dataclass
class SweepDef:
    name: str
    param: str
    values: np.ndarray
    beta: float | None
    theta: float | None
    g: float | None


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _build_ed(beta: float, theta: float, n_modes: int = 2, n_cut: int = 10) -> EDConfig:
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
        output_prefix="hmf_bestmodel_betaflow_blended_codex_v24",
    )


def _u_q_exact(beta: float, omega_q: float, sp: float, sm: float, dz: float) -> tuple[float, float]:
    q0 = float(np.sqrt(max(sp * sm, 0.0)))
    chi = float(np.hypot(dz, q0))
    if chi <= 1e-15:
        return float(np.clip(dz, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0))), float(q0)
    fac = float(np.tanh(chi) / chi)
    u = float(fac * dz)
    q = float(fac * q0)
    u = float(np.clip(u, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)))
    return u, q


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


def _flow_transform(x: float, c: float, b: float, mu: float) -> float:
    sx = 1.0 if x >= 0.0 else -1.0
    ax = abs(x)
    first = ax / (1.0 + c * ax) if c > 0.0 else ax
    second = b * np.tanh(mu * ax)
    return float(sx * (first - second))


def _blend_weight(theta: float, g: float) -> float:
    # Strong correction only near theta ~ pi/2 and moderate-to-strong g.
    s = float(np.sin(theta))
    w_theta = s**8
    w_g = min(1.0, (g / 0.5) ** 2)
    return float(w_theta * w_g)


def _p00_new(
    beta: float,
    omega_q: float,
    theta: float,
    g: float,
    sp: float,
    sm: float,
    dz: float,
    c_base: float,
    b_base: float,
    mu_base: float,
) -> tuple[float, float]:
    u_raw, q_raw = _u_q_exact(beta, omega_q, sp, sm, dz)
    a = 0.5 * beta * omega_q
    d_raw = float(np.arctanh(u_raw))
    x = d_raw - a

    x_corr = _flow_transform(x, c_base, b_base, mu_base)
    d_corr = a + x_corr
    w = _blend_weight(theta, g)
    d_eff = (1.0 - w) * d_raw + w * d_corr

    u_eff = float(np.tanh(d_eff))
    u_eff = float(np.clip(u_eff, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)))
    return _p00_coh_from_uq(beta, omega_q, u_eff, q_raw)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_bestmodel_betaflow_blended_scan_codex_v24.csv"
    summary_csv = out_dir / "hmf_bestmodel_betaflow_blended_summary_codex_v24.csv"
    fig_png = out_dir / "hmf_bestmodel_betaflow_blended_codex_v24.png"
    log_md = out_dir / "hmf_bestmodel_betaflow_blended_log_codex_v24.md"

    # From v23 anchor fit (case-level params at theta=pi/2,g=0.5).
    c_base = 2.645810
    b_base = 0.680000
    mu_base = 0.700000

    sweeps = [
        SweepDef(
            name="coupling",
            param="g",
            values=np.linspace(0.0, 2.0, 11),
            beta=2.0,
            theta=float(np.pi / 4.0),
            g=None,
        ),
        SweepDef(
            name="angle",
            param="theta",
            values=np.linspace(0.0, float(np.pi / 2.0), 11),
            beta=2.0,
            theta=None,
            g=0.5,
        ),
        SweepDef(
            name="temperature",
            param="beta",
            values=np.linspace(0.4, 6.0, 11),
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

            lite = LiteConfig(
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
            sp, sm, dz = _channels(lite, g)

            u_raw, q_raw = _u_q_exact(beta, lite.omega_q, sp, sm, dz)
            p00_raw, coh_raw = _p00_coh_from_uq(beta, lite.omega_q, u_raw, q_raw)

            p00_new, coh_new = _p00_new(
                beta, lite.omega_q, theta, g, sp, sm, dz, c_base, b_base, mu_base
            )

            rho_ord = ordered_gaussian_state(lite, g)
            p00_ord, _p11_ord, coh_ord = extract_density(rho_ord)

            ecfg = _build_ed(beta=beta, theta=theta, n_modes=2, n_cut=10)
            ed_ctx = build_ed_context(ecfg)
            rho_ed = exact_reduced_state(ed_ctx, g)
            p00_ed, _p11_ed, coh_ed = extract_density(rho_ed)

            rows.append(
                {
                    "sweep": sw.name,
                    "param_name": sw.param,
                    "param": float(p),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "weight": _blend_weight(theta, g),
                    "ordered_p00": float(p00_ord),
                    "ordered_coh": float(coh_ord),
                    "exact_raw_p00": float(p00_raw),
                    "exact_raw_coh": float(coh_raw),
                    "new_p00": float(p00_new),
                    "new_coh": float(coh_new),
                    "ed_p00": float(p00_ed),
                    "ed_coh": float(coh_ed),
                    "d_raw_vs_ed_p00": float(p00_raw - p00_ed),
                    "d_new_vs_ed_p00": float(p00_new - p00_ed),
                    "d_ord_vs_ed_p00": float(p00_ord - p00_ed),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["sweep", "param"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.extend(
            [
                {
                    "sweep": str(sweep),
                    "comparison": "raw_vs_ed",
                    "rmse_p00": _rmse(grp["d_raw_vs_ed_p00"]),
                    "rmse_coh": _rmse(grp["exact_raw_coh"] - grp["ed_coh"]),
                },
                {
                    "sweep": str(sweep),
                    "comparison": "new_vs_ed",
                    "rmse_p00": _rmse(grp["d_new_vs_ed_p00"]),
                    "rmse_coh": _rmse(grp["new_coh"] - grp["ed_coh"]),
                },
                {
                    "sweep": str(sweep),
                    "comparison": "ordered_vs_ed",
                    "rmse_p00": _rmse(grp["d_ord_vs_ed_p00"]),
                    "rmse_coh": _rmse(grp["ordered_coh"] - grp["ed_coh"]),
                },
            ]
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "comparison"]).reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), constrained_layout=True)
    for r, sweep in enumerate(["coupling", "angle", "temperature"]):
        grp = df[df["sweep"] == sweep].sort_values("param")
        x = grp["param"].to_numpy(dtype=float)
        xlabel = grp["param_name"].iloc[0]
        if sweep == "angle":
            x = x / np.pi
            xlabel = "theta/pi"

        ax = axes[r, 0]
        ax.plot(x, grp["ed_p00"], color="black", linewidth=2.0, label="ED m2/c10")
        ax.plot(x, grp["exact_raw_p00"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact raw")
        ax.plot(x, grp["new_p00"], color="#0B6E4F", linewidth=2.0, label="Best blended")
        ax.plot(x, grp["ordered_p00"], color="#1F4E79", linestyle=":", linewidth=1.8, label="Ordered")
        ax.set_title(f"{sweep}: population")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)

        ax = axes[r, 1]
        ax.plot(x, grp["ed_coh"], color="black", linewidth=2.0, label="ED m2/c10")
        ax.plot(x, grp["exact_raw_coh"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact raw")
        ax.plot(x, grp["new_coh"], color="#0B6E4F", linewidth=2.0, label="Best blended")
        ax.plot(x, grp["ordered_coh"], color="#1F4E79", linestyle=":", linewidth=1.8, label="Ordered")
        ax.set_title(f"{sweep}: coherence")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|rho_01|")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Best Model (Beta-Flow Blended) Codex v24")
    lines.append("")
    lines.append(f"Fixed branch params: c_base={c_base:.6f}, b_base={b_base:.6f}, mu_base={mu_base:.6f}")
    lines.append("Blend weight: w(theta,g) = sin(theta)^8 * min(1, (g/0.5)^2)")
    lines.append("")
    lines.append("| sweep | comparison | rmse_p00 | rmse_coh |")
    lines.append("|---|---|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['comparison']} | {r['rmse_p00']:.6f} | {r['rmse_coh']:.6f} |"
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
