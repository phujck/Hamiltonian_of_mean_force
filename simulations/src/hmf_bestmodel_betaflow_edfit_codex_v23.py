"""
Best-model construction via bounded beta-flow, fitted to ED anchor data.

Anchor fit:
  theta = pi/2, g = 0.5, beta sweep, ED case n_modes=2, n_cut=10.

Model (diagonal rapidity d):
  x = d_raw - a,  a = beta*omega_q/2
  f(x) = sign(x) * ( |x|/(1+c|x|) - b*tanh(mu|x|) )
  d_eff = a + f(x)
  u_eff = tanh(d_eff)

with angle/coupling scaling:
  c = c0 * g^2 * sin(theta)^2
  b = b0 * g^2 * sin(theta)^2
  mu = mu0
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
        output_prefix="hmf_bestmodel_betaflow_edfit_codex_v23",
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


def _p00_model(
    beta: float,
    omega_q: float,
    theta: float,
    g: float,
    sp: float,
    sm: float,
    dz: float,
    c0: float,
    b0: float,
    mu0: float,
) -> tuple[float, float]:
    u_raw, q_raw = _u_q_exact(beta, omega_q, sp, sm, dz)
    a = 0.5 * beta * omega_q
    d_raw = float(np.arctanh(u_raw))
    x = d_raw - a

    s2 = float(np.sin(theta) ** 2)
    g2 = float(g * g)
    c = max(1e-12, c0 * g2 * s2)
    b = b0 * g2 * s2
    mu = max(1e-12, mu0)

    x_eff = _flow_transform(x, c, b, mu)
    d_eff = a + x_eff
    u_eff = float(np.tanh(d_eff))
    u_eff = float(np.clip(u_eff, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0)))
    return _p00_coh_from_uq(beta, omega_q, u_eff, q_raw)


def _fit_anchor_params() -> tuple[float, float, float, pd.DataFrame]:
    betas = np.linspace(0.4, 6.0, 11)
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0

    anchor_rows: list[dict[str, float]] = []
    p00_ed = []
    ch = []
    for beta in betas:
        lite = LiteConfig(
            beta=float(beta),
            omega_q=omega_q,
            theta=theta,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        sp, sm, dz = _channels(lite, g)
        ch.append((sp, sm, dz))

        ecfg = _build_ed(beta=beta, theta=theta, n_modes=2, n_cut=10)
        ed_ctx = build_ed_context(ecfg)
        rho_ed = exact_reduced_state(ed_ctx, g)
        p00, _p11, _coh = extract_density(rho_ed)
        p00_ed.append(float(p00))
        anchor_rows.append({"beta": float(beta), "ed_p00": float(p00)})

    p00_ed = np.asarray(p00_ed, dtype=float)

    best = None
    # Fit case-level parameters first, then scale to c0,b0 by dividing by g^2 s^2 = 0.25.
    for c_case in np.linspace(0.2, 40.0, 180):
        for b_case in np.linspace(0.0, 3.0, 151):
            for mu0 in np.linspace(0.05, 4.0, 80):
                preds = []
                for beta, (sp, sm, dz) in zip(betas, ch):
                    # c0,b0 such that at this anchor c,b equal case values.
                    c0 = c_case / 0.25
                    b0 = b_case / 0.25
                    p00_m, _coh_m = _p00_model(beta, omega_q, theta, g, sp, sm, dz, c0, b0, mu0)
                    preds.append(p00_m)
                preds = np.asarray(preds, dtype=float)
                err = float(np.mean((preds - p00_ed) ** 2))
                if best is None or err < best[0]:
                    best = (err, c_case, b_case, mu0, preds)

    assert best is not None
    _err, c_case, b_case, mu0, preds = best
    c0 = float(c_case / 0.25)
    b0 = float(b_case / 0.25)

    for i, beta in enumerate(betas):
        anchor_rows[i]["model_p00"] = float(preds[i])
        anchor_rows[i]["delta_model_minus_ed"] = float(preds[i] - p00_ed[i])

    return c0, b0, float(mu0), pd.DataFrame.from_records(anchor_rows)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_bestmodel_betaflow_edfit_scan_codex_v23.csv"
    summary_csv = out_dir / "hmf_bestmodel_betaflow_edfit_summary_codex_v23.csv"
    anchor_csv = out_dir / "hmf_bestmodel_betaflow_anchorfit_codex_v23.csv"
    fig_png = out_dir / "hmf_bestmodel_betaflow_edfit_codex_v23.png"
    log_md = out_dir / "hmf_bestmodel_betaflow_edfit_log_codex_v23.md"

    c0, b0, mu0, anchor = _fit_anchor_params()
    anchor.to_csv(anchor_csv, index=False)

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

            # Raw exact
            u_raw, q_raw = _u_q_exact(beta, lite.omega_q, sp, sm, dz)
            p00_raw, coh_raw = _p00_coh_from_uq(beta, lite.omega_q, u_raw, q_raw)

            # New model
            p00_new, coh_new = _p00_model(beta, lite.omega_q, theta, g, sp, sm, dz, c0, b0, mu0)

            # Ordered
            rho_ord = ordered_gaussian_state(lite, g)
            p00_ord, _p11_ord, coh_ord = extract_density(rho_ord)

            # ED anchor style (fixed m2/c10 for all sweeps)
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
                    "d_raw_vs_ord_p00": float(p00_raw - p00_ord),
                    "d_new_vs_ord_p00": float(p00_new - p00_ord),
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
                {
                    "sweep": str(sweep),
                    "comparison": "raw_vs_ordered",
                    "rmse_p00": _rmse(grp["d_raw_vs_ord_p00"]),
                    "rmse_coh": _rmse(grp["exact_raw_coh"] - grp["ordered_coh"]),
                },
                {
                    "sweep": str(sweep),
                    "comparison": "new_vs_ordered",
                    "rmse_p00": _rmse(grp["d_new_vs_ord_p00"]),
                    "rmse_coh": _rmse(grp["new_coh"] - grp["ordered_coh"]),
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
        ax.plot(x, grp["new_p00"], color="#0B6E4F", linewidth=2.0, label="Best beta-flow")
        ax.plot(x, grp["ordered_p00"], color="#1F4E79", linestyle=":", linewidth=1.8, label="Ordered")
        ax.set_title(f"{sweep}: population")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("rho_00")
        ax.grid(alpha=0.25)

        ax = axes[r, 1]
        ax.plot(x, grp["ed_coh"], color="black", linewidth=2.0, label="ED m2/c10")
        ax.plot(x, grp["exact_raw_coh"], color="#C84B31", linestyle="--", linewidth=1.4, label="Exact raw")
        ax.plot(x, grp["new_coh"], color="#0B6E4F", linewidth=2.0, label="Best beta-flow")
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
    lines.append("# Best Model (Beta-Flow, ED-Fit) Codex v23")
    lines.append("")
    lines.append(f"Fitted params: c0={c0:.6f}, b0={b0:.6f}, mu0={mu0:.6f}")
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
    print("Wrote:", anchor_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(f"Fitted params: c0={c0:.6f}, b0={b0:.6f}, mu0={mu0:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
