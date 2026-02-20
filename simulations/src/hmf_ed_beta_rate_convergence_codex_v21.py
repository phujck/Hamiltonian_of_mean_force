"""
ED truncation convergence for beta-flow diagnostics at theta=pi/2, g=0.5.

Computationally conservative:
- fixed n_modes=2
- varying n_cut
- moderate beta grid
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import BenchmarkConfig as LiteConfig, extract_density, ordered_gaussian_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state


def _d_from_p00(beta: float, omega_q: float, p00: float) -> float:
    a = 0.5 * beta * omega_q
    p = float(np.clip(p00, 1e-15, 1.0 - 1e-15))
    return float(a + 0.5 * np.log(p / (1.0 - p)))


def _build_ed(beta: float, theta: float, n_cut: int) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=2,
        n_cut=int(n_cut),
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_ed_beta_rate_convergence_codex_v21",
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_ed_beta_rate_convergence_scan_codex_v21.csv"
    summary_csv = out_dir / "hmf_ed_beta_rate_convergence_summary_codex_v21.csv"
    fig_png = out_dir / "hmf_ed_beta_rate_convergence_codex_v21.png"
    log_md = out_dir / "hmf_ed_beta_rate_convergence_log_codex_v21.md"

    betas = np.linspace(0.4, 6.0, 15)
    n_cuts = [3, 4, 5, 6, 8, 10]
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0

    rows: list[dict[str, float | int | str]] = []

    # Ordered reference on same beta grid (light settings).
    ord_rows: dict[float, float] = {}
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
        rho_o = ordered_gaussian_state(lite, g)
        p00_o, _, _ = extract_density(rho_o)
        ord_rows[float(beta)] = float(p00_o)

    for n_cut in n_cuts:
        for beta in betas:
            cfg = _build_ed(beta=beta, theta=theta, n_cut=n_cut)
            ctx = build_ed_context(cfg)
            rho = exact_reduced_state(ctx, g)
            p00, _p11, coh = extract_density(rho)
            p00_ord = ord_rows[float(beta)]
            rows.append(
                {
                    "beta": float(beta),
                    "n_cut": int(n_cut),
                    "ed_p00": float(p00),
                    "ed_coh": float(coh),
                    "ordered_p00": float(p00_ord),
                    "ed_minus_ordered_p00": float(p00 - p00_ord),
                    "d_ed": _d_from_p00(beta, omega_q, p00),
                    "d_ordered": _d_from_p00(beta, omega_q, p00_ord),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["n_cut", "beta"]).reset_index(drop=True)

    # Derivative diagnostics by n_cut
    deriv_rows: list[dict[str, float | int]] = []
    for n_cut, grp in df.groupby("n_cut"):
        gdf = grp.sort_values("beta")
        b = gdf["beta"].to_numpy(dtype=float)
        h = float(b[1] - b[0])
        dp = np.gradient(gdf["ed_p00"].to_numpy(dtype=float), h)
        dd = np.gradient(gdf["d_ed"].to_numpy(dtype=float), h)
        margin = dd - 0.5 * omega_q
        deriv_rows.append(
            {
                "n_cut": int(n_cut),
                "ed_p00_beta2": float(np.interp(2.0, b, gdf["ed_p00"].to_numpy(dtype=float))),
                "dp00_max": float(np.max(dp)),
                "dp00_min": float(np.min(dp)),
                "margin_max": float(np.max(margin)),
                "margin_min": float(np.min(margin)),
                "rmse_vs_ordered_p00": float(np.sqrt(np.mean(np.square(gdf["ed_minus_ordered_p00"].to_numpy(dtype=float))))),
            }
        )
        df.loc[gdf.index, "dp00_dbeta"] = dp
        df.loc[gdf.index, "dprime_minus_w2"] = margin

    summary = pd.DataFrame.from_records(deriv_rows).sort_values("n_cut").reset_index(drop=True)

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["ed_p00"], linewidth=1.6, label=f"ED n_cut={n_cut}")
    ax.plot(betas, [ord_rows[float(b)] for b in betas], color="black", linewidth=2.2, label="Ordered")
    ax.set_title("Population vs beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    ax = axes[0, 1]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["dp00_dbeta"], linewidth=1.6, label=f"n_cut={n_cut}")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("d rho_00 / d beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("derivative")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for n_cut in n_cuts:
        gdf = df[df["n_cut"] == n_cut].sort_values("beta")
        ax.plot(gdf["beta"], gdf["dprime_minus_w2"], linewidth=1.6, label=f"n_cut={n_cut}")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("d' - omega_q/2")
    ax.set_xlabel("beta")
    ax.set_ylabel("margin")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(summary["n_cut"], summary["ed_p00_beta2"], "o-", color="#0B6E4F", linewidth=2.0, label="ED p00 @ beta=2")
    ax.axhline(float(np.interp(2.0, betas, np.array([ord_rows[float(b)] for b in betas], dtype=float))), color="black", linewidth=2.0, label="Ordered @ beta=2")
    ax.set_title("Convergence at beta=2")
    ax.set_xlabel("n_cut")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# ED Beta-Rate Convergence (Codex v21)")
    lines.append("")
    lines.append("| n_cut | ed_p00_beta2 | dp00_max | dp00_min | margin_max | margin_min | rmse_vs_ordered_p00 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['n_cut'])} | {r['ed_p00_beta2']:.6f} | {r['dp00_max']:.6f} | {r['dp00_min']:.6f} | "
            f"{r['margin_max']:.6f} | {r['margin_min']:.6f} | {r['rmse_vs_ordered_p00']:.6f} |"
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
