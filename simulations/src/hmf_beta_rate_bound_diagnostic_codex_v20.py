"""
Beta-flow diagnostic for diagonal population turning in theta=pi/2.

Core identity for q=0 channel:
    p00 = 1 / (1 + exp(2(a - d)))
    a = beta * omega_q / 2
    d = atanh(u),  u = gamma * Delta_z

Turning/sign of dp00/dbeta is controlled by:
    dp00/dbeta = 2 p00 (1-p00) (d' - omega_q/2).

This script:
1) computes d_raw (from exact channel), d_ord (inferred from ordered p00),
2) compares slopes d' and margin m=d'-omega_q/2,
3) tests a bounded-flow ansatz on excess x=d_raw-a:
      x_eff = x / (1 + c |x|),   d_eff = a + x_eff
   where c is fit to ordered d(beta).
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


def _p00_from_d(a: float, d: float) -> float:
    z = 1.0 + np.exp(2.0 * (a - d))
    return float(1.0 / z)


def _d_from_p00(beta: float, omega_q: float, p00: float) -> float:
    # p00 = 1/(1+e^{2(a-d)}) => d = a + 0.5*log(p00/(1-p00))
    a = 0.5 * beta * omega_q
    p = float(np.clip(p00, 1e-15, 1.0 - 1e-15))
    return float(a + 0.5 * np.log(p / (1.0 - p)))


def _fit_c_excess(a: np.ndarray, d_raw: np.ndarray, d_target: np.ndarray) -> float:
    x = d_raw - a
    # x_eff = x / (1 + c|x|), d_eff = a + x_eff
    best = None
    for c in np.linspace(0.0, 2000.0, 4001):
        x_eff = x / (1.0 + c * np.abs(x))
        d_eff = a + x_eff
        err = float(np.mean((d_eff - d_target) ** 2))
        if best is None or err < best[0]:
            best = (err, c)
    assert best is not None
    return float(best[1])


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_beta_rate_bound_scan_codex_v20.csv"
    summary_csv = out_dir / "hmf_beta_rate_bound_summary_codex_v20.csv"
    fig_png = out_dir / "hmf_beta_rate_bound_diagnostic_codex_v20.png"
    log_md = out_dir / "hmf_beta_rate_bound_log_codex_v20.md"

    omega_q = 2.0
    g = 0.5
    theta = float(np.pi / 2.0)
    betas = np.linspace(0.2, 6.0, 17)

    rows: list[dict[str, float]] = []
    for beta in betas:
        cfg = BenchmarkConfig(
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

        _sp, _sm, dz = _channels(cfg, g)
        # At theta=pi/2, q=0, so d_raw = dz exactly.
        d_raw = float(dz)
        a = 0.5 * beta * omega_q
        p00_raw = _p00_from_d(a, d_raw)

        p00_ord, _p11_ord, _coh_ord = extract_density(ordered_gaussian_state(cfg, g))
        d_ord = _d_from_p00(beta, omega_q, p00_ord)

        rows.append(
            {
                "beta": float(beta),
                "a": float(a),
                "d_raw": d_raw,
                "p00_raw": p00_raw,
                "p00_ordered": float(p00_ord),
                "d_ordered": float(d_ord),
            }
        )

    df = pd.DataFrame.from_records(rows).sort_values("beta").reset_index(drop=True)
    b = df["beta"].to_numpy(dtype=float)
    a = df["a"].to_numpy(dtype=float)
    d_raw = df["d_raw"].to_numpy(dtype=float)
    d_ord = df["d_ordered"].to_numpy(dtype=float)

    c_fit = _fit_c_excess(a, d_raw, d_ord)
    x = d_raw - a
    x_eff = x / (1.0 + c_fit * np.abs(x))
    d_eff = a + x_eff
    p00_eff = np.array([_p00_from_d(ai, di) for ai, di in zip(a, d_eff)], dtype=float)

    h = float(b[1] - b[0])
    d_raw_p = np.gradient(d_raw, h)
    d_ord_p = np.gradient(d_ord, h)
    d_eff_p = np.gradient(d_eff, h)
    margin_raw = d_raw_p - 0.5 * omega_q
    margin_ord = d_ord_p - 0.5 * omega_q
    margin_eff = d_eff_p - 0.5 * omega_q

    df["d_eff"] = d_eff
    df["p00_eff"] = p00_eff
    df["d_raw_prime"] = d_raw_p
    df["d_ordered_prime"] = d_ord_p
    df["d_eff_prime"] = d_eff_p
    df["margin_raw"] = margin_raw
    df["margin_ordered"] = margin_ord
    df["margin_eff"] = margin_eff
    df["dp00_raw"] = np.gradient(df["p00_raw"].to_numpy(dtype=float), h)
    df["dp00_ordered"] = np.gradient(df["p00_ordered"].to_numpy(dtype=float), h)
    df["dp00_eff"] = np.gradient(df["p00_eff"].to_numpy(dtype=float), h)

    summary = pd.DataFrame.from_records(
        [
            {
                "c_fit_excess": float(c_fit),
                "raw_margin_max": float(np.max(margin_raw)),
                "ordered_margin_max": float(np.max(margin_ord)),
                "eff_margin_max": float(np.max(margin_eff)),
                "raw_margin_min": float(np.min(margin_raw)),
                "ordered_margin_min": float(np.min(margin_ord)),
                "eff_margin_min": float(np.min(margin_eff)),
                "rmse_p00_eff_vs_ordered": float(np.sqrt(np.mean((p00_eff - df["p00_ordered"].to_numpy(dtype=float)) ** 2))),
                "rmse_p00_raw_vs_ordered": float(np.sqrt(np.mean((df["p00_raw"].to_numpy(dtype=float) - df["p00_ordered"].to_numpy(dtype=float)) ** 2))),
            }
        ]
    )

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(b, df["p00_ordered"], color="black", linewidth=2.0, label="Ordered")
    ax.plot(b, df["p00_raw"], color="#C84B31", linestyle="--", linewidth=1.6, label="Raw exact")
    ax.plot(b, df["p00_eff"], color="#0B6E4F", linewidth=2.0, label="Rate-bounded")
    ax.set_title("Population vs beta")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(b, margin_raw, color="#C84B31", linestyle="--", linewidth=1.6, label="raw: d'-w/2")
    ax.plot(b, margin_ord, color="black", linewidth=2.0, label="ordered: d'-w/2")
    ax.plot(b, margin_eff, color="#0B6E4F", linewidth=2.0, label="bounded: d'-w/2")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("Rate condition for turning")
    ax.set_xlabel("beta")
    ax.set_ylabel("margin")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(b, d_raw, color="#C84B31", linestyle="--", linewidth=1.6, label="d raw")
    ax.plot(b, d_ord, color="black", linewidth=2.0, label="d ordered inferred")
    ax.plot(b, d_eff, color="#0B6E4F", linewidth=2.0, label="d bounded")
    ax.plot(b, a, color="#AA3377", linestyle=":", linewidth=1.8, label="a=beta*w/2")
    ax.set_title("Diagonal rapidity")
    ax.set_xlabel("beta")
    ax.set_ylabel("d")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(b, df["dp00_raw"], color="#C84B31", linestyle="--", linewidth=1.6, label="d p00/d beta raw")
    ax.plot(b, df["dp00_ordered"], color="black", linewidth=2.0, label="d p00/d beta ordered")
    ax.plot(b, df["dp00_eff"], color="#0B6E4F", linewidth=2.0, label="d p00/d beta bounded")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("Population derivative")
    ax.set_xlabel("beta")
    ax.set_ylabel("derivative")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Beta Rate-Bound Diagnostic (Codex v20)")
    lines.append("")
    r = summary.iloc[0]
    lines.append(f"- c_fit_excess: {r['c_fit_excess']:.6f}")
    lines.append(f"- raw_margin_max: {r['raw_margin_max']:.6f}")
    lines.append(f"- ordered_margin_max: {r['ordered_margin_max']:.6f}")
    lines.append(f"- eff_margin_max: {r['eff_margin_max']:.6f}")
    lines.append(f"- rmse_p00_raw_vs_ordered: {r['rmse_p00_raw_vs_ordered']:.6f}")
    lines.append(f"- rmse_p00_eff_vs_ordered: {r['rmse_p00_eff_vs_ordered']:.6f}")
    lines.append("")
    lines.append("Interpretation: sign flip requires d' > omega_q/2; bounded-flow ansatz damps excess slope.")
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
