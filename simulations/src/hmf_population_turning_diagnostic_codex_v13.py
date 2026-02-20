"""
Turning-point diagnostic for beta sweep at theta=pi/2 (Codex v13).

Goal:
- Explain why population p00(beta) bends/turns in the compact analytic model.
- Compare Best Analytic vs Ordered vs ED(light).
- Expose diagonal competition condition via derivatives.

Key identity at theta=pi/2 (sigma_+=sigma_-=0):
    m_z = (u - t) / (1 - u t),  t = tanh(a), a=beta*omega/2, u=tanh(delta_eff)
    p00 = (1 + m_z)/2
and equivalently
    m_z = tanh(delta_eff - a)
so turning comes from slope balance:
    d(delta_eff)/d(beta)  vs  omega/2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
    ordered_gaussian_state,
    laplace_k0,
    resonant_r0,
)
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state


def _central_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    if n <= 1:
        return out
    out[0] = (y[1] - y[0]) / max(x[1] - x[0], 1e-14)
    out[-1] = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-14)
    for i in range(1, n - 1):
        out[i] = (y[i + 1] - y[i - 1]) / max(x[i + 1] - x[i - 1], 1e-14)
    return out


def _base_delta0(cfg: LiteConfig) -> float:
    s = float(np.sin(cfg.theta))
    rp = resonant_r0(cfg, cfg.omega_q, 1001)
    rm = resonant_r0(cfg, -cfg.omega_q, 1001)
    return float((s**2) * 0.5 * (rp - rm))


def _best_model_theta_pi2(cfg: LiteConfig, g: float, ren: RenormConfig) -> tuple[float, float, float]:
    # theta=pi/2 expected; coherence channel vanishes.
    a = 0.5 * cfg.beta * cfg.omega_q
    dz0 = _base_delta0(cfg)
    dz_raw = ren.scale * (g * g) * dz0
    chi_raw = abs(dz_raw)
    if chi_raw <= ren.eps:
        run = 1.0
    else:
        chi_cap = max(ren.kappa * abs(a), ren.eps)
        run = 1.0 / (1.0 + chi_raw / chi_cap)
    dz_eff = run * dz_raw
    u = float(np.tanh(dz_eff))
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-14:
        den = 1e-14 if den >= 0.0 else -1e-14
    mz = (u - t) / den
    p00 = float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))
    p11 = float(1.0 - p00)
    ratio = float(p00 / max(p11, 1e-15))
    return p00, ratio, float(dz_eff)


def _ed_light(beta: float, theta: float, g: float) -> tuple[float, float]:
    cfg = EDConfig(
        beta=float(beta),
        omega_q=2.0,
        theta=float(theta),
        n_modes=2,
        n_cut=3,
        omega_min=0.1,
        omega_max=8.0,
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=2.0,
        lambda_points=3,
        output_prefix="hmf_turning_v13",
    )
    ctx = build_ed_context(cfg)
    rho = exact_reduced_state(ctx, float(g))
    p00, p11, _ = extract_density(rho)
    return float(p00), float(p00 / max(p11, 1e-15))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ren = RenormConfig(scale=1.04, kappa=0.94)
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0
    betas = np.linspace(0.2, 6.0, 21)

    rows = []
    for beta in betas:
        cfg = LiteConfig(
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
        p00_ord, p11_ord, coh_ord = extract_density(ordered_gaussian_state(cfg, g))
        p00_best, ratio_best, dz_eff = _best_model_theta_pi2(cfg, g, ren)
        p00_ed, ratio_ed = _ed_light(beta, theta, g)

        a = 0.5 * beta * omega_q
        rows.append(
            {
                "beta": float(beta),
                "a_beta": float(a),
                "t_tanh_a": float(np.tanh(a)),
                "ordered_p00": float(p00_ord),
                "ordered_ratio": float(p00_ord / max(p11_ord, 1e-15)),
                "ordered_coh": float(coh_ord),
                "best_p00": float(p00_best),
                "best_ratio": float(ratio_best),
                "best_delta_eff": float(dz_eff),
                "ed_p00_light": float(p00_ed),
                "ed_ratio_light": float(ratio_ed),
            }
        )

    df = pd.DataFrame.from_records(rows).sort_values("beta").reset_index(drop=True)
    beta_arr = df["beta"].to_numpy(dtype=float)
    df["d_best_p00_dbeta"] = _central_diff(beta_arr, df["best_p00"].to_numpy(dtype=float))
    df["d_ordered_p00_dbeta"] = _central_diff(beta_arr, df["ordered_p00"].to_numpy(dtype=float))
    df["d_ed_p00_dbeta"] = _central_diff(beta_arr, df["ed_p00_light"].to_numpy(dtype=float))
    df["d_deltaeff_dbeta"] = _central_diff(beta_arr, df["best_delta_eff"].to_numpy(dtype=float))
    df["slope_balance_minus_w2"] = df["d_deltaeff_dbeta"] - 0.5 * omega_q
    df["best_minus_ordered_p00"] = df["best_p00"] - df["ordered_p00"]
    df["best_minus_ed_p00"] = df["best_p00"] - df["ed_p00_light"]

    # Detect turning points by sign change of derivative.
    def _turn_points(yprime: np.ndarray, x: np.ndarray) -> list[float]:
        out: list[float] = []
        s = np.sign(yprime)
        for i in range(1, len(s)):
            if s[i] == 0:
                continue
            if s[i - 1] == 0:
                continue
            if s[i] * s[i - 1] < 0:
                out.append(float(0.5 * (x[i] + x[i - 1])))
        return out

    turns_best = _turn_points(df["d_best_p00_dbeta"].to_numpy(), beta_arr)
    turns_ord = _turn_points(df["d_ordered_p00_dbeta"].to_numpy(), beta_arr)
    turns_ed = _turn_points(df["d_ed_p00_dbeta"].to_numpy(), beta_arr)

    scan_csv = out_dir / "hmf_population_turning_scan_codex_v13.csv"
    fig_png = out_dir / "hmf_population_turning_diag_codex_v13.png"
    log_md = out_dir / "hmf_population_turning_log_codex_v13.md"
    df.to_csv(scan_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Panel 1: populations
    ax = axes[0, 0]
    ax.plot(df["beta"], df["ed_p00_light"], color="black", linewidth=2.2, label="ED (light)")
    ax.plot(df["beta"], df["ordered_p00"], color="#C84B31", linestyle="--", linewidth=1.8, label="Ordered")
    ax.plot(df["beta"], df["best_p00"], color="#0B6E4F", linewidth=2.0, label="Best analytic")
    ax.set_title(r"Population vs $\beta$ at $\theta=\pi/2,\ g=0.5$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Panel 2: derivative of populations
    ax = axes[0, 1]
    ax.plot(df["beta"], df["d_ed_p00_dbeta"], color="black", linewidth=2.0, label=r"$d\rho_{00}^{ED}/d\beta$")
    ax.plot(df["beta"], df["d_ordered_p00_dbeta"], color="#C84B31", linestyle="--", linewidth=1.8, label=r"$d\rho_{00}^{ord}/d\beta$")
    ax.plot(df["beta"], df["d_best_p00_dbeta"], color="#0B6E4F", linewidth=2.0, label=r"$d\rho_{00}^{best}/d\beta$")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.set_title("Turning points occur where derivative crosses zero")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("derivative")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Panel 3: diagonal competition in best model
    ax = axes[1, 0]
    ax.plot(df["beta"], df["best_delta_eff"], color="#1F4E79", linewidth=2.0, label=r"$\Delta_{\mathrm{eff}}(\beta)$")
    ax.plot(df["beta"], df["a_beta"], color="#AA3377", linestyle="--", linewidth=1.8, label=r"$a(\beta)=\beta\omega_q/2$")
    ax.set_title(r"Competing diagonal terms in $m_z=\tanh(\Delta_{\mathrm{eff}}-a)$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Panel 4: slope-balance condition
    ax = axes[1, 1]
    ax.plot(df["beta"], df["d_deltaeff_dbeta"], color="#1F4E79", linewidth=2.0, label=r"$d\Delta_{\mathrm{eff}}/d\beta$")
    ax.axhline(0.5 * omega_q, color="#AA3377", linestyle="--", linewidth=1.8, label=r"$\omega_q/2$")
    ax.plot(df["beta"], df["slope_balance_minus_w2"], color="#0B6E4F", linewidth=1.5, label=r"$d\Delta_{\mathrm{eff}}/d\beta-\omega_q/2$")
    ax.axhline(0.0, color="#888888", linewidth=1.0)
    ax.set_title("Turning driver in best model")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("slope")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Population Turning Diagnostic (Codex v13)")
    lines.append("")
    lines.append("Configuration: theta=pi/2, g=0.5, omega_q=2.0")
    lines.append(f"Detected turning betas (best): {turns_best}")
    lines.append(f"Detected turning betas (ordered): {turns_ord}")
    lines.append(f"Detected turning betas (ED light): {turns_ed}")
    lines.append("")
    lines.append("Interpretation: in best model at theta=pi/2, m_z=tanh(Delta_eff-a).")
    lines.append("Turning of p00(beta) is controlled by slope balance dDelta_eff/dbeta crossing omega_q/2.")
    lines.append("")
    lines.append("| beta | best_p00 | ordered_p00 | ed_p00_light | d_best_p00/dbeta | dDelta_eff/dbeta | slope_balance_minus_w2 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['beta']:.3f} | {r['best_p00']:.6f} | {r['ordered_p00']:.6f} | {r['ed_p00_light']:.6f} | "
            f"{r['d_best_p00_dbeta']:.6f} | {r['d_deltaeff_dbeta']:.6f} | {r['slope_balance_minus_w2']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(f"Turning betas best={turns_best}, ordered={turns_ord}, ed_light={turns_ed}")


if __name__ == "__main__":
    main()

