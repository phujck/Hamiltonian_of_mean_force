"""
Three-mode, three-level test with non-uniform resonant node placement.

Hypothesis tested:
- Custom nodes focused on {~0, omega_q, 2*omega_q} may better capture the key response
  structure than uniform spacing.

Fair comparison:
- ED uses the exact custom discrete spectrum.
- Analytic branch is computed with the exact same discrete kernel nodes/weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import RenormConfig
from prl127_qubit_benchmark import (
    SIGMA_X,
    SIGMA_Z,
    IDENTITY_2,
    annihilation_operator,
    embed_mode_operator,
    thermal_state,
    partial_trace_bath,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    omegas: np.ndarray
    n_cut: int


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def _extract_density(rho: np.ndarray) -> tuple[float, float, float]:
    p00 = float(np.real(rho[0, 0]))
    p11 = float(np.real(rho[1, 1]))
    coh = float(abs(rho[0, 1]))
    z = p00 + p11
    if z <= 0.0:
        return 0.5, 0.5, 0.0
    p00 /= z
    p11 /= z
    return p00, p11, coh


def _mode_weights(omegas: np.ndarray) -> np.ndarray:
    # Non-uniform trapezoidal quadrature weights on sorted nodes.
    w = np.asarray(sorted(omegas.tolist()), dtype=float)
    n = len(w)
    if n == 1:
        return np.array([1.0], dtype=float)
    out = np.zeros(n, dtype=float)
    out[0] = 0.5 * (w[1] - w[0])
    out[-1] = 0.5 * (w[-1] - w[-2])
    for i in range(1, n - 1):
        out[i] = 0.5 * (w[i + 1] - w[i - 1])
    return out


def _spectral_density(omega: np.ndarray, q_strength: float, tau_c: float) -> np.ndarray:
    return q_strength * tau_c * omega * np.exp(-tau_c * omega)


def _couplings_from_nodes(omegas: np.ndarray, q_strength: float, tau_c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(sorted(omegas.tolist()), dtype=float)
    weights = _mode_weights(w)
    j = _spectral_density(w, q_strength=q_strength, tau_c=tau_c)
    g2 = np.maximum(j, 0.0) * weights
    gk = np.sqrt(np.maximum(g2, 0.0))
    return w, gk, g2


def _kernel_profile(beta: float, u_abs: np.ndarray, omegas: np.ndarray, g2: np.ndarray) -> np.ndarray:
    k = np.zeros_like(u_abs, dtype=float)
    for wk, g2k in zip(omegas, g2):
        den = np.sinh(0.5 * beta * wk)
        if abs(den) < 1e-14:
            k += (2.0 * g2k) / max(beta * wk, 1e-14)
        else:
            k += g2k * np.cosh(wk * (0.5 * beta - u_abs)) / den
    return k


def _laplace_k0(beta: float, omega: float, omegas: np.ndarray, g2: np.ndarray, n_grid: int = 2001) -> float:
    u = np.linspace(0.0, beta, int(n_grid), dtype=float)
    k = _kernel_profile(beta, u, omegas, g2)
    return float(np.trapezoid(k * np.exp(omega * u), u))


def _resonant_r0(beta: float, omega: float, omegas: np.ndarray, g2: np.ndarray, n_grid: int = 2001) -> float:
    u = np.linspace(0.0, beta, int(n_grid), dtype=float)
    k = _kernel_profile(beta, u, omegas, g2)
    return float(np.trapezoid((beta - u) * k * np.exp(omega * u), u))


def _analytic_discrete_density(
    beta: float,
    omega_q: float,
    theta: float,
    g: float,
    omegas: np.ndarray,
    g2_nodes: np.ndarray,
    renorm: RenormConfig,
) -> tuple[float, float, float]:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    wq = float(omega_q)

    k0_0 = _laplace_k0(beta, 0.0, omegas, g2_nodes)
    k0_p = _laplace_k0(beta, wq, omegas, g2_nodes)
    k0_m = _laplace_k0(beta, -wq, omegas, g2_nodes)
    r_p = _resonant_r0(beta, wq, omegas, g2_nodes)
    r_m = _resonant_r0(beta, -wq, omegas, g2_nodes)

    sigma_plus0 = (c * s / wq) * ((1.0 + np.exp(beta * wq)) * k0_0 - 2.0 * k0_p)
    sigma_minus0 = (c * s / wq) * ((1.0 + np.exp(-beta * wq)) * k0_0 - 2.0 * k0_m)
    delta_z0 = (s**2) * 0.5 * (r_p - r_m)

    g2 = float(g * g)
    sp = float(g2 * sigma_plus0)
    sm = float(g2 * sigma_minus0)
    dz = float(g2 * delta_z0)

    chi_raw = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    if chi_raw > renorm.eps:
        a = 0.5 * beta * wq
        chi_cap = max(renorm.kappa * abs(a), renorm.eps)
        run = 1.0 / (1.0 + chi_raw / chi_cap)
        sp *= run
        sm *= run
        dz *= run

    chi = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-12 else 1.0
    u = gamma * dz
    q = gamma * np.sqrt(max(sp * sm, 0.0))

    a = 0.5 * beta * wq
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-14:
        den = 1e-14 if den >= 0 else -1e-14

    mz = (u - t) / den
    mx = (q / np.cosh(a)) / den

    p00 = 0.5 * (1.0 + mz)
    p11 = 0.5 * (1.0 - mz)
    coh = 0.5 * abs(mx)
    p00 = float(np.clip(p00, 0.0, 1.0))
    p11 = float(np.clip(p11, 0.0, 1.0))
    z = p00 + p11
    if z <= 0.0:
        p00, p11 = 0.5, 0.5
    else:
        p00 /= z
        p11 /= z
    return p00, p11, float(coh)


def _ed_density_custom_nodes(
    beta: float,
    omega_q: float,
    theta: float,
    g: float,
    omegas: np.ndarray,
    gk: np.ndarray,
    g2_nodes: np.ndarray,
    n_cut: int,
) -> tuple[float, float, float]:
    n_modes = len(omegas)
    h_s = 0.5 * omega_q * SIGMA_Z
    x_op = np.cos(theta) * SIGMA_Z - np.sin(theta) * SIGMA_X
    xx = x_op @ x_op

    dim_b = int(n_cut ** n_modes)
    id_b = np.eye(n_cut, dtype=complex)
    a_single = annihilation_operator(n_cut)
    adag_single = a_single.conj().T

    h_b = np.zeros((dim_b, dim_b), dtype=complex)
    b_op = np.zeros((dim_b, dim_b), dtype=complex)
    for m in range(n_modes):
        a_m = embed_mode_operator(a_single, m, n_modes, n_cut)
        adag_m = embed_mode_operator(adag_single, m, n_modes, n_cut)
        n_m = adag_m @ a_m
        h_b += omegas[m] * n_m
        b_op += gk[m] * (a_m + adag_m)

    q_reorg = float(np.sum(g2_nodes / np.maximum(omegas, 1e-12)))
    h_static = np.kron(h_s, np.eye(dim_b, dtype=complex)) + np.kron(IDENTITY_2, h_b)
    h_int = np.kron(x_op, b_op)
    h_counter = np.kron(xx, np.eye(dim_b, dtype=complex)) * q_reorg
    h_tot = h_static + g * h_int + (g * g) * h_counter

    rho_tot = thermal_state(h_tot, beta)
    rho_s = partial_trace_bath(rho_tot, dim_system=2, dim_bath=dim_b)
    return _extract_density(rho_s)


def run_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    omega_q = 1.0
    theta = float(np.pi / 2.0)
    g = 0.5
    q_strength = 5.0
    tau_c = 0.5
    n_cut = 3
    ren = RenormConfig(scale=1.04, kappa=0.94)
    betas = np.linspace(0.6, 10.0, 17, dtype=float)

    # Control vs resonant-focused custom nodes.
    cases = [
        CaseSpec(name="uniform_[0.2,1.0,1.8]", omegas=np.array([0.2, 1.0, 1.8], dtype=float), n_cut=n_cut),
        CaseSpec(name="resonant_[0.05,1.0,2.0]", omegas=np.array([0.05, 1.0, 2.0], dtype=float), n_cut=n_cut),
        CaseSpec(name="resonant_[0.05,1.0,1.8]", omegas=np.array([0.05, 1.0, 1.8], dtype=float), n_cut=n_cut),
    ]

    rows: list[dict[str, float | str | int]] = []
    for case in cases:
        omegas, gk, g2_nodes = _couplings_from_nodes(case.omegas, q_strength=q_strength, tau_c=tau_c)
        for beta in betas:
            ed_p00, _ed_p11, ed_coh = _ed_density_custom_nodes(
                beta=beta,
                omega_q=omega_q,
                theta=theta,
                g=g,
                omegas=omegas,
                gk=gk,
                g2_nodes=g2_nodes,
                n_cut=case.n_cut,
            )
            an_p00, _an_p11, an_coh = _analytic_discrete_density(
                beta=beta,
                omega_q=omega_q,
                theta=theta,
                g=g,
                omegas=omegas,
                g2_nodes=g2_nodes,
                renorm=ren,
            )
            rows.append(
                {
                    "case": case.name,
                    "beta": float(beta),
                    "n_modes": int(len(omegas)),
                    "n_cut": int(case.n_cut),
                    "omega_nodes": ";".join(f"{w:.6f}" for w in omegas.tolist()),
                    "ed_p00": float(ed_p00),
                    "ed_coh": float(ed_coh),
                    "analytic_disc_p00": float(an_p00),
                    "analytic_disc_coh": float(an_coh),
                    "d_p00": float(ed_p00 - an_p00),
                    "d_coh": float(ed_coh - an_coh),
                }
            )

    df = pd.DataFrame.from_records(rows).sort_values(["case", "beta"]).reset_index(drop=True)

    summary_rows: list[dict[str, float | str | int]] = []
    for case, grp in df.groupby("case"):
        summary_rows.append(
            {
                "case": str(case),
                "n_modes": int(grp["n_modes"].iloc[0]),
                "n_cut": int(grp["n_cut"].iloc[0]),
                "omega_nodes": str(grp["omega_nodes"].iloc[0]),
                "rmse_ed_vs_disc_p00": _rmse(grp["d_p00"].to_numpy(float)),
                "rmse_ed_vs_disc_coh": _rmse(grp["d_coh"].to_numpy(float)),
                "p00_at_beta2_ed": float(np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
                "p00_at_beta2_an": float(np.interp(2.0, grp["beta"].to_numpy(float), grp["analytic_disc_p00"].to_numpy(float))),
                "p00_at_beta8_ed": float(np.interp(8.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))),
                "p00_at_beta8_an": float(np.interp(8.0, grp["beta"].to_numpy(float), grp["analytic_disc_p00"].to_numpy(float))),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("rmse_ed_vs_disc_p00").reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_omega1_resonant_nodes_threemode_scan_codex_v41.csv"
    summary_csv = out_dir / "hmf_omega1_resonant_nodes_threemode_summary_codex_v41.csv"
    fig_png = out_dir / "hmf_omega1_resonant_nodes_threemode_codex_v41.png"
    log_md = out_dir / "hmf_omega1_resonant_nodes_threemode_log_codex_v41.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), constrained_layout=True)
    ax = axes[0]
    for case, grp in df.groupby("case"):
        g = grp.sort_values("beta")
        ax.plot(g["beta"], g["ed_p00"], linewidth=2.0, label=f"ED: {case}")
        ax.plot(g["beta"], g["analytic_disc_p00"], linestyle="--", linewidth=1.8, label=f"Analytic: {case}")
    ax.set_title("Three-mode custom-node test (n_cut=3): population")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    for case, grp in df.groupby("case"):
        g = grp.sort_values("beta")
        ax.plot(g["beta"], g["d_p00"], linewidth=2.0, label=case)
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_title("Mismatch: ED - analytic(discrete)")
    ax.set_xlabel("beta")
    ax.set_ylabel("delta rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    best = summary.iloc[0]
    lines: list[str] = []
    lines.append("# Three-Mode Resonant-Node Test (Codex v41)")
    lines.append("")
    lines.append("Setup: omega_q=1, theta=pi/2, g=0.5, n_modes=3, n_cut=3.")
    lines.append("Analytic branch uses the exact same custom discrete kernel nodes as ED.")
    lines.append("")
    lines.append(
        f"Best node set by fair metric: `{best['case']}` with rmse_ed_vs_disc_p00={best['rmse_ed_vs_disc_p00']:.6f}"
    )
    lines.append("")
    lines.append("| case | omega_nodes | rmse_ed_vs_disc_p00 | rmse_ed_vs_disc_coh | p00_at_beta2_ed | p00_at_beta2_an | p00_at_beta8_ed | p00_at_beta8_an |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['case']} | {r['omega_nodes']} | {r['rmse_ed_vs_disc_p00']:.6f} | {r['rmse_ed_vs_disc_coh']:.6e} | "
            f"{r['p00_at_beta2_ed']:.6f} | {r['p00_at_beta2_an']:.6f} | {r['p00_at_beta8_ed']:.6f} | {r['p00_at_beta8_an']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


def main() -> None:
    df, summary = run_scan()
    write_outputs(df, summary)


if __name__ == "__main__":
    main()

