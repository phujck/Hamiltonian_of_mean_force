"""
Low-cost consistency check for the exact v5 TeX density formula.

Goal:
1) Verify Eq. (rho_matrix_v5) matches direct matrix product
       rho ~ exp(-beta H_Q) exp(Delta)
   using the same channel amplitudes.
2) Compare both against the ordered numerical reference.

Writes fresh codex_v14 outputs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    extract_density,
    laplace_k0,
    ordered_gaussian_state,
    resonant_r0,
)


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_PLUS = 0.5 * (SIGMA_X + 1.0j * SIGMA_Y)
SIGMA_MINUS = 0.5 * (SIGMA_X - 1.0j * SIGMA_Y)


@dataclass
class ScanPoint:
    sweep: str
    beta: float
    theta: float
    g: float


def _channels_v5(cfg: BenchmarkConfig, g: float) -> tuple[float, float, float]:
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

    sigma_plus = g2 * (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sigma_minus = g2 * (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    delta_z = g2 * (s * s) * 0.5 * (r_p - r_m)
    return float(sigma_plus), float(sigma_minus), float(delta_z)


def _rho_tex_closed(cfg: BenchmarkConfig, sp: float, sm: float, dz: float) -> np.ndarray:
    a = 0.5 * float(cfg.beta) * float(cfg.omega_q)
    chi2 = dz * dz + sp * sm
    chi = float(np.sqrt(max(chi2, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-14 else 1.0

    z_q = 2.0 * (np.cosh(a) - gamma * dz * np.sinh(a))
    rho = np.array(
        [
            [np.exp(-a) * (1.0 + gamma * dz), np.exp(-a) * gamma * sp],
            [np.exp(a) * gamma * sm, np.exp(a) * (1.0 - gamma * dz)],
        ],
        dtype=complex,
    )
    rho /= z_q
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


def _rho_direct_expm(cfg: BenchmarkConfig, sp: float, sm: float, dz: float) -> np.ndarray:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)

    h_q = 0.5 * w * SIGMA_Z
    delta = dz * SIGMA_Z + sp * SIGMA_PLUS + sm * SIGMA_MINUS

    # Shift each exponential by its spectral abscissa to avoid overflow.
    # Overall scalar factors cancel after trace normalization.
    evals_h = np.linalg.eigvals(-beta * h_q)
    evals_d = np.linalg.eigvals(delta)
    sh = float(np.max(np.real(evals_h)))
    sd = float(np.max(np.real(evals_d)))
    e_h = expm(-beta * h_q - sh * np.eye(2, dtype=complex))
    e_d = expm(delta - sd * np.eye(2, dtype=complex))

    rho_bar = e_h @ e_d
    rho = rho_bar / np.trace(rho_bar)
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


def _max_abs_entry(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _build_points() -> list[ScanPoint]:
    pts: list[ScanPoint] = []

    # Coupling sweep (light)
    for g in np.linspace(0.0, 2.0, 7):
        pts.append(ScanPoint(sweep="g", beta=2.0, theta=float(np.pi / 4.0), g=float(g)))

    # Temperature sweep (focus on turning region)
    for beta in np.array([0.2, 0.8, 1.4, 2.0, 3.0, 4.0, 6.0], dtype=float):
        pts.append(ScanPoint(sweep="beta", beta=float(beta), theta=float(np.pi / 2.0), g=0.5))

    return pts


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows: list[dict[str, float | str]] = []

    for p in _build_points():
        cfg = BenchmarkConfig(
            beta=float(p.beta),
            omega_q=2.0,
            theta=float(p.theta),
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )

        sp, sm, dz = _channels_v5(cfg, p.g)
        rho_tex = _rho_tex_closed(cfg, sp, sm, dz)
        rho_direct = _rho_direct_expm(cfg, sp, sm, dz)
        rho_ord = ordered_gaussian_state(cfg, p.g)

        p00_tex, p11_tex, coh_tex = extract_density(rho_tex)
        p00_dir, p11_dir, coh_dir = extract_density(rho_direct)
        p00_ord, p11_ord, coh_ord = extract_density(rho_ord)

        ratio_tex = float(p00_tex / max(p11_tex, 1e-15))
        ratio_dir = float(p00_dir / max(p11_dir, 1e-15))
        ratio_ord = float(p00_ord / max(p11_ord, 1e-15))

        rows.append(
            {
                "sweep": p.sweep,
                "beta": p.beta,
                "theta": p.theta,
                "g": p.g,
                "sigma_plus": sp,
                "sigma_minus": sm,
                "delta_z": dz,
                "tex_vs_direct_max_abs": _max_abs_entry(rho_tex, rho_direct),
                "tex_p00": p00_tex,
                "direct_p00": p00_dir,
                "ordered_p00": p00_ord,
                "tex_coh": coh_tex,
                "direct_coh": coh_dir,
                "ordered_coh": coh_ord,
                "tex_ratio": ratio_tex,
                "direct_ratio": ratio_dir,
                "ordered_ratio": ratio_ord,
                "tex_minus_ordered_p00": p00_tex - p00_ord,
                "direct_minus_ordered_p00": p00_dir - p00_ord,
                "tex_minus_ordered_coh": coh_tex - coh_ord,
                "direct_minus_ordered_coh": coh_dir - coh_ord,
            }
        )

    df = pd.DataFrame.from_records(rows)

    summary_rows: list[dict[str, float | str]] = []
    for sweep, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": str(sweep),
                "max_tex_vs_direct_entry_diff": float(np.max(grp["tex_vs_direct_max_abs"])),
                "mean_tex_vs_direct_entry_diff": float(np.mean(grp["tex_vs_direct_max_abs"])),
                "rmse_tex_minus_ordered_p00": float(np.sqrt(np.mean(np.square(grp["tex_minus_ordered_p00"])))),
                "rmse_direct_minus_ordered_p00": float(np.sqrt(np.mean(np.square(grp["direct_minus_ordered_p00"])))),
                "rmse_tex_minus_ordered_coh": float(np.sqrt(np.mean(np.square(grp["tex_minus_ordered_coh"])))),
                "rmse_direct_minus_ordered_coh": float(np.sqrt(np.mean(np.square(grp["direct_minus_ordered_coh"])))),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values("sweep").reset_index(drop=True)

    scan_csv = out_dir / "hmf_tex_exact_consistency_scan_codex_v14.csv"
    summary_csv = out_dir / "hmf_tex_exact_consistency_summary_codex_v14.csv"
    log_md = out_dir / "hmf_tex_exact_consistency_log_codex_v14.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    lines: list[str] = []
    lines.append("# TeX Exact-Form Consistency (Codex v14)")
    lines.append("")
    lines.append("Checks whether Eq. (rho_matrix_v5) and direct `exp(-beta H_Q) exp(Delta)` agree numerically.")
    lines.append("")
    lines.append("| sweep | max_tex_vs_direct_entry_diff | mean_tex_vs_direct_entry_diff | rmse_tex_minus_ordered_p00 | rmse_direct_minus_ordered_p00 | rmse_tex_minus_ordered_coh | rmse_direct_minus_ordered_coh |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['max_tex_vs_direct_entry_diff']:.3e} | {r['mean_tex_vs_direct_entry_diff']:.3e} | "
            f"{r['rmse_tex_minus_ordered_p00']:.6f} | {r['rmse_direct_minus_ordered_p00']:.6f} | "
            f"{r['rmse_tex_minus_ordered_coh']:.6f} | {r['rmse_direct_minus_ordered_coh']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
