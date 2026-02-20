"""
Ordered-vs-Exact density agreement and direct-Delta matrix-exponential diagnostics.

Fresh-file diagnostic script (Codex v3):
- Focuses on density components: rho_00, rho_11, |rho_01|, and ratio rho_00/rho_11.
- Compares ordered Gaussian state and exact finite-bath reduced state.
- Tests direct construction rho ~ exp(-beta H_Q) exp(Delta) using Delta as 2x2 matrix,
  including diagonalized-Delta evaluation and product-of-traces normalization checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm, eigh

from prl127_qubit_benchmark import BenchmarkConfig
from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state,
)
from hmf_v5_qubit_core import (
    build_ed_context,
    exact_reduced_state,
    compute_v5_base_channels,
    coupling_channels,
    v5_theory_state,
)


SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


@dataclass
class SweepDef:
    name: str
    param_name: str
    param_values: np.ndarray
    beta_fixed: float | None
    theta_fixed: float | None
    g_fixed: float | None
    xlabel: str
    caption: str


def _project_density(rho: np.ndarray) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    vals, vecs = eigh(herm)
    vals = np.clip(np.real(vals), 1e-15, None)
    out = (vecs * vals) @ vecs.conj().T
    out /= np.trace(out)
    return 0.5 * (out + out.conj().T)


def _normalize_by_trace(mat: np.ndarray) -> np.ndarray:
    z = np.trace(mat)
    if (not np.isfinite(np.real(z))) or (not np.isfinite(np.imag(z))) or abs(z) < 1e-15:
        return 0.5 * IDENTITY_2.copy()
    return mat / z


def _density_obs(rho: np.ndarray) -> tuple[float, float, float, float]:
    rho = _project_density(rho)
    p00 = float(np.real(rho[0, 0]))
    p11 = float(np.real(rho[1, 1]))
    coh = float(abs(rho[0, 1]))
    ratio = float(p00 / max(p11, 1e-15))
    return p00, p11, coh, ratio


def _trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    d = 0.5 * ((rho_a - rho_b) + (rho_a - rho_b).conj().T)
    evals = np.linalg.eigvalsh(d)
    return 0.5 * float(np.sum(np.abs(evals)))


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _delta_matrix(config: BenchmarkConfig, g: float) -> tuple[np.ndarray, float, float, float]:
    base = compute_v5_base_channels(config, n_kernel_grid=1201)
    ch = coupling_channels(base, g)
    sigma_plus = float(ch.sigma_plus)
    sigma_minus = float(ch.sigma_minus)
    delta_z = float(ch.delta_z)
    delta = np.array(
        [
            [delta_z, sigma_plus],
            [sigma_minus, -delta_z],
        ],
        dtype=complex,
    )
    return delta, sigma_plus, sigma_minus, delta_z


def _direct_delta_states(config: BenchmarkConfig, g: float) -> dict[str, np.ndarray | float]:
    """
    Build compact states directly from matrix products.
    """
    beta = float(config.beta)
    omega_q = float(config.omega_q)
    a = 0.5 * beta * omega_q

    a_mat = np.diag([np.exp(-a), np.exp(a)]).astype(complex)  # exp(-beta H_Q)
    delta, sp, sm, dz = _delta_matrix(config, g)

    # Shift Delta to avoid overflow in expm; global scalar cancels in rho normalization.
    lam_shift = float(np.max(np.real(np.linalg.eigvals(delta))))
    delta_shifted = delta - lam_shift * IDENTITY_2
    b_expm = expm(delta_shifted)
    # Diagonalized Delta exponent (rotation in Delta eigenbasis)
    evals, evecs = np.linalg.eig(delta_shifted)
    b_diag = evecs @ np.diag(np.exp(evals)) @ np.linalg.inv(evecs)

    # "True" normalization for AB product.
    rho_ab = _project_density(_normalize_by_trace(a_mat @ b_expm))
    rho_ab_diag = _project_density(_normalize_by_trace(a_mat @ b_diag))

    # Product-of-traces assumption variant (without final renormalization).
    z_prod = np.trace(a_mat) * np.trace(b_expm)
    rho_prod_assumed_raw = (a_mat @ b_expm) / z_prod if abs(z_prod) > 1e-15 else 0.5 * IDENTITY_2.copy()
    rho_prod_assumed = _project_density(rho_prod_assumed_raw)

    # Factorization ratio; should be 1 if Tr(AB)=Tr(A)Tr(B), generally false.
    denom = np.trace(a_mat) * np.trace(b_expm)
    if abs(denom) > 1e-15:
        trace_factor_ratio = np.trace(a_mat @ b_expm) / denom
    else:
        trace_factor_ratio = np.nan + 1.0j * np.nan
    trace_factor_ratio_minus1_abs = float(abs(trace_factor_ratio - 1.0)) if np.isfinite(np.real(trace_factor_ratio)) else np.nan
    b_diag_diff = float(np.max(np.abs(b_expm - b_diag)))

    return {
        "rho_ab": rho_ab,
        "rho_ab_diag": rho_ab_diag,
        "rho_prod_assumed": rho_prod_assumed,
        "trace_factor_ratio_minus1_abs": trace_factor_ratio_minus1_abs,
        "b_diag_diff_maxabs": b_diag_diff,
        "sigma_plus": sp,
        "sigma_minus": sm,
        "delta_z": dz,
    }


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    sweeps = [
        SweepDef(
            name="coupling",
            param_name="g",
            param_values=np.linspace(0.0, 2.0, 15),
            beta_fixed=2.0,
            theta_fixed=np.pi / 4,
            g_fixed=None,
            xlabel=r"$g$",
            caption=r"$\beta=2,\ \theta=\pi/4$",
        ),
        SweepDef(
            name="angle",
            param_name="theta",
            param_values=np.linspace(0.0, np.pi / 2, 15),
            beta_fixed=2.0,
            theta_fixed=None,
            g_fixed=0.5,
            xlabel=r"$\theta/\pi$",
            caption=r"$\beta=2,\ g=0.5$",
        ),
        SweepDef(
            name="temperature",
            param_name="beta",
            param_values=np.linspace(0.2, 6.0, 15),
            beta_fixed=None,
            theta_fixed=np.pi / 2,
            g_fixed=0.5,
            xlabel=r"$\beta$",
            caption=r"$\theta=\pi/2,\ g=0.5$",
        ),
    ]

    rows: list[dict[str, float | str]] = []

    for sweep in sweeps:
        for param in sweep.param_values:
            beta = float(param) if sweep.param_name == "beta" else float(sweep.beta_fixed)
            theta = float(param) if sweep.param_name == "theta" else float(sweep.theta_fixed)
            g = float(param) if sweep.param_name == "g" else float(sweep.g_fixed)

            cfg = BenchmarkConfig(
                beta=beta,
                omega_q=2.0,
                theta=theta,
                lambda_min=0.0,
                lambda_max=max(2.0, g),
                lambda_points=3,
                n_modes=3,
                n_cut=4,
                omega_min=0.1,
                omega_max=8.0,
                q_strength=5.0,
                tau_c=0.5,
                output_prefix="hmf_ord_exact_density_codex_v3",
            )

            # Exact
            ed_ctx = build_ed_context(cfg)
            rho_exact = _project_density(exact_reduced_state(ed_ctx, g))
            ex_p00, ex_p11, ex_coh, ex_ratio = _density_obs(rho_exact)

            # Ordered
            ord_ctx = _build_ordered_quadrature_context(
                cfg,
                n_time_slices=70,
                kl_rank=4,
                gh_order=4,
                max_nodes=300000,
            )
            rho_ord = _project_density(finite_hmf_ordered_gaussian_state(g, ord_ctx))
            or_p00, or_p11, or_coh, or_ratio = _density_obs(rho_ord)

            # Direct Delta matrix exponent variants
            d = _direct_delta_states(cfg, g)
            rho_ab = d["rho_ab"]
            rho_ab_diag = d["rho_ab_diag"]
            rho_prod_assumed = d["rho_prod_assumed"]
            ab_p00, ab_p11, ab_coh, ab_ratio = _density_obs(rho_ab)
            ad_p00, ad_p11, ad_coh, ad_ratio = _density_obs(rho_ab_diag)
            ap_p00, ap_p11, ap_coh, ap_ratio = _density_obs(rho_prod_assumed)

            # Closed compact constructor (for cross-check)
            base = compute_v5_base_channels(cfg, n_kernel_grid=1201)
            ch = coupling_channels(base, g)
            rho_closed = _project_density(v5_theory_state(cfg, ch))
            cl_p00, cl_p11, cl_coh, cl_ratio = _density_obs(rho_closed)

            rows.append(
                {
                    "sweep": sweep.name,
                    "param_name": sweep.param_name,
                    "param": float(param),
                    "beta": beta,
                    "theta": theta,
                    "g": g,
                    "exact_p00": ex_p00,
                    "exact_p11": ex_p11,
                    "exact_coh": ex_coh,
                    "exact_ratio": ex_ratio,
                    "ordered_p00": or_p00,
                    "ordered_p11": or_p11,
                    "ordered_coh": or_coh,
                    "ordered_ratio": or_ratio,
                    "ordered_minus_exact_p00": or_p00 - ex_p00,
                    "ordered_minus_exact_p11": or_p11 - ex_p11,
                    "ordered_minus_exact_coh": or_coh - ex_coh,
                    "ordered_ratio_over_exact": or_ratio / max(ex_ratio, 1e-15),
                    "ordered_trace_distance_exact": _trace_distance(rho_ord, rho_exact),
                    "compact_ab_p00": ab_p00,
                    "compact_ab_p11": ab_p11,
                    "compact_ab_coh": ab_coh,
                    "compact_ab_ratio": ab_ratio,
                    "compact_ab_minus_exact_p00": ab_p00 - ex_p00,
                    "compact_ab_minus_exact_p11": ab_p11 - ex_p11,
                    "compact_ab_minus_exact_coh": ab_coh - ex_coh,
                    "compact_ab_ratio_over_exact": ab_ratio / max(ex_ratio, 1e-15),
                    "compact_ab_trace_distance_exact": _trace_distance(rho_ab, rho_exact),
                    "compact_abdiag_p00": ad_p00,
                    "compact_abdiag_p11": ad_p11,
                    "compact_abdiag_coh": ad_coh,
                    "compact_abdiag_ratio": ad_ratio,
                    "compact_prodassume_p00": ap_p00,
                    "compact_prodassume_p11": ap_p11,
                    "compact_prodassume_coh": ap_coh,
                    "compact_prodassume_ratio": ap_ratio,
                    "compact_closed_p00": cl_p00,
                    "compact_closed_p11": cl_p11,
                    "compact_closed_coh": cl_coh,
                    "compact_closed_ratio": cl_ratio,
                    "compact_closed_trace_distance_exact": _trace_distance(rho_closed, rho_exact),
                    "compact_ab_vs_closed_trace_distance": _trace_distance(rho_ab, rho_closed),
                    "compact_abdiag_vs_ab_trace_distance": _trace_distance(rho_ab_diag, rho_ab),
                    "trace_factor_ratio_minus1_abs": float(d["trace_factor_ratio_minus1_abs"]),
                    "delta_diag_expm_maxabs": float(d["b_diag_diff_maxabs"]),
                    "sigma_plus": float(d["sigma_plus"]),
                    "sigma_minus": float(d["sigma_minus"]),
                    "delta_z": float(d["delta_z"]),
                }
            )

    df = pd.DataFrame.from_records(rows)

    summary_rows = []
    for sweep_name, grp in df.groupby("sweep"):
        summary_rows.append(
            {
                "sweep": sweep_name,
                "model": "ordered_vs_exact",
                "rmse_p00": _rmse(grp["ordered_minus_exact_p00"]),
                "rmse_p11": _rmse(grp["ordered_minus_exact_p11"]),
                "rmse_coh": _rmse(grp["ordered_minus_exact_coh"]),
                "rmse_trace_distance": _rmse(grp["ordered_trace_distance_exact"]),
                "ratio_over_exact_median": float(np.median(grp["ordered_ratio_over_exact"])),
                "ratio_over_exact_min": float(np.min(grp["ordered_ratio_over_exact"])),
                "ratio_over_exact_max": float(np.max(grp["ordered_ratio_over_exact"])),
            }
        )
        summary_rows.append(
            {
                "sweep": sweep_name,
                "model": "compact_ab_vs_exact",
                "rmse_p00": _rmse(grp["compact_ab_minus_exact_p00"]),
                "rmse_p11": _rmse(grp["compact_ab_minus_exact_p11"]),
                "rmse_coh": _rmse(grp["compact_ab_minus_exact_coh"]),
                "rmse_trace_distance": _rmse(grp["compact_ab_trace_distance_exact"]),
                "ratio_over_exact_median": float(np.median(grp["compact_ab_ratio_over_exact"])),
                "ratio_over_exact_min": float(np.min(grp["compact_ab_ratio_over_exact"])),
                "ratio_over_exact_max": float(np.max(grp["compact_ab_ratio_over_exact"])),
            }
        )

    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "model"]).reset_index(drop=True)
    return df, summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    data_csv = out_dir / "hmf_ordered_exact_density_scan_codex_v3.csv"
    summary_csv = out_dir / "hmf_ordered_exact_density_summary_codex_v3.csv"
    fig_png = out_dir / "hmf_ordered_exact_density_ratios_codex_v3.png"
    log_md = out_dir / "hmf_ordered_exact_density_log_codex_v3.md"

    df.to_csv(data_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # 3-panel ratio trend plot (ordered/exact and compact/exact)
    sweeps = [
        ("coupling", r"$g$"),
        ("angle", r"$\theta/\pi$"),
        ("temperature", r"$\beta$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for i, (sweep_name, xlabel) in enumerate(sweeps):
        g = df[df["sweep"] == sweep_name].sort_values("param")
        x = g["param"].to_numpy()
        if sweep_name == "angle":
            x = x / np.pi
        axes[i].plot(x, g["ordered_ratio_over_exact"], color="black", linewidth=2.0, label="Ordered / Exact")
        axes[i].plot(x, g["compact_ab_ratio_over_exact"], color="#C84B31", linewidth=1.8, label="Compact(AB) / Exact")
        axes[i].axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
        axes[i].set_title(sweep_name.capitalize())
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(r"Population Ratio Relative Error")
        axes[i].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(r"Trend of $(\rho_{00}/\rho_{11})_{\mathrm{model}} / (\rho_{00}/\rho_{11})_{\mathrm{exact}}$", fontsize=12)
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    trace_factor_gap_max = float(np.nanmax(df["trace_factor_ratio_minus1_abs"]))
    diag_diff_max = float(np.max(df["delta_diag_expm_maxabs"]))
    ab_closed_rmse = _rmse(df["compact_ab_vs_closed_trace_distance"])

    lines = []
    lines.append("# Ordered-Exact Density Agreement (Codex v3)")
    lines.append("")
    lines.append("## Core Question")
    lines.append("How closely do ordered and exact models agree, and does direct Delta matrix multiplication help?")
    lines.append("")
    lines.append("## Matrix-Exponent Diagnostics")
    lines.append(
        f"- Max |Tr(A B)/(Tr(A)Tr(B)) - 1| across scan: `{trace_factor_gap_max:.6e}` "
        "(product-of-traces assumption is generally not valid for 2x2 AB)."
    )
    lines.append(f"- Max |expm(Delta) - V exp(D) V^-1| elementwise: `{diag_diff_max:.6e}`.")
    lines.append(f"- RMSE trace-distance between compact AB and closed compact constructor: `{ab_closed_rmse:.6e}`.")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| sweep | model | rmse_p00 | rmse_p11 | rmse_coh | rmse_trace_distance | ratio_over_exact_median | ratio_over_exact_min | ratio_over_exact_max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['sweep']} | {row['model']} | {row['rmse_p00']:.6f} | {row['rmse_p11']:.6f} | "
            f"{row['rmse_coh']:.6f} | {row['rmse_trace_distance']:.6f} | {row['ratio_over_exact_median']:.6f} | "
            f"{row['ratio_over_exact_min']:.6f} | {row['ratio_over_exact_max']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", data_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    df, summary = run()
    write_outputs(df, summary, out_dir)


if __name__ == "__main__":
    main()
