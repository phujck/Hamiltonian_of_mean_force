"""
Narrow-bandwidth convergence scan around omega_q.

Goal:
- Test whether narrowing the spectral window improves numerical convergence.
- Keep ED and analytic on the SAME window for each point.

Outputs:
- hmf_narrowband_convergence_scan_codex_v36.csv
- hmf_narrowband_convergence_summary_codex_v36.csv
- hmf_narrowband_convergence_cutoff_summary_codex_v36.csv
- hmf_narrowband_convergence_codex_v36.png
- hmf_narrowband_convergence_log_codex_v36.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
    ordered_gaussian_state,
)
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig


@dataclass(frozen=True)
class EDCase:
    n_modes: int
    n_cut: int


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan


def _tail_fraction(omega_max: float, tau_c: float) -> float:
    x = tau_c * omega_max
    return float(np.exp(-x) * (1.0 + x))


def _window_from_delta(omega_q: float, delta: float, floor: float = 0.1) -> tuple[float, float]:
    omega_min = max(floor, omega_q - delta)
    omega_max = omega_q + delta
    return float(omega_min), float(omega_max)


def _detune_and_dw(omega_min: float, omega_max: float, omega_q: float, n_modes: int) -> tuple[float, float]:
    if n_modes <= 1:
        w = 0.5 * (omega_min + omega_max)
        return float(abs(w - omega_q) / omega_q), float(omega_max - omega_min)
    grid = np.linspace(omega_min, omega_max, n_modes, dtype=float)
    dw = float(grid[1] - grid[0])
    detune = float(np.min(np.abs(grid - omega_q)) / omega_q)
    return detune, dw


def _build_lite(
    beta: float,
    theta: float,
    omega_q: float,
    omega_min: float,
    omega_max: float,
) -> LiteConfig:
    return LiteConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=40,
        n_cut=1,
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        q_strength=5.0,
        tau_c=0.5,
    )


def _build_ed(
    beta: float,
    theta: float,
    omega_q: float,
    omega_min: float,
    omega_max: float,
    case: EDCase,
) -> EDConfig:
    return EDConfig(
        beta=float(beta),
        omega_q=float(omega_q),
        theta=float(theta),
        n_modes=int(case.n_modes),
        n_cut=int(case.n_cut),
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        q_strength=5.0,
        tau_c=0.5,
        lambda_min=0.0,
        lambda_max=1.0,
        lambda_points=2,
        output_prefix="hmf_narrowband_convergence_codex_v36",
    )


def run_scan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    theta = float(np.pi / 2.0)
    g = 0.5
    omega_q = 2.0
    tau_c = 0.5
    ren = RenormConfig(scale=1.04, kappa=0.94)

    # Narrow-to-broad windows around omega_q.
    deltas = np.array([0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0], dtype=float)
    betas = np.linspace(0.6, 10.0, 11, dtype=float)
    cases = [EDCase(n_modes=3, n_cut=4), EDCase(n_modes=3, n_cut=5), EDCase(n_modes=3, n_cut=6)]

    total = len(deltas) * len(betas) * len(cases)
    done = 0

    ref_cache: dict[tuple[float, float], tuple[float, float, float, float]] = {}
    rows: list[dict[str, float | int | str]] = []

    for delta in deltas:
        omega_min, omega_max = _window_from_delta(omega_q=omega_q, delta=float(delta), floor=0.1)
        bandwidth = float(omega_max - omega_min)

        for beta in betas:
            key = (float(delta), float(beta))
            if key not in ref_cache:
                lite = _build_lite(
                    beta=float(beta),
                    theta=theta,
                    omega_q=omega_q,
                    omega_min=omega_min,
                    omega_max=omega_max,
                )
                rho_ord = ordered_gaussian_state(lite, g)
                ord_p00, _ord_p11, ord_coh = extract_density(rho_ord)
                best_p00, _best_p11, best_coh, _best_ratio = _compact_components(
                    lite, g, use_running=True, renorm=ren
                )
                ref_cache[key] = (float(ord_p00), float(ord_coh), float(best_p00), float(best_coh))

            ord_p00, ord_coh, best_p00, best_coh = ref_cache[key]

            for case in cases:
                detune_over_omega_q, dw = _detune_and_dw(
                    omega_min=omega_min,
                    omega_max=omega_max,
                    omega_q=omega_q,
                    n_modes=case.n_modes,
                )

                cfg = _build_ed(
                    beta=float(beta),
                    theta=theta,
                    omega_q=omega_q,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    case=case,
                )
                ctx = build_ed_context(cfg)
                rho_ed = exact_reduced_state(ctx, g)
                ed_p00, _ed_p11, ed_coh = extract_density(rho_ed)

                rows.append(
                    {
                        "delta": float(delta),
                        "bandwidth": bandwidth,
                        "omega_min": omega_min,
                        "omega_max": omega_max,
                        "beta": float(beta),
                        "omega_q": omega_q,
                        "theta": theta,
                        "g": g,
                        "n_modes": int(case.n_modes),
                        "n_cut": int(case.n_cut),
                        "dw": float(dw),
                        "detune_over_omega_q": float(detune_over_omega_q),
                        "tail_fraction": _tail_fraction(omega_max, tau_c),
                        "ordered_p00": ord_p00,
                        "ordered_coh": ord_coh,
                        "best_p00": best_p00,
                        "best_coh": best_coh,
                        "ed_p00": float(ed_p00),
                        "ed_coh": float(ed_coh),
                        "d_ed_ord_p00": float(ed_p00 - ord_p00),
                        "d_ed_best_p00": float(ed_p00 - best_p00),
                        "d_ed_ord_coh": float(ed_coh - ord_coh),
                        "d_ed_best_coh": float(ed_coh - best_coh),
                    }
                )
                done += 1
                if done % 40 == 0:
                    print(f"[PROGRESS] {done}/{total}")

    df = pd.DataFrame.from_records(rows).sort_values(
        ["delta", "n_cut", "beta"]
    ).reset_index(drop=True)

    summary_rows: list[dict[str, float | int]] = []
    for (delta, bandwidth, n_cut), grp in df.groupby(["delta", "bandwidth", "n_cut"]):
        summary_rows.append(
            {
                "delta": float(delta),
                "bandwidth": float(bandwidth),
                "n_cut": int(n_cut),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(grp["omega_max"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_ed_vs_best_p00": _rmse(grp["d_ed_best_p00"].to_numpy(float)),
                "rmse_ed_vs_ordered_p00": _rmse(grp["d_ed_ord_p00"].to_numpy(float)),
                "rmse_ed_vs_best_coh": _rmse(grp["d_ed_best_coh"].to_numpy(float)),
                "rmse_ed_vs_ordered_coh": _rmse(grp["d_ed_ord_coh"].to_numpy(float)),
                "p00_at_beta2": float(
                    np.interp(2.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
                "p00_at_beta8": float(
                    np.interp(8.0, grp["beta"].to_numpy(float), grp["ed_p00"].to_numpy(float))
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["delta", "n_cut"]).reset_index(drop=True)

    # Internal cutoff-convergence metric at fixed window: how close each n_cut is to n_cut=6.
    conv_rows: list[dict[str, float | int]] = []
    for (delta, bandwidth), grp in df.groupby(["delta", "bandwidth"]):
        g4 = grp[grp["n_cut"] == 4].sort_values("beta")
        g5 = grp[grp["n_cut"] == 5].sort_values("beta")
        g6 = grp[grp["n_cut"] == 6].sort_values("beta")
        if len(g4) and len(g6):
            rmse_46 = _rmse(g4["ed_p00"].to_numpy(float) - g6["ed_p00"].to_numpy(float))
        else:
            rmse_46 = np.nan
        if len(g5) and len(g6):
            rmse_56 = _rmse(g5["ed_p00"].to_numpy(float) - g6["ed_p00"].to_numpy(float))
        else:
            rmse_56 = np.nan
        conv_rows.append(
            {
                "delta": float(delta),
                "bandwidth": float(bandwidth),
                "omega_min": float(grp["omega_min"].iloc[0]),
                "omega_max": float(grp["omega_max"].iloc[0]),
                "detune_over_omega_q": float(grp["detune_over_omega_q"].iloc[0]),
                "tail_fraction": float(grp["tail_fraction"].iloc[0]),
                "rmse_cut4_vs_cut6_p00": float(rmse_46),
                "rmse_cut5_vs_cut6_p00": float(rmse_56),
            }
        )
    cutoff_summary = pd.DataFrame.from_records(conv_rows).sort_values("delta").reset_index(drop=True)

    return df, summary, cutoff_summary


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, cutoff_summary: pd.DataFrame) -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_narrowband_convergence_scan_codex_v36.csv"
    summary_csv = out_dir / "hmf_narrowband_convergence_summary_codex_v36.csv"
    cutoff_csv = out_dir / "hmf_narrowband_convergence_cutoff_summary_codex_v36.csv"
    fig_png = out_dir / "hmf_narrowband_convergence_codex_v36.png"
    log_md = out_dir / "hmf_narrowband_convergence_log_codex_v36.md"

    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    cutoff_summary.to_csv(cutoff_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut, grp in summary.groupby("n_cut"):
        g = grp.sort_values("bandwidth")
        ax.plot(g["bandwidth"], g["rmse_ed_vs_best_p00"], "o-", linewidth=1.8, label=f"n_cut={int(n_cut)}")
    ax.set_title("RMSE(ED-best) vs bandwidth")
    ax.set_xlabel("bandwidth = omega_max - omega_min")
    ax.set_ylabel("population RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    g = cutoff_summary.sort_values("bandwidth")
    ax.plot(g["bandwidth"], g["rmse_cut4_vs_cut6_p00"], "o-", linewidth=1.8, label="cut4 vs cut6")
    ax.plot(g["bandwidth"], g["rmse_cut5_vs_cut6_p00"], "s--", linewidth=1.8, label="cut5 vs cut6")
    ax.set_title("Internal cutoff convergence")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("RMSE between ED cutoffs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    # Narrowest and broadest windows at n_cut=6
    s6 = summary[summary["n_cut"] == 6].copy()
    if not s6.empty:
        dmin = float(s6["delta"].min())
        dmax = float(s6["delta"].max())
        for d, style, label in [(dmin, "-", "narrow"), (dmax, "--", "broad")]:
            sub = df[(df["delta"] == d) & (df["n_cut"] == 6)].sort_values("beta")
            ax.plot(sub["beta"], sub["ed_p00"], style, linewidth=2.0, label=f"ED {label}")
            ax.plot(sub["beta"], sub["best_p00"], style, linewidth=1.7, label=f"Best {label}")
            ax.plot(sub["beta"], sub["ordered_p00"], style, linewidth=1.4, label=f"Ordered {label}")
    ax.set_title("Beta sweeps: narrow vs broad window (n_cut=6)")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    g2 = summary[summary["n_cut"] == 6].sort_values("bandwidth")
    ax.plot(g2["bandwidth"], g2["detune_over_omega_q"], "o-", linewidth=1.8, label="detune/omega_q")
    ax.plot(g2["bandwidth"], g2["tail_fraction"], "s--", linewidth=1.8, label="tail fraction")
    ax.set_title("Narrow-band descriptors (n_cut=6)")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("dimensionless")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# Narrowband Convergence Around omega_q (Codex v36)")
    lines.append("")
    lines.append("ED and analytic share the same window at each point.")
    lines.append("Fixed: omega_q=2, theta=pi/2, g=0.5, n_modes=3, n_cut in {4,5,6}.")
    lines.append("")
    lines.append("| delta | bandwidth | n_cut | omega_min | omega_max | detune_over_omega_q | tail_fraction | rmse_ed_vs_best_p00 | rmse_ed_vs_ordered_p00 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.sort_values(["n_cut", "delta"]).iterrows():
        lines.append(
            f"| {r['delta']:.3f} | {r['bandwidth']:.3f} | {int(r['n_cut'])} | "
            f"{r['omega_min']:.3f} | {r['omega_max']:.3f} | {r['detune_over_omega_q']:.3f} | "
            f"{r['tail_fraction']:.4f} | {r['rmse_ed_vs_best_p00']:.6f} | {r['rmse_ed_vs_ordered_p00']:.6f} |"
        )
    lines.append("")
    lines.append("| delta | bandwidth | rmse_cut4_vs_cut6_p00 | rmse_cut5_vs_cut6_p00 |")
    lines.append("|---:|---:|---:|---:|")
    for _, r in cutoff_summary.sort_values("delta").iterrows():
        lines.append(
            f"| {r['delta']:.3f} | {r['bandwidth']:.3f} | {r['rmse_cut4_vs_cut6_p00']:.6f} | {r['rmse_cut5_vs_cut6_p00']:.6f} |"
        )

    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", cutoff_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)


def main() -> None:
    df, summary, cutoff_summary = run_scan()
    write_outputs(df, summary, cutoff_summary)


if __name__ == "__main__":
    main()

