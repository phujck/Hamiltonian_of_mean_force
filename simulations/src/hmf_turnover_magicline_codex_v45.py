"""
Formulate and visualize a crossover "magic line" from existing CSV outputs.

No new simulation data is generated.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_STEM = "hmf_turnover_magicline_codex_v45"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path.name}")
    return pd.read_csv(path)


def _slope_zero_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    slopes = np.diff(y) / np.diff(x)
    mids = 0.5 * (x[1:] + x[:-1])
    for i in range(1, slopes.size):
        s0, s1 = float(slopes[i - 1]), float(slopes[i])
        if s0 <= 0.0 and s1 >= 0.0:
            m0, m1 = float(mids[i - 1]), float(mids[i])
            if abs(s1 - s0) < 1e-15:
                return m1
            return m0 + (0.0 - s0) * (m1 - m0) / (s1 - s0)
    return float("nan")


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    summary = _load_csv(out_dir / "hmf_narrowband_convergence_summary_codex_v36.csv")
    cutoff = _load_csv(out_dir / "hmf_narrowband_convergence_cutoff_summary_codex_v36.csv")
    scan = _load_csv(out_dir / "hmf_narrowband_convergence_scan_codex_v36.csv")

    # Inferred from existing scan setup
    omega_q = float(scan["omega_q"].iloc[0])
    omega_floor = float(scan["omega_min"].min())

    # Derived "magic line" for onset of resonance detuning in the clamped-window regime:
    #   B* = 2 (omega_q - omega_floor)
    # where B = omega_max - omega_min.
    bw_magic = 2.0 * (omega_q - omega_floor)

    # Empirical references from n_cut=6 curve.
    g6 = summary[summary["n_cut"] == 6].sort_values("bandwidth")
    x6 = g6["bandwidth"].to_numpy(dtype=float)
    y6 = g6["rmse_ed_vs_best_p00"].to_numpy(dtype=float)
    i_min6 = int(np.argmin(y6))
    bw_min6 = float(x6[i_min6])
    rmse_min6 = float(y6[i_min6])
    bw_slope0 = _slope_zero_bandwidth(x6, y6)

    # Predicted detune from piecewise formula at n_modes=3 with lower-bound clamping.
    bw_grid = np.linspace(float(summary["bandwidth"].min()), float(summary["bandwidth"].max()), 300)
    detune_pred = np.maximum(0.0, omega_floor + 0.5 * bw_grid - omega_q) / omega_q

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for n_cut, grp in summary.groupby("n_cut"):
        g = grp.sort_values("bandwidth")
        ax.plot(g["bandwidth"], g["rmse_ed_vs_best_p00"], "o-", linewidth=1.8, label=f"n_cut={int(n_cut)}")
    ax.axvline(bw_magic, color="black", linestyle="--", linewidth=1.3, label=rf"magic line $B^*={bw_magic:.2f}$")
    ax.axvline(bw_min6, color="#0B6E4F", linestyle=":", linewidth=1.5, label=rf"n_cut=6 min $B_{{min}}={bw_min6:.2f}$")
    if np.isfinite(bw_slope0):
        ax.axvline(bw_slope0, color="#1F4E79", linestyle="-.", linewidth=1.3, label=rf"slope-zero $B_0={bw_slope0:.2f}$")
    ax.set_title("Turnover in RMSE(ED - analytic discrete)")
    ax.set_xlabel("bandwidth B = omega_max - omega_min")
    ax.set_ylabel("population RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    det6 = g6["detune_over_omega_q"].to_numpy(dtype=float)
    ax.plot(g6["bandwidth"], det6, "o", color="#0B6E4F", label="data: detune/omega_q")
    ax.plot(bw_grid, detune_pred, "-", color="black", linewidth=1.6, label="predicted piecewise detune")
    ax.axvline(bw_magic, color="black", linestyle="--", linewidth=1.3)
    ax.set_title("Magic-line condition predicts detuning onset")
    ax.set_xlabel("bandwidth B")
    ax.set_ylabel("detune/omega_q")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    c = cutoff.sort_values("bandwidth")
    ax.plot(c["bandwidth"], c["rmse_cut4_vs_cut6_p00"], "o-", linewidth=1.8, label="cut4 vs cut6")
    ax.plot(c["bandwidth"], c["rmse_cut5_vs_cut6_p00"], "s--", linewidth=1.8, label="cut5 vs cut6")
    ax.axvline(bw_magic, color="black", linestyle="--", linewidth=1.3, label=rf"magic line $B^*$")
    ax.set_title("Cutoff instability grows after crossover")
    ax.set_xlabel("bandwidth B")
    ax.set_ylabel("RMSE between ED cutoffs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    s = summary[summary["n_cut"] == 6].sort_values("bandwidth")
    ax.plot(s["bandwidth"], s["tail_fraction"], "s--", linewidth=1.8, color="#d62728", label="tail_fraction")
    ax2 = ax.twinx()
    ax2.plot(s["bandwidth"], s["detune_over_omega_q"], "o-", linewidth=1.8, color="#1f77b4", label="detune/omega_q")
    ax.axvline(bw_magic, color="black", linestyle="--", linewidth=1.3)
    ax.set_title("Descriptor crossover: tail vs detuning")
    ax.set_xlabel("bandwidth B")
    ax.set_ylabel("tail_fraction", color="#d62728")
    ax2.set_ylabel("detune/omega_q", color="#1f77b4")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

    fig_png = out_dir / f"{OUT_STEM}.png"
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    # Summary table
    out_summary = pd.DataFrame(
        [
            {
                "omega_q": omega_q,
                "omega_floor": omega_floor,
                "magic_bandwidth_B_star": bw_magic,
                "empirical_B_min_ncut6": bw_min6,
                "empirical_rmse_min_ncut6": rmse_min6,
                "empirical_B_slope_zero_ncut6": bw_slope0,
                "gap_B_star_minus_B_min": bw_magic - bw_min6,
                "gap_B_star_minus_B_slope0": bw_magic - bw_slope0 if np.isfinite(bw_slope0) else np.nan,
            }
        ]
    )
    out_summary_csv = out_dir / f"{OUT_STEM}_summary.csv"
    out_summary.to_csv(out_summary_csv, index=False)

    # Markdown log with explicit formula.
    log_md = out_dir / f"{OUT_STEM}_log.md"
    lines: list[str] = []
    lines.append("# Turnover Magic-Line Formulation (Codex v45)")
    lines.append("")
    lines.append("## Formulated condition")
    lines.append("For the bandwidth scan with lower-bound clamping `omega_min = max(omega_floor, omega_q - delta)`,")
    lines.append("the turnover onset is where the resonance pinning breaks and detuning becomes nonzero:")
    lines.append("")
    lines.append(r"$B^* = 2(\omega_q - \omega_{\mathrm{floor}})$")
    lines.append("")
    lines.append("with `B = omega_max - omega_min`.")
    lines.append("")
    lines.append("Equivalent piecewise detuning predictor (n_modes=3, uniform spacing):")
    lines.append("")
    lines.append(r"$\dfrac{\delta_\mathrm{detune}(B)}{\omega_q} = \dfrac{\max\!\left(0,\ \omega_{\mathrm{floor}} + \dfrac{B}{2} - \omega_q\right)}{\omega_q}$")
    lines.append("")
    lines.append("## Numbers from current CSVs")
    for k, v in out_summary.iloc[0].items():
        lines.append(f"- {k}: {float(v):.6f}")
    lines.append("")
    lines.append("Generated figure:")
    lines.append(f"- {fig_png.name}")
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", fig_png.name)
    print("Wrote:", out_summary_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
