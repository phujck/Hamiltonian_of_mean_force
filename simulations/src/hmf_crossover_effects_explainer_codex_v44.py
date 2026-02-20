"""
Crossover-effect diagnostics from existing scan outputs.

Builds a compact visual explanation of:
1) bandwidth crossover in RMSE(ED-analytic_discrete),
2) onset of cutoff instability,
3) sweep-dependent tradeoff between analytic_discrete and analytic_continuous.

No heavy simulation is run here; this script only reads existing CSV outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _rmse(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(x))))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.name}")
    return pd.read_csv(path)


def _bandwidth_slope(group: pd.DataFrame, ycol: str) -> tuple[np.ndarray, np.ndarray]:
    g = group.sort_values("bandwidth")
    x = g["bandwidth"].to_numpy(dtype=float)
    y = g[ycol].to_numpy(dtype=float)
    if x.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    mids = 0.5 * (x[1:] + x[:-1])
    slope = dy / dx
    return mids, slope


def build_outputs() -> None:
    out_dir = Path(__file__).resolve().parent

    v36_summary = _load_csv(out_dir / "hmf_narrowband_convergence_summary_codex_v36.csv")
    v36_cutoff = _load_csv(out_dir / "hmf_narrowband_convergence_cutoff_summary_codex_v36.csv")
    v36_scan = _load_csv(out_dir / "hmf_narrowband_convergence_scan_codex_v36.csv")
    v43_summary = _load_csv(out_dir / "hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_summary.csv")

    fig_png = out_dir / "hmf_crossover_effects_explainer_codex_v44.png"
    summary_csv = out_dir / "hmf_crossover_effects_explainer_summary_codex_v44.csv"
    log_md = out_dir / "hmf_crossover_effects_explainer_log_codex_v44.md"

    # Per-cut minima in RMSE(ED vs analytic_discrete population)
    min_rows: list[dict[str, float | int]] = []
    for n_cut, grp in v36_summary.groupby("n_cut"):
        g = grp.sort_values("bandwidth")
        i = int(np.argmin(g["rmse_ed_vs_best_p00"].to_numpy(dtype=float)))
        min_rows.append(
            {
                "n_cut": int(n_cut),
                "best_bandwidth": float(g["bandwidth"].iloc[i]),
                "best_delta": float(g["delta"].iloc[i]),
                "best_rmse_ed_vs_disc_p00": float(g["rmse_ed_vs_best_p00"].iloc[i]),
                "rmse_ed_vs_ordered_p00_at_best": float(g["rmse_ed_vs_ordered_p00"].iloc[i]),
            }
        )
    min_df = pd.DataFrame(min_rows).sort_values("n_cut").reset_index(drop=True)

    # Crossover marker from n_cut=6 minimum
    ref6 = min_df[min_df["n_cut"] == 6]
    if ref6.empty:
        crossover_bw = float(np.mean(min_df["best_bandwidth"]))
    else:
        crossover_bw = float(ref6["best_bandwidth"].iloc[0])

    # Two-regime beta traces for n_cut=6 at narrow/crossover/broad windows.
    # Choose deltas closest to bandwidths: narrow~1.6, crossover~3.2, broad~4.9
    target_bws = np.array([1.6, 3.2, 4.9], dtype=float)
    s6 = v36_scan[v36_scan["n_cut"] == 6].copy()
    picked: list[tuple[str, float, float]] = []
    seen = set()
    for label, tbw in zip(["narrow", "crossover", "broad"], target_bws):
        dsub = s6[["delta", "bandwidth"]].drop_duplicates().copy()
        dsub["dist"] = np.abs(dsub["bandwidth"] - tbw)
        dsub = dsub.sort_values("dist")
        row = dsub.iloc[0]
        key = float(row["delta"])
        if key in seen and len(dsub) > 1:
            row = dsub.iloc[1]
            key = float(row["delta"])
        seen.add(key)
        picked.append((label, float(row["delta"]), float(row["bandwidth"])))

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4), constrained_layout=True)

    # A) RMSE vs bandwidth with minima
    ax = axes[0, 0]
    for n_cut, grp in v36_summary.groupby("n_cut"):
        g = grp.sort_values("bandwidth")
        ax.plot(
            g["bandwidth"],
            g["rmse_ed_vs_best_p00"],
            marker="o",
            linewidth=1.8,
            label=f"n_cut={int(n_cut)}",
        )
        i = int(np.argmin(g["rmse_ed_vs_best_p00"].to_numpy(dtype=float)))
        ax.scatter(
            [float(g["bandwidth"].iloc[i])],
            [float(g["rmse_ed_vs_best_p00"].iloc[i])],
            s=50,
            zorder=5,
            color=ax.lines[-1].get_color(),
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axvline(crossover_bw, color="black", linestyle="--", linewidth=1.2, label=f"crossover ~ {crossover_bw:.2f}")
    ax.set_title("Crossover signature: RMSE(ED-disc) vs bandwidth")
    ax.set_xlabel("bandwidth = omega_max - omega_min")
    ax.set_ylabel("population RMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # B) Slope sign flip
    ax = axes[0, 1]
    for n_cut, grp in v36_summary.groupby("n_cut"):
        mids, slope = _bandwidth_slope(grp, "rmse_ed_vs_best_p00")
        ax.plot(mids, slope, marker="o", linewidth=1.6, label=f"n_cut={int(n_cut)}")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.axvline(crossover_bw, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Slope d(RMSE)/d(bandwidth): sign change")
    ax.set_xlabel("bandwidth (midpoint)")
    ax.set_ylabel("finite-difference slope")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # C) Cutoff instability
    ax = axes[0, 2]
    c = v36_cutoff.sort_values("bandwidth")
    ax.plot(c["bandwidth"], c["rmse_cut4_vs_cut6_p00"], "o-", linewidth=1.8, label="cut4 vs cut6")
    ax.plot(c["bandwidth"], c["rmse_cut5_vs_cut6_p00"], "s--", linewidth=1.8, label="cut5 vs cut6")
    ax.axvline(crossover_bw, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Internal cutoff divergence")
    ax.set_xlabel("bandwidth")
    ax.set_ylabel("RMSE between ED cutoffs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # D) Beta traces by regime (n_cut=6)
    ax = axes[1, 0]
    style = {
        "narrow": ("#1f77b4", "-"),
        "crossover": ("#2ca02c", "-."),
        "broad": ("#d62728", "--"),
    }
    for label, d, bw in picked:
        sub = s6[s6["delta"] == d].sort_values("beta")
        color, ls = style[label]
        ax.plot(sub["beta"], sub["ed_p00"], color=color, linestyle=ls, linewidth=2.0, label=f"ED {label} (bw={bw:.1f})")
        ax.plot(
            sub["beta"],
            sub["best_p00"],
            color=color,
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            label=f"disc {label}",
        )
    ax.set_title("Beta profiles across regimes (n_cut=6)")
    ax.set_xlabel("beta")
    ax.set_ylabel("rho_00")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # E) Population RMSE tradeoff in v43 (disc vs cont)
    ax = axes[1, 1]
    s = v43_summary.sort_values("sweep")
    x = np.arange(len(s))
    w = 0.35
    ax.bar(x - w / 2, s["rmse_disc_p00"], width=w, label="disc p00", color="#0B6E4F")
    ax.bar(x + w / 2, s["rmse_cont_p00"], width=w, label="cont p00", color="#1F4E79")
    ax.set_xticks(x)
    ax.set_xticklabels(s["sweep"].tolist())
    ax.set_title("No single branch wins all population sweeps")
    ax.set_ylabel("RMSE(ED-analytic)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # F) Coherence RMSE tradeoff in v43 (disc vs cont)
    ax = axes[1, 2]
    ax.bar(x - w / 2, s["rmse_disc_coh"], width=w, label="disc coh", color="#66c2a5")
    ax.bar(x + w / 2, s["rmse_cont_coh"], width=w, label="cont coh", color="#80b1d3")
    ax.set_xticks(x)
    ax.set_xticklabels(s["sweep"].tolist())
    ax.set_title("Coherence tradeoff by sweep")
    ax.set_ylabel("RMSE(ED-analytic)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    # Save a concise numerical summary
    # Add slope-sign info for n_cut=6
    g6 = v36_summary[v36_summary["n_cut"] == 6].sort_values("bandwidth")
    mids6, slope6 = _bandwidth_slope(g6, "rmse_ed_vs_best_p00")
    sign_flip_bw = float("nan")
    if slope6.size >= 2:
        for i in range(1, slope6.size):
            if slope6[i - 1] < 0.0 and slope6[i] > 0.0:
                sign_flip_bw = float(mids6[i])
                break

    overall = pd.DataFrame(
        [
            {
                "crossover_bandwidth_ref_ncut6": crossover_bw,
                "ncut6_slope_signflip_bandwidth": sign_flip_bw,
                "cutoff_rmse46_at_crossover_bw": float(
                    np.interp(crossover_bw, c["bandwidth"].to_numpy(float), c["rmse_cut4_vs_cut6_p00"].to_numpy(float))
                ),
                "cutoff_rmse46_at_max_bw": float(c.sort_values("bandwidth")["rmse_cut4_vs_cut6_p00"].iloc[-1]),
                "disc_vs_cont_population_rmse_gap_coupling": float(
                    s[s["sweep"] == "coupling"]["rmse_disc_p00"].iloc[0]
                    - s[s["sweep"] == "coupling"]["rmse_cont_p00"].iloc[0]
                ),
                "disc_vs_cont_population_rmse_gap_temperature": float(
                    s[s["sweep"] == "temperature"]["rmse_disc_p00"].iloc[0]
                    - s[s["sweep"] == "temperature"]["rmse_cont_p00"].iloc[0]
                ),
                "disc_vs_cont_population_rmse_gap_angle": float(
                    s[s["sweep"] == "angle"]["rmse_disc_p00"].iloc[0]
                    - s[s["sweep"] == "angle"]["rmse_cont_p00"].iloc[0]
                ),
            }
        ]
    )
    summary_out = min_df.copy()
    summary_out.to_csv(summary_csv, index=False)

    lines: list[str] = []
    lines.append("# Crossover Effects Explainer (Codex v44)")
    lines.append("")
    lines.append("## Bandwidth crossover evidence")
    lines.append("| n_cut | best_bandwidth | best_delta | best_rmse_ed_vs_disc_p00 | rmse_ed_vs_ordered_p00_at_best |")
    lines.append("|---:|---:|---:|---:|---:|")
    for _, r in min_df.iterrows():
        lines.append(
            f"| {int(r['n_cut'])} | {r['best_bandwidth']:.3f} | {r['best_delta']:.3f} | "
            f"{r['best_rmse_ed_vs_disc_p00']:.6f} | {r['rmse_ed_vs_ordered_p00_at_best']:.6f} |"
        )
    lines.append("")
    lines.append("## Interpretation metrics")
    for col, val in overall.iloc[0].items():
        lines.append(f"- {col}: {val:.6f}")
    lines.append("")
    lines.append("## Regime picks used in beta panel")
    for label, d, bw in picked:
        lines.append(f"- {label}: delta={d:.3f}, bandwidth={bw:.3f}")
    lines.append("")
    lines.append("Generated figure:")
    lines.append(f"- {fig_png.name}")
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", fig_png.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(min_df.to_string(index=False))
    print("")
    print(overall.to_string(index=False))


def main() -> None:
    build_outputs()


if __name__ == "__main__":
    main()
