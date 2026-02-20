"""
Plot v5 regime figures from saved CSV outputs only.

Inputs:
- simulations/results/data/<prefix>_scan.csv
- simulations/results/data/<prefix>_summary.csv

Outputs:
- simulations/results/figures/<prefix>_panels.png
- simulations/results/figures/<prefix>_panel_a_geometry.png
- simulations/results/figures/<prefix>_panel_b_coherence.png
- simulations/results/figures/<prefix>_panel_c_susceptibility.png
- simulations/results/figures/<prefix>_panel_d_trace_distance.png
- Copies of each figure into manuscript/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _color_map(beta_omega_values: Iterable[float]) -> Dict[float, Tuple[float, float, float, float]]:
    ordered = list(sorted(beta_omega_values))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(ordered)))
    return {b: colors[i] for i, b in enumerate(ordered)}


def _theory_tag(scan_df: pd.DataFrame) -> str:
    if "theory_model" not in scan_df.columns:
        return "theory"
    vals = [str(v) for v in scan_df["theory_model"].dropna().unique().tolist()]
    if not vals:
        return "theory"
    return vals[0]


def _downsample_indices(n: int, marker_every: int) -> np.ndarray:
    if marker_every <= 1:
        return np.arange(n, dtype=int)
    idx = np.arange(0, n, marker_every, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _add_beta_legend(ax: plt.Axes, beta_values: Iterable[float], colors: Dict[float, Tuple[float, float, float, float]]) -> None:
    handles = [
        Line2D([0], [0], color=colors[b], lw=2.5, label=rf"$\beta\omega_q={b:g}$")
        for b in sorted(beta_values)
    ]
    legend = ax.legend(handles=handles, title=r"Temperature", fontsize=8, title_fontsize=9, loc="upper left")
    ax.add_artist(legend)


def _plot_panel_a(ax: plt.Axes, scan_df: pd.DataFrame, colors: Dict[float, Tuple[float, float, float, float]], marker_every: int) -> None:
    ax_r = ax.twinx()
    beta_values = sorted(scan_df["beta_omega_q"].unique())

    for beta_omega_q in beta_values:
        color = colors[beta_omega_q]
        sub = scan_df[scan_df["beta_omega_q"] == beta_omega_q].sort_values("g")
        g = sub["g"].to_numpy()
        idx = _downsample_indices(len(sub), marker_every)

        ax.plot(g, sub["phi_th"], color=color, lw=2.0)
        ax.plot(g[idx], sub["phi_ed"].to_numpy()[idx], linestyle="None", marker="o", markersize=3.4, color=color)

        ax_r.plot(g, sub["r_th"], color=color, lw=1.8, linestyle="--")
        ax_r.plot(g[idx], sub["r_ed"].to_numpy()[idx], linestyle="None", marker="x", markersize=3.2, color=color)

    _add_beta_legend(ax, beta_values, colors)
    style_handles = [
        Line2D([0], [0], color="k", lw=2.0, label=r"$\phi_{\rm th}$"),
        Line2D([0], [0], color="k", marker="o", linestyle="None", markersize=4, label=r"$\phi_{\rm ED}$"),
        Line2D([0], [0], color="k", lw=1.8, linestyle="--", label=r"$r_{\rm th}$"),
        Line2D([0], [0], color="k", marker="x", linestyle="None", markersize=4, label=r"$r_{\rm ED}$"),
    ]
    ax_r.legend(handles=style_handles, fontsize=8, loc="lower right")

    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\phi(g)$")
    ax_r.set_ylabel(r"$r(g)$")
    ax.set_ylim(0.0, np.pi)
    ax_r.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title("(a) Bloch Geometry")


def _plot_panel_b(ax: plt.Axes, scan_df: pd.DataFrame, colors: Dict[float, Tuple[float, float, float, float]], marker_every: int) -> None:
    theory_tag = _theory_tag(scan_df)
    beta_values = sorted(scan_df["beta_omega_q"].unique())
    for beta_omega_q in beta_values:
        color = colors[beta_omega_q]
        sub = scan_df[scan_df["beta_omega_q"] == beta_omega_q].sort_values("g")
        g = sub["g"].to_numpy()
        idx = _downsample_indices(len(sub), marker_every)
        ax.plot(g, sub["coherence_th"], color=color, lw=2.0, label=rf"{theory_tag} $\beta\omega_q={beta_omega_q:g}$")
        ax.plot(
            g[idx],
            sub["coherence_ed"].to_numpy()[idx],
            linestyle="None",
            marker="o",
            markersize=3.4,
            color=color,
            label=rf"ED $\beta\omega_q={beta_omega_q:g}$",
        )
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$C(g)=2|\rho_{01}|$")
    ax.set_title("(b) Bare-Basis Coherence")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="best")


def _plot_panel_c(
    ax: plt.Axes,
    scan_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    colors: Dict[float, Tuple[float, float, float, float]],
    marker_every: int,
) -> None:
    theory_tag = _theory_tag(scan_df)
    beta_values = sorted(scan_df["beta_omega_q"].unique())
    for beta_omega_q in beta_values:
        color = colors[beta_omega_q]
        sub = scan_df[scan_df["beta_omega_q"] == beta_omega_q].sort_values("g")
        g = sub["g"].to_numpy()
        idx = _downsample_indices(len(sub), marker_every)
        ax.plot(g, sub["xi_phi_th"], color=color, lw=2.0, label=rf"{theory_tag} $\beta\omega_q={beta_omega_q:g}$")
        ax.plot(
            g[idx],
            sub["xi_phi_ed"].to_numpy()[idx],
            linestyle="None",
            marker="o",
            markersize=3.4,
            color=color,
            label=rf"ED $\beta\omega_q={beta_omega_q:g}$",
        )

        summary_row = summary_df.loc[summary_df["beta_omega_q"] == beta_omega_q]
        if len(summary_row) == 1:
            g_star = float(summary_row.iloc[0]["g_star"])
            if np.isfinite(g_star):
                ax.axvline(g_star, color=color, linestyle=":", lw=1.4, alpha=0.95)

    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$\Xi_\phi(g)=\partial_g\phi$")
    ax.set_title("(c) Crossover Susceptibility")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=1, loc="best")


def _plot_panel_d(ax: plt.Axes, scan_df: pd.DataFrame, colors: Dict[float, Tuple[float, float, float, float]], marker_every: int) -> None:
    beta_values = sorted(scan_df["beta_omega_q"].unique())
    for beta_omega_q in beta_values:
        color = colors[beta_omega_q]
        sub = scan_df[scan_df["beta_omega_q"] == beta_omega_q].sort_values("g")
        g = sub["g"].to_numpy()
        idx = _downsample_indices(len(sub), marker_every)
        ax.plot(g, sub["trace_distance"], color=color, lw=2.0, label=rf"$\beta\omega_q={beta_omega_q:g}$")
        ax.plot(g[idx], sub["trace_distance"].to_numpy()[idx], linestyle="None", marker="o", markersize=2.8, color=color)
    ax.set_xlabel(r"$g$")
    ax.set_ylabel(r"$D(g)=\frac{1}{2}\|\rho_Q^{\rm ED}-\rho_Q^{\rm th}\|_1$")
    ax.set_title("(d) State-Level Agreement")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=1, loc="best")


def _save_and_copy(fig: plt.Figure, out_path: Path, ms_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    ms_path = ms_dir / out_path.name
    ms_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ms_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot v5 regime figures from saved CSV data.")
    parser.add_argument("--output-prefix", type=str, default="hmf_v5_regime")
    parser.add_argument("--marker-every", type=int, default=4)
    return parser


def run_from_args(args: argparse.Namespace) -> Dict[str, Path]:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "simulations" / "results" / "data"
    fig_dir = project_root / "simulations" / "results" / "figures"
    ms_dir = project_root / "manuscript" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ms_dir.mkdir(parents=True, exist_ok=True)

    scan_path = data_dir / f"{args.output_prefix}_scan.csv"
    summary_path = data_dir / f"{args.output_prefix}_summary.csv"
    if not scan_path.exists():
        raise FileNotFoundError(f"Missing scan CSV: {scan_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")

    scan_df = pd.read_csv(scan_path).sort_values(["beta_omega_q", "g"]).reset_index(drop=True)
    summary_df = pd.read_csv(summary_path).sort_values("beta_omega_q").reset_index(drop=True)
    colors = _color_map(scan_df["beta_omega_q"].unique())

    paths: Dict[str, Path] = {}

    # Composite 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    _plot_panel_a(axes[0, 0], scan_df, colors, args.marker_every)
    _plot_panel_b(axes[0, 1], scan_df, colors, args.marker_every)
    _plot_panel_c(axes[1, 0], scan_df, summary_df, colors, args.marker_every)
    _plot_panel_d(axes[1, 1], scan_df, colors, args.marker_every)
    composite_path = fig_dir / f"{args.output_prefix}_panels.png"
    _save_and_copy(fig, composite_path, ms_dir)
    paths["composite"] = composite_path

    # Panel A.
    fig_a, ax_a = plt.subplots(figsize=(6.2, 4.4))
    _plot_panel_a(ax_a, scan_df, colors, args.marker_every)
    path_a = fig_dir / f"{args.output_prefix}_panel_a_geometry.png"
    _save_and_copy(fig_a, path_a, ms_dir)
    paths["panel_a"] = path_a

    # Panel B.
    fig_b, ax_b = plt.subplots(figsize=(6.2, 4.4))
    _plot_panel_b(ax_b, scan_df, colors, args.marker_every)
    path_b = fig_dir / f"{args.output_prefix}_panel_b_coherence.png"
    _save_and_copy(fig_b, path_b, ms_dir)
    paths["panel_b"] = path_b

    # Panel C.
    fig_c, ax_c = plt.subplots(figsize=(6.2, 4.4))
    _plot_panel_c(ax_c, scan_df, summary_df, colors, args.marker_every)
    path_c = fig_dir / f"{args.output_prefix}_panel_c_susceptibility.png"
    _save_and_copy(fig_c, path_c, ms_dir)
    paths["panel_c"] = path_c

    # Panel D.
    fig_d, ax_d = plt.subplots(figsize=(6.2, 4.4))
    _plot_panel_d(ax_d, scan_df, colors, args.marker_every)
    path_d = fig_dir / f"{args.output_prefix}_panel_d_trace_distance.png"
    _save_and_copy(fig_d, path_d, ms_dir)
    paths["panel_d"] = path_d

    return paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    paths = run_from_args(args)
    print("v5 regime plotting complete.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
