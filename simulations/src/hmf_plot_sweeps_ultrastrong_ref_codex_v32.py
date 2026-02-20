"""
Convergence dashboard with ultrastrong-limit reference overlays.

Reads existing JAX convergence data and adds PRL Eq. (7)/(8) projected-state
ultrastrong references for population and coherence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prl127_qubit_benchmark import SIGMA_X, SIGMA_Z, ultrastrong_projected_state


def _ultrastrong_components(beta: float, theta: float, omega_q: float) -> tuple[float, float]:
    h_s = 0.5 * omega_q * SIGMA_Z
    x_op = np.cos(theta) * SIGMA_Z - np.sin(theta) * SIGMA_X
    rho_us = ultrastrong_projected_state(h_s, x_op, beta)
    p00 = float(np.real(rho_us[0, 0]))
    re01 = float(np.real(rho_us[0, 1]))
    return p00, re01


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "hmf_jax_convergence_results.csv"
    fig_path = out_dir / "hmf_definitive_convergence_suite_ultrastrong_codex_v32.png"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["g"] = df["g"].round(3)
    df["theta"] = df["theta"].round(3)
    df["beta"] = df["beta"].round(2)

    # Keep omega_q explicit for the reference line.
    omega_q = 2.0

    max_g = float(df["g"].max())
    max_th = float(df["theta"].max())
    max_beta = float(df["beta"].max())

    print(
        f"Rendering convergence dashboard with ultrastrong references "
        f"(g={max_g}, theta={max_th}, beta={max_beta})"
    )

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    sectors = df.groupby(["n_modes", "n_cut"])

    # --- ROW 1: TEMPERATURE CONVERGENCE (vs Beta) ---
    for (nm, nc), group in sectors:
        sub = group[(group["g"] == max_g) & (group["theta"] == max_th)].sort_values("beta")
        label = f"m={nm}, c={nc}"
        axes[0, 0].plot(sub["beta"], sub["p00"], "o-", label=label, alpha=0.6, markersize=4)
        axes[0, 1].plot(sub["beta"], sub["re01"], "s-", label=label, alpha=0.6, markersize=4)

    beta_grid = np.sort(
        df[(df["g"] == max_g) & (df["theta"] == max_th)]["beta"].unique().astype(float)
    )
    if beta_grid.size:
        p00_us = np.array([_ultrastrong_components(b, max_th, omega_q)[0] for b in beta_grid], dtype=float)
        re01_us = np.array([_ultrastrong_components(b, max_th, omega_q)[1] for b in beta_grid], dtype=float)
        axes[0, 0].plot(
            beta_grid,
            p00_us,
            "--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )
        axes[0, 1].plot(
            beta_grid,
            re01_us,
            "--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )

    axes[0, 0].set_title(f"Population vs Beta (g={max_g}, theta={max_th})")
    axes[0, 1].set_title(f"Coherence Re(rho01) vs Beta (g={max_g}, theta={max_th})")

    # --- ROW 2: COUPLING CONVERGENCE (vs g) ---
    for (nm, nc), group in sectors:
        sub = group[(group["beta"] == max_beta) & (group["theta"] == max_th)].sort_values("g")
        label = f"m={nm}, c={nc}"
        axes[1, 0].plot(sub["g"], sub["p00"], "o-", label=label, alpha=0.6, markersize=4)
        axes[1, 1].plot(sub["g"], sub["re01"], "s-", label=label, alpha=0.6, markersize=4)

    g_grid = np.sort(df[(df["beta"] == max_beta) & (df["theta"] == max_th)]["g"].unique().astype(float))
    if g_grid.size:
        p00_const, re01_const = _ultrastrong_components(max_beta, max_th, omega_q)
        axes[1, 0].axhline(
            p00_const,
            linestyle="--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )
        axes[1, 1].axhline(
            re01_const,
            linestyle="--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )

    axes[1, 0].set_title(f"Population vs Coupling g (beta={max_beta}, theta={max_th})")
    axes[1, 1].set_title(f"Coherence vs Coupling g (beta={max_beta}, theta={max_th})")

    # --- ROW 3: GEOMETRY CONVERGENCE (vs theta) ---
    for (nm, nc), group in sectors:
        sub = group[(group["beta"] == max_beta) & (group["g"] == max_g)].sort_values("theta")
        label = f"m={nm}, c={nc}"
        axes[2, 0].plot(sub["theta"], sub["p00"], "o-", label=label, alpha=0.6, markersize=4)
        axes[2, 1].plot(sub["theta"], sub["re01"], "s-", label=label, alpha=0.6, markersize=4)

    theta_grid = np.sort(df[(df["beta"] == max_beta) & (df["g"] == max_g)]["theta"].unique().astype(float))
    if theta_grid.size:
        p00_theta = np.array([_ultrastrong_components(max_beta, th, omega_q)[0] for th in theta_grid], dtype=float)
        re01_theta = np.array([_ultrastrong_components(max_beta, th, omega_q)[1] for th in theta_grid], dtype=float)
        axes[2, 0].plot(
            theta_grid,
            p00_theta,
            "--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )
        axes[2, 1].plot(
            theta_grid,
            re01_theta,
            "--",
            color="black",
            linewidth=2.2,
            label="Ultrastrong reference",
        )

    axes[2, 0].set_title(f"Population vs Angle theta (beta={max_beta}, g={max_g})")
    axes[2, 1].set_title(f"Coherence vs Angle theta (beta={max_beta}, g={max_g})")

    for ax in axes.flatten():
        ax.legend(fontsize="x-small", ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Physical Parameter")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {fig_path.name}")


if __name__ == "__main__":
    main()

