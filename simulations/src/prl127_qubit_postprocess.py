"""
Postprocess PRL 127.250601 qubit benchmark scans into manuscript-ready summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("lambda").reset_index(drop=True)


def summarize(df: pd.DataFrame, tag: str) -> dict:
    row0 = df.iloc[0]
    row_max = df.iloc[-1]
    row_min_us = df.loc[df["trace_distance_to_ultrastrong"].idxmin()]
    row_cross = df.iloc[(df["trace_distance_to_tau"] - df["trace_distance_to_ultrastrong"]).abs().idxmin()]
    return {
        "tag": tag,
        "lambda_min": float(row0["lambda"]),
        "lambda_max": float(row_max["lambda"]),
        "d_tau_lambda_min": float(row0["trace_distance_to_tau"]),
        "d_us_lambda_min": float(row0["trace_distance_to_ultrastrong"]),
        "d_tau_lambda_max": float(row_max["trace_distance_to_tau"]),
        "d_us_lambda_max": float(row_max["trace_distance_to_ultrastrong"]),
        "c_hs_lambda_max": float(row_max["coherence_hs_basis"]),
        "lambda_min_d_us": float(row_min_us["lambda"]),
        "d_us_min": float(row_min_us["trace_distance_to_ultrastrong"]),
        "lambda_cross": float(row_cross["lambda"]),
        "d_tau_cross": float(row_cross["trace_distance_to_tau"]),
        "d_us_cross": float(row_cross["trace_distance_to_ultrastrong"]),
    }


def make_beta_comparison_figure(dfs: List[pd.DataFrame], labels: List[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for df, label in zip(dfs, labels):
        axes[0].plot(df["lambda"], df["trace_distance_to_tau"], label=rf"$D(\rho,\tau)$, {label}")
        axes[0].plot(df["lambda"], df["trace_distance_to_ultrastrong"], linestyle="--", label=rf"$D(\rho,\rho_{{US}})$, {label}")
        axes[1].plot(df["lambda"], df["coherence_hs_basis"], label=label)

    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Trace distance")
    axes[0].set_title("Crossover versus coupling")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\ell_1$ coherence in $H_S$ basis")
    axes[1].set_title("Energetic coherence persistence")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "simulations" / "results" / "data"
    fig_dir = project_root / "simulations" / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scans = {
        r"$\beta=0.5$": data_dir / "prl127_qubit_beta0p5_scan.csv",
        r"$\beta=1.0$": data_dir / "prl127_qubit_beta1_prod_scan.csv",
        r"$\beta=2.0$": data_dir / "prl127_qubit_beta2_scan.csv",
    }

    dfs = []
    labels = []
    summaries = []
    for label, path in scans.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing scan CSV: {path}")
        df = load_scan(path)
        dfs.append(df)
        labels.append(label)
        summaries.append(summarize(df, tag=label))

    summary_df = pd.DataFrame(summaries)
    summary_path = data_dir / "prl127_qubit_beta_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    fig_path = fig_dir / "prl127_qubit_beta_comparison.png"
    make_beta_comparison_figure(dfs, labels, fig_path)

    print("Postprocess complete.")
    print(f"Summary CSV: {summary_path}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
