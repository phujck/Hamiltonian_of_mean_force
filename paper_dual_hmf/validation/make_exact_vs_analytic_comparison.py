"""Generate quantitative exact-vs-analytic agreement diagnostics.

This script compares exact closed-form objects against their asymptotic/dual
analytic estimates on the strong branch x = g / g_star >= 1.

Outputs:
  paper_dual_hmf/manuscript/figures/hmf_exact_vs_analytic_error.pdf
  paper_dual_hmf/manuscript/figures/hmf_exact_vs_analytic_error.png
  paper_dual_hmf/validation/output/exact_vs_analytic_metrics.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_dual_hmf" / "manuscript" / "figures"
OUT_DIR = ROOT / "paper_dual_hmf" / "validation" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gamma_exact_from_x(x: np.ndarray) -> np.ndarray:
    chi = x * x
    return np.tanh(chi) / chi


def gamma_asym_from_x(x: np.ndarray) -> np.ndarray:
    return 1.0 / (x * x)


def entropy_bits_from_chi(chi: np.ndarray) -> np.ndarray:
    r = np.abs(np.tanh(chi))
    lp = 0.5 * (1.0 + r)
    lm = 0.5 * (1.0 - r)
    tiny = np.finfo(float).tiny
    lp = np.clip(lp, tiny, 1.0)
    lm = np.clip(lm, tiny, 1.0)
    return -(lp * np.log2(lp) + lm * np.log2(lm))


def entropy_exact_from_x(x: np.ndarray) -> np.ndarray:
    return entropy_bits_from_chi(x * x)


def entropy_asym_from_x(x: np.ndarray) -> np.ndarray:
    chi = x * x
    return ((1.0 + 2.0 * chi) * np.exp(-2.0 * chi)) / np.log(2.0)


def window_metrics(x: np.ndarray, err: np.ndarray, xmin: float) -> tuple[float, float]:
    mask = x >= xmin
    if not np.any(mask):
        return float("nan"), float("nan")
    vals = err[mask]
    return float(np.max(vals)), float(np.sqrt(np.mean(vals**2)))


def main() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "figure.dpi": 220,
        }
    )

    # Same reduced-coupling window as the strong-branch diagnostic figures.
    x = np.geomspace(1.0, 8.0, 5000)
    windows = [1.0, 1.2, 1.5, 2.0]

    gamma_exact = gamma_exact_from_x(x)
    gamma_asym = gamma_asym_from_x(x)
    gamma_rel = np.abs(gamma_asym - gamma_exact) / np.maximum(np.abs(gamma_exact), 1e-16)

    s_exact = entropy_exact_from_x(x)
    s_asym = entropy_asym_from_x(x)
    s_abs = np.abs(s_asym - s_exact)

    s_rel_mask = s_exact >= 1e-3
    s_rel = np.full_like(s_abs, np.nan)
    s_rel[s_rel_mask] = s_abs[s_rel_mask] / s_exact[s_rel_mask]

    rows: list[dict[str, float]] = []
    for xmin in windows:
        gamma_max, gamma_rms = window_metrics(x, gamma_rel, xmin)
        s_abs_max, s_abs_rms = window_metrics(x, s_abs, xmin)

        rel_window_mask = (x >= xmin) & s_rel_mask
        if np.any(rel_window_mask):
            s_rel_vals = s_rel[rel_window_mask]
            s_rel_max = float(np.max(s_rel_vals))
            s_rel_rms = float(np.sqrt(np.mean(s_rel_vals**2)))
        else:
            s_rel_max = float("nan")
            s_rel_rms = float("nan")

        rows.append(
            {
                "x_min": xmin,
                "gamma_max_rel": gamma_max,
                "gamma_rms_rel": gamma_rms,
                "entropy_max_abs_bits": s_abs_max,
                "entropy_rms_abs_bits": s_abs_rms,
                "entropy_max_rel_for_S_gt_1e-3": s_rel_max,
                "entropy_rms_rel_for_S_gt_1e-3": s_rel_rms,
            }
        )

    csv_path = OUT_DIR / "exact_vs_analytic_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x_min",
                "gamma_max_rel",
                "gamma_rms_rel",
                "entropy_max_abs_bits",
                "entropy_rms_abs_bits",
                "entropy_max_rel_for_S_gt_1e-3",
                "entropy_rms_rel_for_S_gt_1e-3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Plot error profiles.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)

    ax1.plot(x, 100.0 * gamma_rel, color="#0b5394", lw=1.7, label=r"$|\gamma_{\rm asym}-\gamma|/\gamma$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1.0, 8.0)
    ax1.set_ylim(1e-6, 5e1)
    ax1.axvline(1.0, color="0.3", lw=1.0, ls="--")
    ax1.axvline(1.5, color="0.5", lw=0.9, ls=":")
    ax1.text(1.03, 2.5e1, r"self-dual", fontsize=7, color="0.25")
    ax1.text(1.53, 2.5e1, r"$x=1.5$", fontsize=7, color="0.35")
    ax1.set_xlabel(r"$x=g/g^\star$")
    ax1.set_ylabel(r"relative error (\%)")
    ax1.set_title(r"(a) Crossover function $\gamma(\chi)$")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.plot(x, s_abs, color="#990000", lw=1.7, label=r"$|S_Q^{\rm asym}-S_Q|$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1.0, 8.0)
    ax2.set_ylim(1e-16, 2e-1)
    ax2.axvline(1.0, color="0.3", lw=1.0, ls="--")
    ax2.axvline(1.5, color="0.5", lw=0.9, ls=":")
    ax2.text(1.03, 8e-2, r"self-dual", fontsize=7, color="0.25")
    ax2.text(1.53, 8e-2, r"$x=1.5$", fontsize=7, color="0.35")
    ax2.set_xlabel(r"$x=g/g^\star$")
    ax2.set_ylabel(r"absolute error (bits)")
    ax2.set_title(r"(b) Entropy asymptotic")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Exact-vs-analytic agreement on the strong branch", fontsize=10.1)

    out_pdf = FIG_DIR / "hmf_exact_vs_analytic_error.pdf"
    out_png = FIG_DIR / "hmf_exact_vs_analytic_error.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {csv_path}")
    for row in rows:
        print(
            "x_min={x_min:.1f} | gamma max/rms rel=({gmax:.4e}, {grms:.4e}) | "
            "S max/rms abs=({smax:.4e}, {srms:.4e})".format(
                x_min=row["x_min"],
                gmax=row["gamma_max_rel"],
                grms=row["gamma_rms_rel"],
                smax=row["entropy_max_abs_bits"],
                srms=row["entropy_rms_abs_bits"],
            )
        )


if __name__ == "__main__":
    main()
