"""
Check whether inferred diagonal factor resembles kernel-derived functions (Codex v11).

Uses:
- hmf_delta_factor_scan_codex_v8.csv (k_needed_for_delta vs g at theta=pi/2),
- direct kernel moments K(0), K(+w), K(-w), R(+w), R(-w)

Builds candidate curves and compares RMSE/correlation against k_needed(g).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    laplace_k0,
    resonant_r0,
)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if np.count_nonzero(m) < 3:
        return np.nan
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def _fit_affine_to_target(y: np.ndarray, u: np.ndarray) -> tuple[float, float, np.ndarray]:
    # y ~ a + b*u
    m = np.isfinite(y) & np.isfinite(u)
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan, np.full_like(y, np.nan)
    X = np.column_stack([np.ones_like(u[m]), u[m]])
    coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = np.full_like(y, np.nan, dtype=float)
    yhat[m] = a + b * u[m]
    return a, b, yhat


def _fit_pade_g2(y: np.ndarray, g: np.ndarray) -> tuple[float, float, np.ndarray]:
    # y ~ a + b/(1 + c g^2), with grid over c and linear solve in (a,b)
    m = np.isfinite(y) & np.isfinite(g)
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan, np.nan, np.full_like(y, np.nan)
    yy = y[m]
    gg = g[m]
    best = None
    for c in np.linspace(0.0, 20.0, 1201):
        u = 1.0 / (1.0 + c * gg * gg)
        X = np.column_stack([np.ones_like(u), u])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        yfit = a + b * u
        err = float(np.mean((yy - yfit) ** 2))
        if best is None or err < best[0]:
            best = (err, a, b, c, yfit)
    assert best is not None
    _, a, b, c, yfit = best
    yhat = np.full_like(y, np.nan, dtype=float)
    yhat[m] = yfit
    return a, b, c, yhat


def _corr(y: np.ndarray, u: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(u)
    if np.count_nonzero(m) < 3:
        return np.nan
    yy = y[m]
    uu = u[m]
    if np.std(yy) <= 1e-15 or np.std(uu) <= 1e-15:
        return np.nan
    return float(np.corrcoef(yy, uu)[0, 1])


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    in_csv = out_dir / "hmf_delta_factor_scan_codex_v8.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}")
    df = pd.read_csv(in_csv).sort_values("g").reset_index(drop=True)

    # Fixed configuration used in v8 inference
    beta = 2.0
    omega_q = 2.0
    theta = float(np.pi / 2.0)
    cfg = BenchmarkConfig(
        beta=beta,
        omega_q=omega_q,
        theta=theta,
        n_modes=40,
        n_cut=1,
        omega_min=0.1,
        omega_max=10.0,
        q_strength=5.0,
        tau_c=0.5,
    )

    # Kernel scalars at g=1 base
    k0_0 = laplace_k0(cfg, 0.0, 1001)
    k0_p = laplace_k0(cfg, omega_q, 1001)
    k0_m = laplace_k0(cfg, -omega_q, 1001)
    r_p = resonant_r0(cfg, omega_q, 1001)
    r_m = resonant_r0(cfg, -omega_q, 1001)
    r_minus = 0.5 * (r_p - r_m)
    k_minus = 0.5 * (k0_p - k0_m)
    k_plus = 0.5 * (k0_p + k0_m)

    g = df["g"].to_numpy(dtype=float)
    y = df["k_needed_for_delta"].to_numpy(dtype=float)
    run = df["run_factor"].to_numpy(dtype=float)

    # Candidate basis curves (all normalized/trend-only)
    candidates: dict[str, np.ndarray] = {}
    candidates["run_factor"] = run
    candidates["inv_run_factor"] = 1.0 / np.maximum(run, 1e-12)
    candidates["g2"] = g * g
    candidates["g2_Rminus"] = (g * g) * abs(r_minus)
    candidates["g2_Kminus"] = (g * g) * abs(k_minus)
    candidates["g2_Kplus"] = (g * g) * abs(k_plus)

    # gamma-like diagonal factor from R^- only
    x = np.maximum((g * g) * abs(r_minus), 1e-12)
    candidates["gamma_diag_like"] = np.tanh(x) / x

    # Fit each candidate with affine mapping y ~ a + b*u
    rows = []
    fit_curves = {"g": g, "k_needed": y}
    for name, u in candidates.items():
        a, b, yhat = _fit_affine_to_target(y, u)
        rows.append(
            {
                "candidate": name,
                "corr_with_kneeded": _corr(y, u),
                "fit_a": a,
                "fit_b": b,
                "rmse": _rmse(y, yhat),
            }
        )
        fit_curves[f"fit_{name}"] = yhat

    # Also fit pure Padé in g^2; often best empirical model
    a_p, b_p, c_p, yhat_p = _fit_pade_g2(y, g)
    rows.append(
        {
            "candidate": "pade_g2",
            "corr_with_kneeded": np.nan,
            "fit_a": a_p,
            "fit_b": b_p,
            "fit_c": c_p,
            "rmse": _rmse(y, yhat_p),
        }
    )
    fit_curves["fit_pade_g2"] = yhat_p

    summary = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    fit_df = pd.DataFrame(fit_curves)

    summary_csv = out_dir / "hmf_kernel_factor_similarity_summary_codex_v11.csv"
    curves_csv = out_dir / "hmf_kernel_factor_similarity_curves_codex_v11.csv"
    fig_png = out_dir / "hmf_kernel_factor_similarity_codex_v11.png"
    log_md = out_dir / "hmf_kernel_factor_similarity_log_codex_v11.md"
    summary.to_csv(summary_csv, index=False)
    fit_df.to_csv(curves_csv, index=False)

    # Plot only top 4 fits for readability
    top = summary.head(4)["candidate"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(g, y, "o-", color="black", linewidth=2, label="k_needed(g) inferred")
    colors = ["#0B6E4F", "#1F4E79", "#AA3377", "#CC7722"]
    for c, name in zip(colors, top):
        ax.plot(g, fit_df[f"fit_{name}"], "--", linewidth=1.8, color=c, label=f"{name}")
    ax.set_xlabel("g")
    ax.set_ylabel("k_needed")
    ax.set_title("Does inferred diagonal factor match a kernel-derived trend?")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_png, dpi=180)
    plt.close(fig)

    lines = []
    lines.append("# Kernel Similarity Diagnostic (Codex v11)")
    lines.append("")
    lines.append(f"Kernel scalars: K0={k0_0:.6f}, K+={k_plus:.6f}, K-={k_minus:.6f}, R-={r_minus:.6f}")
    lines.append("")
    lines.append("| candidate | corr_with_kneeded | fit_a | fit_b | fit_c | rmse |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        fc = r["fit_c"] if "fit_c" in r and pd.notna(r.get("fit_c", np.nan)) else np.nan
        lines.append(
            f"| {r['candidate']} | {r.get('corr_with_kneeded', np.nan):.6f} | {r['fit_a']:.6f} | "
            f"{r['fit_b']:.6f} | {fc:.6f} | {r['rmse']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", summary_csv.name)
    print("Wrote:", curves_csv.name)
    print("Wrote:", fig_png.name)
    print("Wrote:", log_md.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
