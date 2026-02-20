"""
Diagonal-channel mismatch diagnostic with coherence plots (Codex v8).

Purpose:
1) Test whether flipping sign of Delta_z helps (it should not, but quantified here).
2) Infer an effective Delta_z factor from ordered data at theta=pi/2.
3) Test g-dependent diagonal scalings (including extra g^2 hypothesis).
4) Compare coherence matching (|rho_01|) at generic theta.

Outputs are fresh codex_v8 files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    RenormConfig,
    laplace_k0,
    ordered_gaussian_state,
    resonant_r0,
    extract_density,
)


@dataclass
class ChannelTriple:
    sigma_plus: float
    sigma_minus: float
    delta_z: float


def _rmse(x: Iterable[float]) -> float:
    arr = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0


def _base_channels(cfg: BenchmarkConfig) -> ChannelTriple:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    c = float(np.cos(cfg.theta))
    s = float(np.sin(cfg.theta))
    k0_0 = laplace_k0(cfg, 0.0, 1001)
    k0_p = laplace_k0(cfg, w, 1001)
    k0_m = laplace_k0(cfg, -w, 1001)
    r_p = resonant_r0(cfg, w, 1001)
    r_m = resonant_r0(cfg, -w, 1001)
    sigma_plus0 = (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sigma_minus0 = (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    delta_z0 = (s**2) * 0.5 * (r_p - r_m)
    return ChannelTriple(float(sigma_plus0), float(sigma_minus0), float(delta_z0))


def _apply_running(cfg: BenchmarkConfig, channels: ChannelTriple, ren: RenormConfig) -> tuple[ChannelTriple, float]:
    sp = float(channels.sigma_plus)
    sm = float(channels.sigma_minus)
    dz = float(channels.delta_z)
    chi_raw = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    if chi_raw <= ren.eps:
        return ChannelTriple(sp, sm, dz), 1.0
    a = 0.5 * cfg.beta * cfg.omega_q
    chi_cap = max(ren.kappa * abs(a), ren.eps)
    run = 1.0 / (1.0 + chi_raw / chi_cap)
    return ChannelTriple(run * sp, run * sm, run * dz), float(run)


def _components_from_channels(cfg: BenchmarkConfig, ch: ChannelTriple) -> tuple[float, float, float, float]:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    sp = float(ch.sigma_plus)
    sm = float(ch.sigma_minus)
    dz = float(ch.delta_z)
    chi = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-12 else 1.0
    u = gamma * dz
    q = gamma * np.sqrt(max(sp * sm, 0.0))
    a = 0.5 * beta * w
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-14:
        den = 1e-14 if den >= 0.0 else -1e-14
    mz = (u - t) / den
    mx = (q / np.cosh(a)) / den
    p00 = float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))
    p11 = float(np.clip(0.5 * (1.0 - mz), 0.0, 1.0))
    z = p00 + p11
    if z <= 0.0:
        p00, p11 = 0.5, 0.5
    else:
        p00, p11 = p00 / z, p11 / z
    coh = float(0.5 * abs(mx))
    ratio = float(p00 / max(p11, 1e-15))
    return p00, p11, coh, ratio


def _ordered_u_from_p00(beta: float, omega_q: float, p00: float) -> float:
    mz = 2.0 * float(p00) - 1.0
    t = float(np.tanh(0.5 * beta * omega_q))
    den = 1.0 + mz * t
    if abs(den) < 1e-14:
        return np.nan
    return float((mz + t) / den)


def _model_with_delta_transform(
    cfg: BenchmarkConfig,
    g: float,
    ren: RenormConfig,
    delta_sign: float = 1.0,
    delta_mult_const: float = 1.0,
    delta_mult_gpow: float = 0.0,
) -> tuple[float, float, float, float, float]:
    base0 = _base_channels(cfg)
    g2 = float(g * g)
    base = ChannelTriple(g2 * base0.sigma_plus, g2 * base0.sigma_minus, g2 * base0.delta_z)
    run_ch, run_factor = _apply_running(cfg, base, ren)
    mult = float(delta_mult_const * (g ** delta_mult_gpow if g > 0 else 1.0))
    ch = ChannelTriple(run_ch.sigma_plus, run_ch.sigma_minus, delta_sign * mult * run_ch.delta_z)
    p00, p11, coh, ratio = _components_from_channels(cfg, ch)
    return p00, p11, coh, ratio, run_factor


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ren = RenormConfig(scale=1.04, kappa=0.94)

    # A) Infer diagonal mismatch factor at theta=pi/2 via ordered p00.
    beta_ref = 2.0
    theta_ref = float(np.pi / 2.0)
    gs = np.linspace(0.0, 2.0, 21)
    rows_a = []
    for g in gs:
        cfg = BenchmarkConfig(
            beta=beta_ref,
            omega_q=2.0,
            theta=theta_ref,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        p00_ord, p11_ord, _ = extract_density(ordered_gaussian_state(cfg, float(g)))
        u_ord = _ordered_u_from_p00(cfg.beta, cfg.omega_q, p00_ord)

        # Baseline running (+delta)
        p00_plus, _, _, ratio_plus, run_factor = _model_with_delta_transform(
            cfg, float(g), ren, delta_sign=1.0, delta_mult_const=1.0, delta_mult_gpow=0.0
        )
        u_plus = float(np.tanh(ren.scale * (g * g) * _base_channels(cfg).delta_z * run_factor))

        # Sign-flipped delta
        p00_minus, _, _, ratio_minus, _ = _model_with_delta_transform(
            cfg, float(g), ren, delta_sign=-1.0, delta_mult_const=1.0, delta_mult_gpow=0.0
        )

        # Infer needed effective dz factor in this channel.
        base0 = _base_channels(cfg)
        base_dz = ren.scale * (g * g) * base0.delta_z
        run_dz = run_factor * base_dz
        dz_ord = float(np.arctanh(np.clip(u_ord, -0.999999999999, 0.999999999999))) if np.isfinite(u_ord) else np.nan
        k_needed = float(dz_ord / run_dz) if abs(run_dz) > 1e-14 and np.isfinite(dz_ord) else np.nan

        rows_a.append(
            {
                "g": float(g),
                "ordered_p00": float(p00_ord),
                "ordered_ratio": float(p00_ord / max(p11_ord, 1e-15)),
                "ordered_u_inferred": float(u_ord),
                "compact_plus_p00": float(p00_plus),
                "compact_plus_ratio": float(ratio_plus),
                "compact_minus_p00": float(p00_minus),
                "compact_minus_ratio": float(ratio_minus),
                "k_needed_for_delta": k_needed,
                "run_factor": float(run_factor),
            }
        )
    df_a = pd.DataFrame.from_records(rows_a)
    k_const = float(np.nanmedian(df_a.loc[df_a["g"] >= 0.1, "k_needed_for_delta"]))

    # B) Compare coherence at generic theta using candidate delta transforms.
    theta_cmp = float(np.pi / 4.0)
    beta_cmp = 2.0
    rows_b = []
    for g in np.linspace(0.0, 2.0, 21):
        cfg = BenchmarkConfig(
            beta=beta_cmp,
            omega_q=2.0,
            theta=theta_cmp,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        p00_ord, p11_ord, coh_ord = extract_density(ordered_gaussian_state(cfg, float(g)))
        ratio_ord = float(p00_ord / max(p11_ord, 1e-15))

        variants = {
            "plus_baseline": dict(delta_sign=1.0, delta_mult_const=1.0, delta_mult_gpow=0.0),
            "minus_signflip": dict(delta_sign=-1.0, delta_mult_const=1.0, delta_mult_gpow=0.0),
            "plus_const_k": dict(delta_sign=1.0, delta_mult_const=k_const, delta_mult_gpow=0.0),
            "plus_extra_g2": dict(delta_sign=1.0, delta_mult_const=k_const, delta_mult_gpow=2.0),
        }
        for name, spec in variants.items():
            p00, p11, coh, ratio, _ = _model_with_delta_transform(cfg, float(g), ren, **spec)
            rows_b.append(
                {
                    "g": float(g),
                    "variant": name,
                    "ordered_p00": float(p00_ord),
                    "ordered_coh": float(coh_ord),
                    "ordered_ratio": ratio_ord,
                    "model_p00": float(p00),
                    "model_coh": float(coh),
                    "model_ratio": float(ratio),
                    "dp00": float(p00 - p00_ord),
                    "dcoh": float(coh - coh_ord),
                    "ratio_over_ordered": float(ratio / max(ratio_ord, 1e-15)),
                }
            )
    df_b = pd.DataFrame.from_records(rows_b)

    # Summary
    rows_s = []
    for name, g in df_b.groupby("variant"):
        rows_s.append(
            {
                "variant": name,
                "rmse_p00_vs_ordered": _rmse(g["dp00"]),
                "rmse_coh_vs_ordered": _rmse(g["dcoh"]),
                "ratio_over_ordered_median": float(np.median(g["ratio_over_ordered"])),
                "ratio_over_ordered_max": float(np.max(g["ratio_over_ordered"])),
            }
        )
    summary = pd.DataFrame.from_records(rows_s).sort_values("rmse_p00_vs_ordered").reset_index(drop=True)

    # Write data
    path_a = out_dir / "hmf_delta_factor_scan_codex_v8.csv"
    path_b = out_dir / "hmf_delta_coherence_variants_codex_v8.csv"
    path_s = out_dir / "hmf_delta_factor_summary_codex_v8.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    summary.to_csv(path_s, index=False)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # A1: theta=pi/2 population
    ax = axes[0, 0]
    ax.plot(df_a["g"], df_a["ordered_p00"], color="black", linewidth=2, label="Ordered")
    ax.plot(df_a["g"], df_a["compact_plus_p00"], color="#0B6E4F", linewidth=2, label="Delta sign +")
    ax.plot(df_a["g"], df_a["compact_minus_p00"], color="#C84B31", linestyle="--", linewidth=1.8, label="Delta sign -")
    ax.set_title(r"$\theta=\pi/2$: population sign test")
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # A2: inferred k(g)
    ax = axes[0, 1]
    ax.plot(df_a["g"], df_a["k_needed_for_delta"], color="#1F4E79", linewidth=2, label="k_needed(g)")
    ax.axhline(k_const, color="#AA5500", linestyle="--", linewidth=1.5, label=f"k_const={k_const:.3f}")
    ax.set_title(r"Inferred diagonal factor from ordered ($\theta=\pi/2$)")
    ax.set_xlabel("g")
    ax.set_ylabel("k")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # B1: coherence comparison at theta=pi/4
    ax = axes[1, 0]
    ord_g = df_b[df_b["variant"] == "plus_baseline"][["g", "ordered_coh"]].drop_duplicates().sort_values("g")
    ax.plot(ord_g["g"], ord_g["ordered_coh"], color="black", linewidth=2, label="Ordered")
    for name, color, ls in [
        ("plus_baseline", "#0B6E4F", "-"),
        ("plus_const_k", "#1F4E79", "-."),
        ("plus_extra_g2", "#AA3377", ":"),
        ("minus_signflip", "#C84B31", "--"),
    ]:
        g = df_b[df_b["variant"] == name].sort_values("g")
        ax.plot(g["g"], g["model_coh"], color=color, linestyle=ls, linewidth=1.8, label=name)
    ax.set_title(r"$\theta=\pi/4$: coherence comparison")
    ax.set_xlabel("g")
    ax.set_ylabel(r"$|\rho_{01}|$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # B2: population comparison at theta=pi/4
    ax = axes[1, 1]
    ord_p = df_b[df_b["variant"] == "plus_baseline"][["g", "ordered_p00"]].drop_duplicates().sort_values("g")
    ax.plot(ord_p["g"], ord_p["ordered_p00"], color="black", linewidth=2, label="Ordered")
    for name, color, ls in [
        ("plus_baseline", "#0B6E4F", "-"),
        ("plus_const_k", "#1F4E79", "-."),
        ("plus_extra_g2", "#AA3377", ":"),
        ("minus_signflip", "#C84B31", "--"),
    ]:
        g = df_b[df_b["variant"] == name].sort_values("g")
        ax.plot(g["g"], g["model_p00"], color=color, linestyle=ls, linewidth=1.8, label=name)
    ax.set_title(r"$\theta=\pi/4$: population comparison")
    ax.set_xlabel("g")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig_path = out_dir / "hmf_delta_factor_coherence_diag_codex_v8.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Log
    log = []
    log.append("# Delta-Factor and Coherence Diagnostic (Codex v8)")
    log.append("")
    log.append(f"- Inferred constant diagonal factor from theta=pi/2: `k_const = {k_const:.6f}`")
    log.append("- Sign-flip and extra-g^2 variants included.")
    log.append("")
    log.append("| variant | rmse_p00_vs_ordered | rmse_coh_vs_ordered | ratio_over_ordered_median | ratio_over_ordered_max |")
    log.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        log.append(
            f"| {r['variant']} | {r['rmse_p00_vs_ordered']:.6f} | {r['rmse_coh_vs_ordered']:.6f} | "
            f"{r['ratio_over_ordered_median']:.6f} | {r['ratio_over_ordered_max']:.6f} |"
        )
    log_path = out_dir / "hmf_delta_factor_log_codex_v8.md"
    log_path.write_text("\n".join(log), encoding="utf-8")

    print("Wrote:", path_a.name)
    print("Wrote:", path_b.name)
    print("Wrote:", path_s.name)
    print("Wrote:", fig_path.name)
    print("Wrote:", log_path.name)
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

