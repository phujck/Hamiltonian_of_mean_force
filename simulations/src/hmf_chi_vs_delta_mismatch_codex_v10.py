"""
Targeted diagnostic: is mismatch mainly in chi (gamma channel) or delta_z channel?

We compare analytic-vs-ordered population observables with three one-parameter variants:
  1) baseline
  2) chi-only scale k_chi in gamma: gamma = tanh(k_chi*chi)/(k_chi*chi)
  3) delta-only scale k_delta on delta_z before chi is formed

All runs are lightweight and intended for SAFE wrapper execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    RenormConfig,
    laplace_k0,
    resonant_r0,
    ordered_gaussian_state,
    extract_density,
)


def _rmse(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


@dataclass
class BaseChannels:
    sigma_plus: float
    sigma_minus: float
    delta_z: float
    run_factor: float


def _base_channels_running(cfg: BenchmarkConfig, g: float, ren: RenormConfig) -> BaseChannels:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    c = float(np.cos(cfg.theta))
    s = float(np.sin(cfg.theta))
    k0_0 = laplace_k0(cfg, 0.0, 1001)
    k0_p = laplace_k0(cfg, w, 1001)
    k0_m = laplace_k0(cfg, -w, 1001)
    r_p = resonant_r0(cfg, w, 1001)
    r_m = resonant_r0(cfg, -w, 1001)
    sp0 = (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sm0 = (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    dz0 = (s**2) * 0.5 * (r_p - r_m)

    g2 = float(g * g)
    sp = ren.scale * g2 * sp0
    sm = ren.scale * g2 * sm0
    dz = ren.scale * g2 * dz0

    chi_raw = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    if chi_raw <= ren.eps:
        run = 1.0
    else:
        a = 0.5 * beta * w
        chi_cap = max(ren.kappa * abs(a), ren.eps)
        run = 1.0 / (1.0 + chi_raw / chi_cap)

    return BaseChannels(
        sigma_plus=float(run * sp),
        sigma_minus=float(run * sm),
        delta_z=float(run * dz),
        run_factor=float(run),
    )


def _density_from_channels(
    beta: float,
    omega_q: float,
    sigma_plus: float,
    sigma_minus: float,
    delta_z: float,
    chi_scale: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Analytically normalized component route.
    chi_scale only affects gamma via chi -> chi_scale*chi.
    """
    chi = float(np.sqrt(max(delta_z * delta_z + sigma_plus * sigma_minus, 0.0)))
    chi_eff = float(chi_scale * chi)
    if chi_eff > 1e-12:
        gamma = float(np.tanh(chi_eff) / chi_eff)
    else:
        gamma = 1.0

    u = gamma * delta_z
    q = gamma * np.sqrt(max(sigma_plus * sigma_minus, 0.0))
    a = 0.5 * beta * omega_q
    t = float(np.tanh(a))
    den = 1.0 - u * t
    if abs(den) < 1e-14:
        den = 1e-14 if den >= 0 else -1e-14
    mz = (u - t) / den
    mx = (q / np.cosh(a)) / den

    p00 = float(np.clip(0.5 * (1.0 + mz), 0.0, 1.0))
    p11 = float(np.clip(0.5 * (1.0 - mz), 0.0, 1.0))
    z = p00 + p11
    if z <= 0:
        p00, p11 = 0.5, 0.5
    else:
        p00, p11 = p00 / z, p11 / z
    coh = float(0.5 * abs(mx))
    ratio = float(p00 / max(p11, 1e-15))
    return p00, p11, coh, ratio


def _evaluate_variant(
    cfg: BenchmarkConfig,
    g: float,
    ren: RenormConfig,
    variant: str,
    k: float,
) -> tuple[float, float, float, float]:
    ch = _base_channels_running(cfg, g, ren)
    if variant == "baseline":
        return _density_from_channels(cfg.beta, cfg.omega_q, ch.sigma_plus, ch.sigma_minus, ch.delta_z, chi_scale=1.0)
    if variant == "chi_only":
        return _density_from_channels(cfg.beta, cfg.omega_q, ch.sigma_plus, ch.sigma_minus, ch.delta_z, chi_scale=k)
    if variant == "delta_only":
        return _density_from_channels(
            cfg.beta, cfg.omega_q, ch.sigma_plus, ch.sigma_minus, k * ch.delta_z, chi_scale=1.0
        )
    raise ValueError(f"Unknown variant: {variant}")


def _fit_k_on_coupling(variant: str, ren: RenormConfig) -> float:
    beta = 2.0
    theta = float(np.pi / 4.0)
    gs = np.linspace(0.0, 2.0, 21)
    k_grid = np.linspace(0.4, 1.6, 241)
    best = (np.inf, 1.0)

    # Cache ordered references and base channels once.
    cache = []
    for g in gs:
        cfg = BenchmarkConfig(
            beta=beta,
            omega_q=2.0,
            theta=theta,
            n_modes=40,
            n_cut=1,
            omega_min=0.1,
            omega_max=10.0,
            q_strength=5.0,
            tau_c=0.5,
        )
        p00_ord, p11_ord, _ = extract_density(ordered_gaussian_state(cfg, float(g)))
        ratio_ord = float(p00_ord / max(p11_ord, 1e-15))
        base = _base_channels_running(cfg, float(g), ren)
        cache.append((cfg, base, p00_ord, ratio_ord))

    for k in k_grid:
        errs = []
        for cfg, base, p00_ord, ratio_ord in cache:
            if variant == "chi_only":
                p00, p11, _, ratio = _density_from_channels(
                    cfg.beta, cfg.omega_q, base.sigma_plus, base.sigma_minus, base.delta_z, chi_scale=float(k)
                )
            elif variant == "delta_only":
                p00, p11, _, ratio = _density_from_channels(
                    cfg.beta, cfg.omega_q, base.sigma_plus, base.sigma_minus, float(k) * base.delta_z, chi_scale=1.0
                )
            else:
                raise ValueError(f"Unknown variant for fit: {variant}")
            errs.append((p00 - p00_ord) ** 2)
            errs.append((ratio / max(ratio_ord, 1e-15) - 1.0) ** 2)
        score = float(np.mean(errs))
        if score < best[0]:
            best = (score, float(k))
    return best[1]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    ren = RenormConfig(scale=1.04, kappa=0.94)

    k_chi = _fit_k_on_coupling("chi_only", ren)
    k_delta = _fit_k_on_coupling("delta_only", ren)

    rows = []
    sweeps = [
        ("coupling", "g", np.linspace(0.0, 2.0, 21), 2.0, float(np.pi / 4.0), None),
        ("temperature", "beta", np.linspace(0.2, 6.0, 21), None, float(np.pi / 2.0), 0.5),
    ]
    for sweep_name, p_name, p_vals, beta_fixed, theta_fixed, g_fixed in sweeps:
        for p in p_vals:
            beta = float(p) if p_name == "beta" else float(beta_fixed)
            theta = float(p) if p_name == "theta" else float(theta_fixed)
            g = float(p) if p_name == "g" else float(g_fixed)
            cfg = BenchmarkConfig(
                beta=beta,
                omega_q=2.0,
                theta=theta,
                n_modes=40,
                n_cut=1,
                omega_min=0.1,
                omega_max=10.0,
                q_strength=5.0,
                tau_c=0.5,
            )
            p00_ord, p11_ord, coh_ord = extract_density(ordered_gaussian_state(cfg, g))
            ratio_ord = float(p00_ord / max(p11_ord, 1e-15))

            variants = [
                ("baseline", 1.0),
                ("chi_only", k_chi),
                ("delta_only", k_delta),
            ]
            for variant, kval in variants:
                p00, p11, coh, ratio = _evaluate_variant(cfg, g, ren, variant, kval)
                rows.append(
                    {
                        "sweep": sweep_name,
                        "param_name": p_name,
                        "param": float(p),
                        "beta": beta,
                        "theta": theta,
                        "g": g,
                        "variant": variant,
                        "k_value": float(kval),
                        "ordered_p00": float(p00_ord),
                        "ordered_coh": float(coh_ord),
                        "ordered_ratio": float(ratio_ord),
                        "model_p00": float(p00),
                        "model_coh": float(coh),
                        "model_ratio": float(ratio),
                        "dp00": float(p00 - p00_ord),
                        "dcoh": float(coh - coh_ord),
                        "ratio_rel_err": float(ratio / max(ratio_ord, 1e-15) - 1.0),
                    }
                )

    df = pd.DataFrame.from_records(rows)
    summary_rows = []
    for (sweep, variant), g in df.groupby(["sweep", "variant"]):
        summary_rows.append(
            {
                "sweep": sweep,
                "variant": variant,
                "k_value": float(g["k_value"].iloc[0]),
                "rmse_p00": _rmse(g["dp00"]),
                "rmse_coh": _rmse(g["dcoh"]),
                "rmse_ratio_rel": _rmse(g["ratio_rel_err"]),
                "ratio_rel_median": float(np.median(g["ratio_rel_err"])),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["sweep", "rmse_p00"]).reset_index(drop=True)

    scan_csv = out_dir / "hmf_chi_vs_delta_scan_codex_v10.csv"
    summary_csv = out_dir / "hmf_chi_vs_delta_summary_codex_v10.csv"
    log_md = out_dir / "hmf_chi_vs_delta_log_codex_v10.md"
    df.to_csv(scan_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    lines = []
    lines.append("# Chi vs Delta Mismatch Diagnostic (Codex v10)")
    lines.append("")
    lines.append(f"- fitted k_chi (coupling sweep objective): `{k_chi:.6f}`")
    lines.append(f"- fitted k_delta (coupling sweep objective): `{k_delta:.6f}`")
    lines.append("")
    lines.append("| sweep | variant | k_value | rmse_p00 | rmse_coh | rmse_ratio_rel | ratio_rel_median |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['sweep']} | {r['variant']} | {r['k_value']:.6f} | {r['rmse_p00']:.6f} | "
            f"{r['rmse_coh']:.6f} | {r['rmse_ratio_rel']:.6f} | {r['ratio_rel_median']:.6f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", summary_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(f"k_chi={k_chi:.6f}, k_delta={k_delta:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
