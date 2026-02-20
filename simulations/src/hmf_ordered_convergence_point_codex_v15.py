"""
Single-point ordered convergence probe for the diagonal-population mismatch.

Target point chosen from the problematic regime:
    beta=2.0, theta=pi/2, g=0.5

This script varies ordered numerical controls (time slices, KL rank, GH order)
and checks whether ordered rho_00 moves toward the exact closed-form rho_00.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig,
    IDENTITY_2,
    _kernel_profile,
    _x_tilde_operator,
    laplace_k0,
    resonant_r0,
)


@dataclass
class NumCfg:
    n_slices: int
    kl_rank: int
    gh_order: int


def _project_density(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    evals, evecs = eigh(rho)
    evals = np.clip(np.real(evals), 1e-15, None)
    rho = (evecs * evals) @ evecs.conj().T
    rho /= np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


def ordered_state_param(cfg: BenchmarkConfig, g: float, num: NumCfg) -> np.ndarray:
    t = np.linspace(0.0, cfg.beta, int(num.n_slices), dtype=float)
    dt = float(t[1] - t[0])
    u = np.abs(t[:, None] - t[None, :])
    u = np.minimum(u, cfg.beta - u)
    cov = _kernel_profile(cfg, u)

    evals, evecs = eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = np.clip(evals[idx], 0.0, None)
    evecs = evecs[:, idx]
    eff_rank = min(int(num.kl_rank), int(np.count_nonzero(evals > 1e-14)))

    if eff_rank > 0:
        kl_basis = evecs[:, :eff_rank] * np.sqrt(evals[:eff_rank])
    else:
        kl_basis = np.zeros((len(t), 0), dtype=float)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(int(num.gh_order))
    eta_1d = np.sqrt(2.0) * gh_x
    weights_1d = gh_w / np.sqrt(np.pi)

    x_grid = np.asarray([_x_tilde_operator(cfg, tau) for tau in t], dtype=complex)
    w_avg = np.zeros((2, 2), dtype=complex)

    for inds in itertools.product(range(int(num.gh_order)), repeat=eff_rank):
        ind_arr = np.asarray(inds, dtype=int)
        eta_vec = eta_1d[ind_arr] if eff_rank > 0 else np.zeros(0, dtype=float)
        xi = g * (kl_basis @ eta_vec) if eff_rank > 0 else np.zeros(len(t), dtype=float)
        weight = float(np.prod(weights_1d[ind_arr])) if eff_rank > 0 else 1.0

        u_op = IDENTITY_2.copy()
        for n in range(len(t)):
            v = dt * xi[n]
            step = np.cosh(v) * IDENTITY_2 - np.sinh(v) * x_grid[n]
            u_op = step @ u_op
        w_avg += weight * u_op

    hs = 0.5 * cfg.omega_q * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    from scipy.linalg import expm

    rho = expm(-cfg.beta * hs) @ w_avg
    return _project_density(rho)


def _tex_p00(cfg: BenchmarkConfig, g: float) -> float:
    beta = float(cfg.beta)
    w = float(cfg.omega_q)
    c = float(np.cos(cfg.theta))
    s = float(np.sin(cfg.theta))
    g2 = float(g * g)

    k0_0 = laplace_k0(cfg, 0.0, 1201)
    k0_p = laplace_k0(cfg, w, 1201)
    k0_m = laplace_k0(cfg, -w, 1201)
    r_p = resonant_r0(cfg, w, 1201)
    r_m = resonant_r0(cfg, -w, 1201)

    sp = g2 * (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sm = g2 * (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    dz = g2 * (s * s) * 0.5 * (r_p - r_m)

    chi = float(np.sqrt(max(dz * dz + sp * sm, 0.0)))
    gamma = float(np.tanh(chi) / chi) if chi > 1e-14 else 1.0
    a = 0.5 * beta * w
    z_q = 2.0 * (np.cosh(a) - gamma * dz * np.sinh(a))
    p00 = float(np.exp(-a) * (1.0 + gamma * dz) / z_q)
    return p00


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / "hmf_ordered_convergence_point_scan_codex_v15.csv"
    log_md = out_dir / "hmf_ordered_convergence_point_log_codex_v15.md"

    cfg = BenchmarkConfig(
        beta=2.0,
        omega_q=2.0,
        theta=float(np.pi / 2.0),
        n_modes=40,
        n_cut=1,
        omega_min=0.1,
        omega_max=10.0,
        q_strength=5.0,
        tau_c=0.5,
    )
    g = 0.5
    target_tex = _tex_p00(cfg, g)

    combos = [
        NumCfg(40, 4, 4),  # baseline
        NumCfg(40, 5, 4),
        NumCfg(40, 6, 4),
        NumCfg(40, 4, 5),
        NumCfg(40, 5, 5),
        NumCfg(60, 4, 4),
        NumCfg(60, 5, 4),
        NumCfg(60, 5, 5),
    ]

    rows: list[dict[str, float | int]] = []
    for num in combos:
        t0 = perf_counter()
        rho = ordered_state_param(cfg, g, num)
        dt = perf_counter() - t0
        p00 = float(np.real(rho[0, 0]))
        rows.append(
            {
                "n_slices": int(num.n_slices),
                "kl_rank": int(num.kl_rank),
                "gh_order": int(num.gh_order),
                "ordered_p00": p00,
                "tex_p00": target_tex,
                "delta_ordered_minus_tex": p00 - target_tex,
                "abs_delta": abs(p00 - target_tex),
                "runtime_s": dt,
            }
        )

    df = pd.DataFrame.from_records(rows).sort_values(["n_slices", "kl_rank", "gh_order"]).reset_index(drop=True)
    df.to_csv(scan_csv, index=False)

    lines: list[str] = []
    lines.append("# Ordered Convergence Probe (Codex v15)")
    lines.append("")
    lines.append("Target point: beta=2.0, theta=pi/2, g=0.5")
    lines.append(f"Exact closed-form tex_p00 = {target_tex:.12f}")
    lines.append("")
    lines.append("| n_slices | kl_rank | gh_order | ordered_p00 | delta_ordered_minus_tex | abs_delta | runtime_s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {int(r['n_slices'])} | {int(r['kl_rank'])} | {int(r['gh_order'])} | {r['ordered_p00']:.12f} | "
            f"{r['delta_ordered_minus_tex']:.12f} | {r['abs_delta']:.12f} | {r['runtime_s']:.3f} |"
        )
    log_md.write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", scan_csv.name)
    print("Wrote:", log_md.name)
    print("")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
