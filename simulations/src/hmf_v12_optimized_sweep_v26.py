"""
Optimized HMF sweep script based on v12-v25 architecture.
Best bang-for-buck parameters: n_modes=3, n_cut=7.

Features:
- Fixed high-efficiency bath discretization.
- Temperature sweep for rho_00 and coherence.
- Comparison against 'Best v12' analytic theory.
- Checkpointed progress and automated plotting.
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    RenormConfig,
    extract_density,
    ordered_gaussian_state,
)
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig

def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0

def run_sweep(args: argparse.Namespace):
    out_dir = Path(__file__).resolve().parent
    scan_csv = out_dir / f"{args.output_prefix}_scan.csv"
    fig_png = out_dir / f"{args.output_prefix}.png"
    log_md = out_dir / f"{args.output_prefix}_log.md"

    betas = np.linspace(args.beta_min, args.beta_max, args.beta_points)
    
    # Optimized settings
    N_MODES = 3
    N_CUT = 7
    G = 0.5
    THETA = np.pi / 2.0
    
    ren = RenormConfig(scale=1.04, kappa=0.94)
    
    results = []
    
    print(f"[START] Optimized sweep: n_modes={N_MODES}, n_cut={N_CUT}, points={len(betas)}")
    
    for i, beta in enumerate(betas):
        t0 = time.perf_counter()
        
        # Analytic best
        lite_cfg = LiteConfig(
            beta=float(beta), omega_q=2.0, theta=THETA,
            n_modes=24, n_cut=1, omega_min=0.1, omega_max=8.0,
            q_strength=5.0, tau_c=0.5
        )
        rho_ord = ordered_gaussian_state(lite_cfg, G)
        ord_p00, _, ord_coh = extract_density(rho_ord)
        best_p00, _, best_coh, _ = _compact_components(lite_cfg, G, use_running=True, renorm=ren)
        
        # Optimized ED
        ed_cfg = EDConfig(
            beta=float(beta), omega_q=2.0, theta=THETA,
            n_modes=N_MODES, n_cut=N_CUT, omega_min=0.1, omega_max=8.0,
            q_strength=5.0, tau_c=0.5, lambda_min=0.0, lambda_max=1.0,
            lambda_points=2, output_prefix="temp_ed"
        )
        ed_ctx = build_ed_context(ed_cfg)
        rho_ed = exact_reduced_state(ed_ctx, G)
        ed_p00, _, ed_coh = extract_density(rho_ed)
        
        dt = time.perf_counter() - t0
        print(f"  [{i+1}/{len(betas)}] beta={beta:.2f} | ED p00={ed_p00:.4f}, Best p00={best_p00:.4f} | {dt:.2f}s")
        
        results.append({
            "beta": beta,
            "ed_p00": ed_p00,
            "best_p00": best_p00,
            "ordered_p00": ord_p00,
            "ed_coh": ed_coh,
            "best_coh": best_coh,
            "ordered_coh": ord_coh,
            "elapsed_s": dt
        })
        
    df = pd.DataFrame(results)
    df.to_csv(scan_csv, index=False)
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df["beta"], df["ed_p00"], "o", label=f"ED (m={N_MODES}, c={N_CUT})", markersize=4)
    ax.plot(df["beta"], df["best_p00"], "-", label="Best Analytic (v12)", linewidth=2)
    ax.plot(df["beta"], df["ordered_p00"], "--", label="Ordered Cumulant", alpha=0.7)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.set_title(f"Optimized HMF Sweep (n_modes={N_MODES}, n_cut={N_CUT})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(fig_png, dpi=150)
    
    # Log
    rmse_p00 = _rmse(df["ed_p00"] - df["best_p00"])
    lines = [
        f"# Optimized HMF Sweep Log (v26)",
        f"",
        f"Settings: n_modes={N_MODES}, n_cut={N_CUT}, g={G}, theta=pi/2",
        f"RMSE (ED vs Best Analytic): {rmse_p00:.6f}",
        f"",
        df.to_string(index=False)
    ]
    log_md.write_text("\n".join(lines))
    print(f"[DONE] Results saved to {scan_csv.name} and {fig_png.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-min", type=float, default=0.4)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--beta-points", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="hmf_v12_optimized_sweep_v26")
    args = parser.parse_args()
    run_sweep(args)

if __name__ == "__main__":
    main()
