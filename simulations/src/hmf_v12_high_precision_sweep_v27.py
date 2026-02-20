"""
High-precision HMF sweep script (v27).
Parameters: n_modes=4, n_cut=10.
Includes checkpointing for robust execution of large Hilbert space simulations.
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
    
    # High-precision settings
    N_MODES = 4
    N_CUT = 10
    G = 0.5
    THETA = np.pi / 2.0
    
    ren = RenormConfig(scale=1.04, kappa=0.94)
    
    if scan_csv.exists() and args.resume:
        df = pd.read_csv(scan_csv)
        done_betas = set(df["beta"].round(6))
    else:
        df = pd.DataFrame()
        done_betas = set()

    print(f"[START] High-precision sweep: n_modes={N_MODES}, n_cut={N_CUT}, total={len(betas)}")
    
    t_start = time.perf_counter()
    
    for i, beta in enumerate(betas):
        if round(beta, 6) in done_betas:
            continue
            
        t0 = time.perf_counter()
        
        # Analytic benchmarks
        lite_cfg = LiteConfig(
            beta=float(beta), omega_q=2.0, theta=THETA,
            n_modes=24, n_cut=1, omega_min=0.1, omega_max=8.0,
            q_strength=5.0, tau_c=0.5
        )
        rho_ord = ordered_gaussian_state(lite_cfg, G)
        ord_p00, _, ord_coh = extract_density(rho_ord)
        best_p00, _, best_coh, _ = _compact_components(lite_cfg, G, use_running=True, renorm=ren)
        
        # High-precision ED
        ed_cfg = EDConfig(
            beta=float(beta), omega_q=2.0, theta=THETA,
            n_modes=N_MODES, n_cut=N_CUT, omega_min=0.1, omega_max=8.0,
            q_strength=5.0, tau_c=0.5, lambda_min=0.0, lambda_max=1.0,
            lambda_points=2, output_prefix=f"{args.output_prefix}_edtmp"
        )
        ed_ctx = build_ed_context(ed_cfg)
        rho_ed = exact_reduced_state(ed_ctx, G)
        ed_p00, _, ed_coh = extract_density(rho_ed)
        
        dt = time.perf_counter() - t0
        print(f"  [{i+1}/{len(betas)}] beta={beta:.3f} | ED={ed_p00:.4f}, Best={best_p00:.4f} | {dt:.1f}s")
        
        row = {
            "beta": beta,
            "ed_p00": ed_p00,
            "best_p00": best_p00,
            "ordered_p00": ord_p00,
            "ed_coh": ed_coh,
            "best_coh": best_coh,
            "ordered_coh": ord_coh,
            "elapsed_s": dt
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(scan_csv, index=False) # Checkpoint every point
        
    # Final Plots and Log
    df = df.sort_values("beta")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(df["beta"], df["ed_p00"], "o", label=f"ED (m={N_MODES}, c={N_CUT})", markersize=5)
    ax.plot(df["beta"], df["best_p00"], "-", label="Best Analytic (v12)", linewidth=2, color="#0B6E4F")
    ax.plot(df["beta"], df["ordered_p00"], "--", label="Ordered Cumulant", alpha=0.6, color="black")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\rho_{00}$")
    ax.set_title(f"High-Precision HMF Sweep (n_modes={N_MODES}, n_cut={N_CUT})")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(fig_png, dpi=180)
    
    rmse_p00 = _rmse((df["ed_p00"] - df["best_p00"]).to_numpy())
    
    lines = [
        f"# High-Precision HMF Sweep Log (v27)",
        f"",
        f"Settings: n_modes={N_MODES}, n_cut={N_CUT}, g={G}, theta=pi/2",
        f"RMSE (ED vs Best Analytic): {rmse_p00:.6f}",
        f"Total Elapsed Time: {(time.perf_counter() - t_start)/60.0:.1f} min",
        f"",
        df.to_string(index=False)
    ]
    log_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] High-precision results in {scan_csv.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-min", type=float, default=0.4)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--beta-points", type=int, default=15)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--output-prefix", type=str, default="hmf_v12_high_precision_sweep_v27")
    args = parser.parse_args()
    run_sweep(args)

if __name__ == "__main__":
    main()
