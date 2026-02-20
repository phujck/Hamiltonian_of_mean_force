"""
Ultra-High Precision HMF Simulation Sweep (v13).
Designed for remote execution (Google Colab / HPC).

Sweeps:
- Physical: beta, g, theta
- Convergence: n_modes, n_cut

Features:
- Self-contained configuration for remote portability.
- Robust checkpointing (CSV) to handle long runtimes and interruptions.
- Detailed logging of Hilbert space size and diagonalization time.
"""

from __future__ import annotations

import argparse
import time
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Standard HMF library imports (assumed available in PYTHONPATH or same directory)
from hmf_model_comparison_standalone_codex_v1 import (
    BenchmarkConfig as LiteConfig,
    extract_density,
    ordered_gaussian_state,
    RenormConfig,
)
from hmf_component_normalized_compare_codex_v5 import _compact_components
from hmf_v5_qubit_core import build_ed_context, exact_reduced_state
from prl127_qubit_benchmark import BenchmarkConfig as EDConfig

def _key_of(beta, g, theta, n_modes, n_cut):
    return f"b{beta:.3f}|g{g:.3f}|t{theta:.3f}|m{n_modes}|c{n_cut}"

def run_ultra_sweep(args):
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    scan_csv = out_dir / f"{args.output_prefix}_scan.csv"
    meta_json = out_dir / f"{args.output_prefix}_metadata.json"
    
    # Define Sweep Ranges
    betas = np.linspace(args.beta_min, args.beta_max, args.beta_points)
    gs = np.linspace(args.g_min, args.g_max, args.g_points)
    thetas = np.linspace(args.theta_min, args.theta_max, args.theta_points)
    n_modes_list = [int(x) for x in args.n_modes_list.split(",")]
    n_cut_list = [int(x) for x in args.n_cut_list.split(",")]
    
    tasks = [
        (b, g, t, m, c)
        for b in betas
        for g in gs
        for t in thetas
        for m in n_modes_list
        for c in n_cut_list
    ]
    
    total_tasks = len(tasks)
    checkpoint_every = 5
    
    # Resume Logic
    if scan_csv.exists() and args.resume:
        df = pd.read_csv(scan_csv)
        completed_keys = set(df["key"].astype(str).tolist())
    else:
        df = pd.DataFrame()
        completed_keys = set()
        
    print(f"[ULTRA-SWEEP] Total tasks: {total_tasks}, Existing: {len(completed_keys)}")
    
    ren = RenormConfig(scale=1.04, kappa=0.94)
    
    # Save Metadata for plotting script
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "physical_params": {"beta": betas.tolist(), "g": gs.tolist(), "theta": thetas.tolist()},
        "convergence_params": {"n_modes": n_modes_list, "n_cut": n_cut_list},
        "fixed_params": {"omega_q": 2.0, "omega_min": 0.1, "omega_max": 8.0, "q_strength": 5.0, "tau_c": 0.5}
    }
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    t_start = time.perf_counter()
    new_rows = []

    try:
        for idx, (beta, g, theta, n_modes, n_cut) in enumerate(tasks, start=1):
            key = _key_of(beta, g, theta, n_modes, n_cut)
            if key in completed_keys:
                continue
                
            hilbert_size = (n_cut) ** n_modes
            print(f"[{idx}/{total_tasks}] Running {key} (D={hilbert_size})...")
            
            t0 = time.perf_counter()
            row: dict[str, Any] = {
                "key": key,
                "beta": beta,
                "g": g,
                "theta": theta,
                "n_modes": n_modes,
                "n_cut": n_cut,
                "hilbert_size": hilbert_size,
                "status": "ok",
                "error": ""
            }
            
            try:
                # Analytic Benchmarks
                lite_cfg = LiteConfig(
                    beta=float(beta), omega_q=2.0, theta=float(theta),
                    n_modes=24, n_cut=1, omega_min=0.1, omega_max=8.0,
                    q_strength=5.0, tau_c=0.5
                )
                rho_ord = ordered_gaussian_state(lite_cfg, float(g))
                ord_p00, _, ord_coh = extract_density(rho_ord)
                best_p00, _, best_coh, _ = _compact_components(lite_cfg, float(g), use_running=True, renorm=ren)
                
                row.update({
                    "ord_p00": float(ord_p00),
                    "ord_p11": float(1.0 - ord_p00),
                    "ord_re01": float(np.real(rho_ord[0, 1])),
                    "ord_im01": float(np.imag(rho_ord[0, 1])),
                    "ord_coh": float(ord_coh),
                    "best_p00": float(best_p00),
                    "best_p11": float(1.0 - best_p00),
                    "best_re01": float(np.real(best_coh)), # best_coh is often just the magnitude in some models, need to be careful
                    "best_coh": float(abs(best_coh))
                })

                # High-Precision ED
                ed_cfg = EDConfig(
                    beta=float(beta), omega_q=2.0, theta=float(theta),
                    n_modes=int(n_modes), n_cut=int(n_cut), 
                    omega_min=0.1, omega_max=8.0, q_strength=5.0, tau_c=0.5,
                    lambda_min=0.0, lambda_max=max(1.0, float(g)),
                    lambda_points=2, output_prefix=f"{args.output_prefix}_tmp"
                )
                ed_ctx = build_ed_context(ed_cfg)
                rho_ed = exact_reduced_state(ed_ctx, float(g))
                ed_p00, ed_p11, ed_coh = extract_density(rho_ed)
                
                row.update({
                    "ed_p00": float(ed_p00),
                    "ed_p11": float(ed_p11),
                    "ed_re01": float(np.real(rho_ed[0, 1])),
                    "ed_im01": float(np.imag(rho_ed[0, 1])),
                    "ed_coh": float(ed_coh)
                })
                
            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
                row["status"] = "error"
                row["error"] = str(e)
            
            row["elapsed_s"] = time.perf_counter() - t0
            new_rows.append(row)
            completed_keys.add(key)
            
            # Flush to disk regularly
            if len(new_rows) >= checkpoint_every:
                temp_df = pd.DataFrame(new_rows)
                df = pd.concat([df, temp_df], ignore_index=True)
                df.to_csv(scan_csv, index=False)
                new_rows = []
                print(f"  [CHECKPOINT] Total time: {(time.perf_counter() - t_start)/60.0:.1f}m")
                
    except KeyboardInterrupt:
        print("[INTERRUPTED] Stopping and saving...")
        
    # Final save
    if new_rows:
        temp_df = pd.DataFrame(new_rows)
        df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv(scan_csv, index=False)
    print(f"[DONE] Sweep complete. Data: {scan_csv.name}")

def main():
    parser = argparse.ArgumentParser(description="Ultra-Precision HMF Sweep Tool.")
    # Physical
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--beta-points", type=int, default=11)
    parser.add_argument("--g-min", type=float, default=0.1)
    parser.add_argument("--g-max", type=float, default=1.0)
    parser.add_argument("--g-points", type=int, default=3)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=1.5707)
    parser.add_argument("--theta-points", type=int, default=2)
    
    # Convergence
    parser.add_argument("--n-modes-list", type=str, default="3,4,5,6")
    parser.add_argument("--n-cut-list", type=str, default="6,8,10,12")
    
    # Execution
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-prefix", type=str, default="hmf_v13_ultra_scan")
    parser.add_argument("--resume", action="store_true", default=True)
    
    args = parser.parse_args()
    run_ultra_sweep(args)

if __name__ == "__main__":
    main()
