"""
Simulation script for HMF v4 validation.
- Runs Exact Diagonalization (ED) and HMF v4 analytic calculations.
- Saves results to simulations/results/data/hmf_v4_validation.csv.

Run with: ./run_safe.ps1 simulations/src/hmf_v4_generate.py
"""

import numpy as np
import pandas as pd
from scipy.linalg import logm
import os

from prl127_qubit_benchmark import (
    BenchmarkConfig,
    SIGMA_X,
    SIGMA_Z,
    IDENTITY_2,
    build_static_operators,
    thermal_state,
    partial_trace_bath,
    trace_distance,
)
from prl127_qubit_analytic_bridge import (
    compute_delta_coefficients,
    finite_hmf_symmetric_product_state,
)

SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def run_validation(config: BenchmarkConfig):
    print(f"Running validation for beta={config.beta}, w_q={config.omega_q}, theta={config.theta}")
    
    # Build operators (expensive)
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)
    
    _, delta_x, delta_y, delta_z = compute_delta_coefficients(config, n_kernel_grid=4001)

    dim_system = h_s.shape[0]
    dim_bath = h_static.shape[0] // dim_system
    
    lambdas = np.linspace(config.lambda_min, config.lambda_max, config.lambda_points)
    
    results = []
    
    for i, lam in enumerate(lambdas):
        if i % 5 == 0:
            print(f"Processing lambda {lam:.2f} ({i+1}/{len(lambdas)})...")
            
        # Exact Calculation
        h_tot = h_static + lam * h_int + (lam**2) * h_counter
        rho_tot = thermal_state(h_tot, config.beta)
        rho_ex = partial_trace_bath(rho_tot, dim_system, dim_bath)
        
        # Analytic HMF v4
        rho_v4 = finite_hmf_symmetric_product_state(config, lam, delta_x, delta_y, delta_z)
        
        # Distance
        dist_v4 = trace_distance(rho_ex, rho_v4)
        
        # Extract numerical fields
        log_rho = logm(rho_ex + 1e-12 * IDENTITY_2)
        h_eff_num = (-1.0/config.beta) * log_rho
        
        hx_num = np.real(np.trace(h_eff_num @ SIGMA_X)) / 2.0
        hy_num = np.real(np.trace(h_eff_num @ SIGMA_Y)) / 2.0
        hz_num = np.real(np.trace(h_eff_num @ SIGMA_Z)) / 2.0

        log_rho_v4 = logm(rho_v4 + 1e-12 * IDENTITY_2)
        h_eff_v4 = (-1.0 / config.beta) * log_rho_v4
        hx_v4 = np.real(np.trace(h_eff_v4 @ SIGMA_X)) / 2.0
        hy_v4 = np.real(np.trace(h_eff_v4 @ SIGMA_Y)) / 2.0
        hz_v4 = np.real(np.trace(h_eff_v4 @ SIGMA_Z)) / 2.0
        sym_nonherm = float(np.max(np.abs(rho_v4 - rho_v4.conj().T)))
        
        # Rotate numerical fields using the model-implied in-plane angle
        phi = np.arctan2(hy_v4, hx_v4)
        hx_rot = hx_num * np.cos(phi) + hy_num * np.sin(phi)
        hy_rot = -hx_num * np.sin(phi) + hy_num * np.cos(phi)
        
        results.append({
            "lambda": lam,
            "dist_v4": dist_v4,
            "hx_v4": hx_v4,
            "hy_v4": hy_v4,
            "hz_v4": hz_v4,
            "phi_v4": float(phi),
            "hx_num_lab": hx_num,
            "hy_num_lab": hy_num,
            "hx_num_rot": hx_rot,
            "hy_num_rot": hy_rot,
            "hz_num": hz_num,
            "sym_nonherm_maxabs": sym_nonherm,
        })
        
    df = pd.DataFrame(results)
    
    # Save output
    output_path = os.path.join("simulations", "results", "data", "hmf_v4_validation.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    # Standard Config
    conf = BenchmarkConfig(
        beta=1.0, 
        omega_q=3.0, 
        theta=0.25, 
        lambda_max=4.0, 
        n_modes=4,   # Reduced from 5 to prevent OOM
        n_cut=6,     # Sufficient cutoff
        omega_max=10.0
    )
    
    run_validation(conf)
