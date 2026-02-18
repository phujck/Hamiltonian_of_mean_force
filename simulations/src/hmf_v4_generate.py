"""
Simulation script for HMF v4 validation.
- Runs Exact Diagonalization (ED) and HMF v4 analytic calculations.
- Saves results to simulations/results/data/hmf_v4_validation.csv.

Run with: ./run_safe.ps1 simulations/src/hmf_v4_generate.py
"""

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm
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
    spectral_density_exp
)

SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

def analytic_hmf_v4(config: BenchmarkConfig, lam_coupling: float, q_reorg_integral: float) -> np.ndarray:
    """
    Computes H_MF using the analytic result from results_v4.tex.
    """
    beta = config.beta
    omega_q = config.omega_q
    theta = config.theta
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Calculate paper's lambda (reorganization energy)
    lambda_paper = (lam_coupling**2) * (q_reorg_integral / np.pi)
    
    # Effective Kernel K(beta)
    kernel_eff = (4.0 * lambda_paper / omega_q) * (np.sinh(beta * omega_q / 2.0)**2)
    
    # Argument sK
    sk_val = s * kernel_eff
    
    # Omega definition
    phi = beta * omega_q / 2.0
    cos_omega = np.cos(phi) * np.cos(sk_val) + s * np.sin(phi) * np.sin(sk_val)
    
    omega_val = np.arccos(cos_omega)
    sin_omega = np.sin(omega_val)
    
    if abs(sin_omega) < 1e-9:
        prefactor = 1.0 # Limit case
        if abs(omega_val) > 1e-9:
             prefactor = omega_val / sin_omega
    else:
        prefactor = omega_val / sin_omega
        
    term_x = c * np.sin(sk_val)
    term_z = -np.sin(phi) * np.cos(sk_val) - s * np.cos(phi) * np.sin(sk_val)
    
    h_x = -(1.0/beta) * prefactor * term_x
    h_z = -(1.0/beta) * prefactor * term_z
    
    # Construct Hamiltonian
    H_mf = h_x * SIGMA_X + h_z * SIGMA_Z
    
    # Compute state
    rho = expm(-beta * H_mf)
    rho /= np.trace(rho)
    
    return rho, h_x, h_z


def run_validation(config: BenchmarkConfig):
    print(f"Running validation for beta={config.beta}, w_q={config.omega_q}, theta={config.theta}")
    
    # Build operators (expensive)
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)
    
    # Calculate q_reorg manually as before
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)])
        delta_omega = config.omega_max - config.omega_min
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes)
        delta_omega = omegas[1] - omegas[0]
    j_vals = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    g_vals = np.sqrt(np.maximum(j_vals, 0.0) * delta_omega)
    q_reorg = float(np.sum((g_vals**2) / np.maximum(omegas, 1e-12)))

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
        rho_v4, hx_v4, hz_v4 = analytic_hmf_v4(config, lam, q_reorg)
        
        # Distance
        dist_v4 = trace_distance(rho_ex, rho_v4)
        
        # Extract numerical fields
        log_rho = logm(rho_ex + 1e-12 * IDENTITY_2)
        h_eff_num = (-1.0/config.beta) * log_rho
        
        hx_num = np.real(np.trace(h_eff_num @ SIGMA_X)) / 2.0
        hy_num = np.real(np.trace(h_eff_num @ SIGMA_Y)) / 2.0
        hz_num = np.real(np.trace(h_eff_num @ SIGMA_Z)) / 2.0
        
        # Rotate numerical fields
        phi = config.beta * config.omega_q / 2.0
        hx_rot = hx_num * np.cos(phi) + hy_num * np.sin(phi)
        hy_rot = -hx_num * np.sin(phi) + hy_num * np.cos(phi)
        
        results.append({
            "lambda": lam,
            "dist_v4": dist_v4,
            "hx_v4": hx_v4,
            "hz_v4": hz_v4,
            "hx_num_lab": hx_num,
            "hy_num_lab": hy_num,
            "hx_num_rot": hx_rot,
            "hy_num_rot": hy_rot,
            "hz_num": hz_num
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
