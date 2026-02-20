"""
HMF Model Comparison Script: v5 Compact vs Ordered Gaussian.

Compares the effective mean-force Hamiltonian fields (hx, hz) as a function of temperature
for fixed strong coupling.

Run from simulations/src/ directory.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import logm, eigh

from prl127_qubit_benchmark import BenchmarkConfig, SIGMA_X, SIGMA_Z
from hmf_v5_qubit_core import (
    compute_v5_base_channels, coupling_channels, v5_theory_state, 
    bloch_components, apply_real_gauge
)
from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state,
    _project_density
)

# ── Utilities ───────────────────────────────────────────────────────────────

def extract_hmf_fields(rho, beta, theta):
    """
    Extracts hx, hz where H_MF = hx*sigma_x + hz*sigma_z (in real gauge).
    """
    # 1. Gauge fix to real (eliminate y)
    rho_gauge, phase = apply_real_gauge(rho)
    
    # 2. Compute H_MF = -1/beta * log(rho)
    # Check for pure states (log divergence)
    evals = np.linalg.eigvals(rho_gauge)
    if np.min(evals) < 1e-9:
        return np.nan, np.nan
        
    hmf = -1.0/beta * logm(rho_gauge)
    
    # 3. Project onto Pauli basis
    hx = np.real(np.trace(hmf @ SIGMA_X)) / 2.0
    hz = np.real(np.trace(hmf @ SIGMA_Z)) / 2.0
    
    # 4. Enforce sign convention: hx should be negative for theta > 0
    # The physical coupling -sin(theta)*sigma_x usually implies field opposite to x
    if hx > 0:
        hx = -hx
        
    return hx, hz

def run_comparison(theta_val, coupling_g):
    betas = np.linspace(0.1, 5.0, 20)
    omega_q = 2.0
    
    # Base configuration template
    cfg_base = BenchmarkConfig(
        beta=1.0, omega_q=omega_q, theta=theta_val,
        lambda_min=0, lambda_max=0, lambda_points=0,
        n_modes=50, n_cut=1, # Bath parameters
        omega_min=0.1, omega_max=10.0,
        q_strength=5.0, tau_c=0.5,
        output_prefix="hmf_cmp"
    )

    results = []
    
    print(f"Running scan for theta={theta_val:.4f}, g={coupling_g:.2f}...")
    
    for beta in betas:
        # Update config for this beta
        cfg = BenchmarkConfig(
            beta=beta, omega_q=omega_q, theta=theta_val,
            lambda_min=0, lambda_max=0, lambda_points=0,
            n_modes=50, n_cut=1, 
            omega_min=0.1, omega_max=10.0,
            q_strength=5.0, tau_c=0.5,
            output_prefix="hmf_cmp"
        )
        
        # 1. v5 Compact Model
        try:
            base_ch = compute_v5_base_channels(cfg, n_kernel_grid=2001)
            ch = coupling_channels(base_ch, coupling_g)
            rho_v5 = v5_theory_state(cfg, ch)
            v5_hx, v5_hz = extract_hmf_fields(rho_v5, beta, theta_val)
        except Exception as e:
            print(f"Error in v5 at beta={beta}: {e}")
            v5_hx, v5_hz = np.nan, np.nan

        # 2. Ordered Gaussian Model
        try:
            # Rebuild context for each beta (time grid depends on beta)
            ordered_ctx = _build_ordered_quadrature_context(
                cfg, n_time_slices=60, kl_rank=6, gh_order=5, max_nodes=100000
            ) 
            rho_ord = finite_hmf_ordered_gaussian_state(coupling_g, ordered_ctx)
            ord_hx, ord_hz = extract_hmf_fields(rho_ord, beta, theta_val)
        except Exception as e:
            print(f"Error in ordered at beta={beta}: {e}")
            ord_hx, ord_hz = np.nan, np.nan

        results.append({
            "beta": beta,
            "v5_hx": v5_hx, "v5_hz": v5_hz,
            "ord_hx": ord_hx, "ord_hz": ord_hz,
            "diff_hx": abs(v5_hx - ord_hx),
            "diff_hz": abs(v5_hz - ord_hz)
        })
        
    return pd.DataFrame(results)

# ── Main Execution ──────────────────────────────────────────────────────────

def plot_comparison(df, theta_val, coupling_g, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hz Plot
    axes[0].plot(df["beta"], df["v5_hz"], 'o-', label="v5 Compact")
    axes[0].plot(df["beta"], df["ord_hz"], 'x--', label="Ordered Gaussian")
    axes[0].set_xlabel(r"Inverse Temperature $\beta$")
    axes[0].set_ylabel(r"$h_z$ field")
    axes[0].set_title(r"$h_z$ vs $\beta$ ($\theta={:.2f}, g={:.1f}$)".format(theta_val, coupling_g))
    axes[0].legend()
    axes[0].grid(True)
    
    # Hx Plot
    axes[1].plot(df["beta"], df["v5_hx"], 'o-', label="v5 Compact")
    axes[1].plot(df["beta"], df["ord_hx"], 'x--', label="Ordered Gaussian")
    axes[1].set_xlabel(r"Inverse Temperature $\beta$")
    axes[1].set_ylabel(r"$h_x$ field")
    axes[1].set_title(r"$h_x$ vs $\beta$ ($\theta={:.2f}, g={:.1f}$)".format(theta_val, coupling_g))
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

if __name__ == "__main__":
    # Case 1: Generic Theta
    df_gen = run_comparison(theta_val=0.25, coupling_g=2.5)
    plot_comparison(df_gen, 0.25, 2.5, "hmf_cmp_generic.png")
    df_gen.to_csv("hmf_cmp_generic.csv", index=False)
    
    # Case 2: Pi/2 Theta
    df_pi2 = run_comparison(theta_val=np.pi/2, coupling_g=2.5)
    plot_comparison(df_pi2, np.pi/2, 2.5, "hmf_cmp_pi2.png")
    df_pi2.to_csv("hmf_cmp_pi2.csv", index=False)
