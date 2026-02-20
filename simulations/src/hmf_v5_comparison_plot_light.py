"""
Fast, lightweight HMF Model Comparison.
Generates PNG plots for hmf_model_comparison.tex.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm

from prl127_qubit_benchmark import BenchmarkConfig, SIGMA_X, SIGMA_Z
from hmf_v5_qubit_core import (
    compute_v5_base_channels, coupling_channels, v5_theory_state, 
    apply_real_gauge
)
from prl127_qubit_analytic_bridge import (
    _build_ordered_quadrature_context,
    finite_hmf_ordered_gaussian_state
)

# ── Minimal Utils ──────────────────────────────────────────────────────────

def get_hmf_fields(rho, beta):
    # Gauge fix to real (eliminate y)
    rho_gauge, _ = apply_real_gauge(rho)
    
    # Check for pure states (log divergence)
    evals = np.linalg.eigvals(rho_gauge)
    if np.min(evals) < 1e-9:
        return np.nan, np.nan
        
    hmf = -1.0/beta * logm(rho_gauge)
    hx = np.real(np.trace(hmf @ SIGMA_X)) / 2.0
    hz = np.real(np.trace(hmf @ SIGMA_Z)) / 2.0
    if hx > 0: hx = -hx
    return hx, hz

def run_fast_scan(theta_val, coupling_g, filename):
    betas = np.linspace(0.1, 4.0, 15) # Reduced points for speed
    omega_q = 2.0
    
    v5_hx, v5_hz = [], []
    ord_hx, ord_hz = [], []
    
    # Pre-build ordered context (expensive part) ONCE per beta? No, depends on beta.
    # But we can use lower resolution for speed since it's "lightweight"
    
    print(f"Scanning theta={theta_val:.2f}, g={coupling_g}...")
    
    for beta in betas:
        cfg = BenchmarkConfig(
            beta=beta, omega_q=omega_q, theta=theta_val,
            lambda_min=0, lambda_max=0, lambda_points=0,
            n_modes=40, n_cut=1, 
            omega_min=0.1, omega_max=10.0,
            q_strength=5.0, tau_c=0.5,
            output_prefix="tmp"
        )
        
        # v5
        try:
            base_ch = compute_v5_base_channels(cfg, n_kernel_grid=1001)
            ch = coupling_channels(base_ch, coupling_g)
            rho = v5_theory_state(cfg, ch)
            hx, hz = get_hmf_fields(rho, beta)
            v5_hx.append(hx); v5_hz.append(hz)
        except:
            v5_hx.append(np.nan); v5_hz.append(np.nan)

        # Ordered
        try:
            # Low resolution context for speed
            ctx = _build_ordered_quadrature_context(
                cfg, n_time_slices=40, kl_rank=4, gh_order=4, max_nodes=50000
            )
            rho = finite_hmf_ordered_gaussian_state(coupling_g, ctx)
            hx, hz = get_hmf_fields(rho, beta)
            ord_hx.append(hx); ord_hz.append(hz)
        except:
            ord_hx.append(np.nan); ord_hz.append(np.nan)
            
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Hz
    axes[0].plot(betas, v5_hz, 'o-', label="v5 Compact")
    axes[0].plot(betas, ord_hz, 'x--', label="Ordered (Ref)")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$h_z$")
    axes[0].set_title(r"$h_z$ ($\theta={:.2f}$)".format(theta_val))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Hx
    axes[1].plot(betas, v5_hx, 'o-', label="v5 Compact")
    axes[1].plot(betas, ord_hx, 'x--', label="Ordered (Ref)")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel(r"$h_x$")
    axes[1].set_title(r"$h_x$ ($\theta={:.2f}$)".format(theta_val))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

if __name__ == "__main__":
    run_fast_scan(np.pi/2, 2.5, "hmf_cmp_pi2_light.png")
    run_fast_scan(0.25, 2.5, "hmf_cmp_generic_light.png")
