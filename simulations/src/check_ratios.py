
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import sys
import os
from pathlib import Path

# Add local directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prl127_qubit_benchmark import BenchmarkConfig
from prl127_qubit_analytic_bridge import compute_delta_coefficients

def main():
    omega_q = 3.0
    theta = 0.25
    
    # Low temp scan (High beta)
    temps = np.linspace(0.05, 1.0, 50)
    betas = 1.0 / temps
    
    ratios_zx = []
    ratios_yx = []
    
    base_config = BenchmarkConfig(
        beta=1.0,
        omega_q=omega_q,
        theta=theta,
        lambda_min=0, lambda_max=0, lambda_points=1,
        n_modes=2, n_cut=4, omega_min=0.5, omega_max=8.0, q_strength=10.0, tau_c=1.0,
        output_prefix="ratio_scan"
    )

    print(f"Scanning {len(temps)} temperatures...")
    
    for beta in betas:
        config = replace(base_config, beta=beta)
        _, dx, dy, dz = compute_delta_coefficients(config, n_kernel_grid=4000)
        ratios_zx.append(dz / dx)
        ratios_yx.append(dy / dx)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.plot(temps, ratios_zx, label=r'$\delta_z / \delta_x$', linewidth=2)
    ax.plot(temps, ratios_yx, label=r'$\delta_y / \delta_x$', linewidth=2, linestyle='--')
    
    # Targeted limit values?
    # X = cos(theta)Z - sin(theta)X
    # If Delta ~ X^2, that's isotropic/scalar? No Pauli products.
    # If Delta ~ X on the diagonal? 
    # Let's just see the values.
    
    exact_ratio_guess = -np.tan(theta) 
    # Delta ~ c * X would imply Z/X = cos/-sin = -cot(theta). 
    # Check what we get.
    
    ax.set_xlabel(r'Temperature $k_B T / \hbar \omega_q$')
    ax.set_ylabel(r'Ratio')
    ax.set_title(r'Coefficient Ratios vs Low Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"Low T Ratio z/x: {ratios_zx[0]}")
    print(f"Low T Ratio y/x: {ratios_yx[0]}")
    print(f"theta={theta}, tan(theta)={np.tan(theta)}, sin^2/sin2={np.sin(theta)**2/np.sin(2*theta)}")
    
    project_root = Path(__file__).resolve().parents[2]
    ms_fig_dir = project_root / "manuscript" / "figures"
    ms_fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = ms_fig_dir / "hmf_qubit_ratios_vs_temp.png"
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

if __name__ == "__main__":
    main()
