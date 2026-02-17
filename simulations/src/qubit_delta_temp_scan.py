
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import replace
import sys
import os

# Add local directory to path regarding of where it is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prl127_qubit_benchmark import BenchmarkConfig
from prl127_qubit_analytic_bridge import compute_delta_coefficients

def main():
    # Parameters matching the default bridge config
    omega_q = 3.0
    theta = 0.25
    
    # Temperature scan
    temps = np.linspace(0.1, 5.0, 50)
    betas = 1.0 / temps
    
    delta_x_list = []
    delta_y_list = []
    delta_z_list = []
    
    # Use standard config but vary beta
    base_config = BenchmarkConfig(
        beta=1.0, # placeholder
        omega_q=omega_q,
        theta=theta,
        lambda_min=0, lambda_max=0, lambda_points=1, # unused
        n_modes=2, n_cut=4, omega_min=0.5, omega_max=8.0, q_strength=10.0, tau_c=1.0, # bath params
        output_prefix="temp_scan"
    )

    print(f"Scanning {len(temps)} temperatures from {temps[0]} to {temps[-1]}...")
    
    for beta in betas:
        config = replace(base_config, beta=beta)
        # alpha0, delta_x, delta_y, delta_z
        _, dx, dy, dz = compute_delta_coefficients(config, n_kernel_grid=2000)
        delta_x_list.append(dx)
        delta_y_list.append(dy)
        delta_z_list.append(dz)
        
    delta_x = np.array(delta_x_list)
    delta_y = np.array(delta_y_list)
    delta_z = np.array(delta_z_list)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    ax.semilogy(temps, np.abs(delta_x), label=r'$|\delta_x|$ (Transverse Coherence)', linewidth=2)
    ax.semilogy(temps, np.abs(delta_y), label=r'$|\delta_y|$ (Imaginary Phase)', linewidth=2, linestyle='--')
    ax.semilogy(temps, np.abs(delta_z), label=r'$|\delta_z|$ (Population Shift)', linewidth=2, linestyle='-.')
    
    ax.set_xlabel(r'Temperature $k_B T / \hbar \omega_q$') 
    ax.set_ylabel(r'Coefficient Amplitude (Log Scale)')
    ax.set_title(r'Influence Coefficients vs Temperature ($\theta=0.25$)')
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    # Save to manuscript figures
    project_root = Path(__file__).resolve().parents[2]
    ms_fig_dir = project_root / "manuscript" / "figures"
    ms_fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = ms_fig_dir / "hmf_qubit_deltas_vs_temp.png"
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    main()
