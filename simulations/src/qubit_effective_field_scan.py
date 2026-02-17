
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prl127_qubit_benchmark import BenchmarkConfig
from prl127_qubit_analytic_bridge import compute_delta_coefficients

def main():
    omega_q = 3.0
    theta = 0.25 
    beta = 2.0
    lambdas = np.linspace(0.0, 4.0, 200) # Scan up to 4
    
    config = BenchmarkConfig(
        beta=beta,
        omega_q=omega_q,
        theta=theta,
        lambda_min=0, lambda_max=0, lambda_points=1,
        n_modes=2, n_cut=4, omega_min=0.5, omega_max=8.0, q_strength=10.0, tau_c=1.0,
        output_prefix="field_scan"
    )

    print(f"Computing integrals...")
    _, dx, dy, dz = compute_delta_coefficients(config, n_kernel_grid=4000)
    print(f"Integrals: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    
    hx_list, hy_list, hz_list = [], [], []
    beta_eff_list = []
    
    A = beta * omega_q / 2.0
    cA = np.cosh(A)
    sA = np.sinh(A)
    
    for lam in lambdas:
        lam2 = lam**2
        B_sq_arg = dx**2 - dy**2 + dz**2
        B_val = lam2 * np.sqrt(complex(B_sq_arg)) # Complex handle
        
        # Asymptotic / Stable decision
        # If real(B) is large, use asymptotic
        if np.abs(B_val) > 20.0:
            # Asymptotic h ~ B
            # hx = lam2 * (dx cA + dy sA)
            # hy = lam2 * (dy cA + dx sA)
            # hz = lam2 * dz cA - B sA
            # N = 2 B exp(-B) (approx)
            
            hx_val = lam2 * (dx * cA + dy * sA)
            hy_val = lam2 * (dy * cA + dx * sA)
            # Use real part of B for hz term if B is real?
            # B is effectively real here (dx>dy checked).
            hz_val = lam2 * dz * cA - B_val * sA
            
            # Recalculate N for plot
            if np.real(B_val) > 50:
                 N_val = 0.0 # Underflow safe
            else:
                 # h approx B. N = B/sinh(B) approx 2B exp(-B)
                 N_val = 2.0 * np.abs(B_val) * np.exp(-np.abs(B_val))

        else:
            # Exact stable
            if abs(B_val) < 1e-9:
                sB_over_B = 1.0
                cB = 1.0
            else:
                sB_over_B = np.sinh(B_val) / B_val
                cB = np.cosh(B_val)
                
            Lambda = lam2 * sB_over_B
            Vx = Lambda * (dx * cA + dy * sA)
            iVy_mag = Lambda * (dy * cA + dx * sA)
            Vz = Lambda * dz * cA - cB * sA
            
            V_sq = Vx**2 - iVy_mag**2 + Vz**2
            V_norm = np.sqrt(V_sq)
            
            h_val = np.arcsinh(V_norm)
            if abs(h_val) < 1e-9:
                N_val = 1.0
            else:
                N_val = h_val / np.sinh(h_val)
                
            hx_val = N_val * Vx
            hy_val = N_val * iVy_mag
            hz_val = N_val * Vz

        hx_list.append(np.real(hx_val))
        hy_list.append(np.real(hy_val))
        hz_list.append(np.real(hz_val))
        beta_eff_list.append(np.real(beta * N_val))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.plot(lambdas, hx_list, label=r'$h_x$')
    ax1.plot(lambdas, hy_list, label=r'$|h_y|$', linestyle='--')
    ax1.plot(lambdas, hz_list, label=r'$h_z$', linestyle='-.')
    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel('Effective Field Component')
    ax1.set_title('Effective Hamiltonian vs Coupling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(lambdas, beta_eff_list, color='purple', linewidth=2)
    ax2.set_xlabel(r'$\lambda$')
    ax2.set_ylabel(r'$\beta_{eff} = \beta \mathcal{N}$')
    ax2.set_title(r'Effective Inverse Temperature')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(__file__).parents[2] / "manuscript" / "figures" / "hmf_qubit_effective_field_vs_lambda.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
