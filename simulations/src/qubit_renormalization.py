
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sys
from pathlib import Path

# Add the src directory to the path to allow imports
sys.path.append(str(Path(__file__).parent))

from prl127_qubit_benchmark import (
    BenchmarkConfig, 
    build_static_operators, 
    thermal_state, 
    partial_trace_bath,
    SIGMA_Z, 
    SIGMA_X
)

def calculate_theory_renormalization(config: BenchmarkConfig, coupling_lambda: float) -> float:
    """
    Calculates the theoretical renormalized frequency omega_MF using the formulas from Section 4.
    
    H_MF = (omega_MF / 2) * sigma_z
    omega_MF = omega - Xi_cs / beta
    
    Xi_cs = (1/beta) * sum_ell kappa_ell * Re[B_ell * C_ell^*]
    """
    beta = config.beta
    omega = config.omega_q
    
    # 1. Reconstruct the discrete spectral density used in the exact simulation
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)])
        delta_omega = config.omega_max - config.omega_min
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes)
        delta_omega = omegas[1] - omegas[0]
        
    j_vals = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    # The benchmark scales the INTERACTION term by lambda, effectively scaling g_k by lambda
    # g_k = lambda * sqrt(J * delta)
    # kappa_ell scales as g_k^2, so as lambda^2
    
    g_vals_squared = j_vals * delta_omega * (coupling_lambda**2)
    
    # 2. Calculate kappa_ell for a range of Matsubara frequencies
    # We need enough Matsubara modes to converge the sum over ell
    n_matsubara = 1000 
    indices = np.arange(-n_matsubara, n_matsubara + 1)
    nu_ell = 2 * np.pi * indices / beta
    
    # kappa_ell = sum_k (2 * omega_k * g_k^2) / (omega_k^2 + nu_ell^2)
    kappa_ell = np.zeros_like(nu_ell)
    for k in range(len(omegas)):
        w_k = omegas[k]
        g2_k = g_vals_squared[k]
        term = (2 * w_k * g2_k) / (w_k**2 + nu_ell**2)
        kappa_ell += term
        
    # 3. Calculate B_ell and C_ell
    # From Section 4 Results:
    # B_ell = 0.5 * ( (e^(beta*omega)-1)/(omega + i*nu) + (1-e^(-beta*omega))/(omega - i*nu) )
    # C_ell = 0.5 * ( (e^(beta*omega)-1)/(omega + i*nu) - (1-e^(-beta*omega))/(omega - i*nu) )
    
    exp_plus = np.exp(beta * omega)
    exp_minus = np.exp(-beta * omega)
    
    term1 = (exp_plus - 1) / (omega + 1j * nu_ell)
    term2 = (1 - exp_minus) / (omega - 1j * nu_ell)
    
    B_ell = 0.5 * (term1 + term2)
    C_ell = 0.5 * (term1 - term2)
    
    # 4. Calculate Xi_cs and omega_MF
    # delta_z = (1/2) * Xi_cs * |n_perp|^2 * h_hat
    # Here |n_perp| = 1 (transverse coupling lambda * sigma_x)
    # So delta = (1/2) * Xi_cs
    # omega_MF = omega - 2 * delta / beta = omega - Xi_cs / beta
    
    # Xi_cs = (1/beta) * sum_ell kappa_ell * Re[B C*]
    sumand = kappa_ell * np.real(B_ell * np.conj(C_ell))
    Xi_cs = (1.0 / beta) * np.sum(sumand)
    
    # Theoretical prediction:
    # omega_MF = omega - Xi_cs
    # Wait, check the scaling. 
    # Eq 111: omega_MF = omega - 2 gamma / beta
    # Eq 80: gamma = ...
    # Eq 86: delta = 1/2 Xi_cs ... 
    # The detailed derivation says:
    # omega_MF = omega - (1/beta) * Xi_cs  (if my derivation in thought process was correct)
    # Let's verify derived result in thought process:
    # z = x + y = -beta*omega/2 + delta_z
    # -beta*omega_MF/2 = z
    # -beta*omega_MF/2 = -beta*omega/2 + delta_z
    # omega_MF = omega - 2*delta_z / beta
    # delta_z = 0.5 * Xi_cs
    # omega_MF = omega - Xi_cs / beta
    
    return omega - Xi_cs / beta

def calculate_exact_renormalization(config: BenchmarkConfig, coupling_lambda: float) -> float:
    # We must ensure the Exact calculation uses the SAME parameters as Theory.
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)
    
    # Construct total Hamiltonian
    # H = H_0 + lambda * H_int + lambda^2 * H_counter
    # Note: build_static_operators puts the counter term in h_counter already scaled by nothing, 
    # but the usual code scales it by lambda^2.
    # We keep the counter term to ensure stability, or remove it?
    # The THEORY did NOT include a counter term.
    # If we include it in Exact, we simply shift the global energy, 
    # but since H_counter = lambda^2 * Q * X^2 and X^2 = I for transverse qubit,
    # H_counter is PROPORTIONAL TO IDENTITY.
    # So it shifts eigenvalues but NOT the splitting.
    # So we can keep it or leave it, it won't affect omega_MF.
    
    h_tot = h_static + coupling_lambda * h_int + (coupling_lambda**2) * h_counter
    
    # Exact reduced state
    dim_system = 2
    dim_bath = h_static.shape[0] // dim_system
    rho_tot = thermal_state(h_tot, config.beta)
    rho_s = partial_trace_bath(rho_tot, dim_system, dim_bath)
    
    # Diagonalize rho_s
    evals = eigh(rho_s, eigvals_only=True)
    # evals are probabilities p1, p2. p1 < p2 usually if T is low and ground state favored.
    # H_MF = -1/beta log rho_S
    # E_1 = -1/beta log p1
    # E_2 = -1/beta log p2
    # omega_MF = E_1 - E_2 = (1/beta) * (log p2 - log p1)
    
    # p_ground should be larger. So log p_ground is less negative. E_ground is lower.
    # rho = exp(-beta H_MF) / Z
    # p_ground = exp(-beta E_ground) / Z
    # p_excited = exp(-beta E_excited) / Z
    # p_ground/p_excited = exp(beta (E_excited - E_ground)) = exp(beta * omega_MF)
    # omega_MF = (1/beta) log(p_max / p_min)
    
    p_min = np.min(evals)
    p_max = np.max(evals)
    
    if p_min < 1e-15:
        return 0.0 # Numerical instability or T=0
        
    return (1.0 / config.beta) * np.log(p_max / p_min)

def run_simulation():
    # Configure for transverse coupling
    # theta=pi/2 => z-component of x_op is 0. x-component is -1.
    config = BenchmarkConfig(
        beta=1.0,
        omega_q=2.0,
        theta=np.pi/2, 
        n_modes=5,   # Small number for fast exact diag
        n_cut=6,     # Sufficient cutoff
        q_strength=1.0, # Base strength
        lambda_max=2.0,
        lambda_points=10
    )
    
    lambdas = np.linspace(0, config.lambda_max, config.lambda_points)
    exact_omegas = []
    theory_omegas = []
    
    print(f"Running renormalization check with {config.n_modes} modes...")
    
    for lam in lambdas:
        w_exact = calculate_exact_renormalization(config, lam)
        w_theory = calculate_theory_renormalization(config, lam)
        exact_omegas.append(w_exact)
        theory_omegas.append(w_theory)
        print(f"Lambda={lam:.2f} | Exact={w_exact:.4f} | Theory={w_theory:.4f}")
        
    # Plotting
    project_root = Path(__file__).resolve().parents[2]
    fig_dir = project_root / "simulations" / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "qubit_renormalization_check.png"
    
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, exact_omegas, 'o', label='Exact (ED)')
    plt.plot(lambdas, theory_omegas, '-', label='Theory (Section 4)')
    plt.xlabel('Coupling strength $\lambda$')
    plt.ylabel(r'Renormalized Frequency $\omega_{\mathrm{MF}}$')
    plt.title(f'Qubit Renormalization Check (N={config.n_modes} modes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path)
    print(f"Plot saved to {fig_path}")

if __name__ == "__main__":
    run_simulation()
