"""
Validation of the simplified HMF v4 result against Exact Diagonalization.
Compares the state fidelity and the Hamiltonian components (h_x, h_z) 
extracted from the exact equilibrium state.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import logm, expm

from prl127_qubit_benchmark import (
    BenchmarkConfig,
    SIGMA_X,
    SIGMA_Z,
    IDENTITY_2,
    build_static_operators,
    thermal_state,
    partial_trace_bath,
    trace_distance,
    bare_gibbs_state
)

SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

def analytic_hmf_v4(config: BenchmarkConfig, lam_coupling: float, q_reorg_integral: float) -> np.ndarray:
    """
    Computes H_MF using the analytic result from results_v4.tex.
    
    H_MF = h_x sigma_x + h_z sigma_z
    
    where:
      lambda = q_reorg_integral / pi  (from mapping Int J/w = pi * lambda)
      K_eff = (4 * lambda / omega_q) * sinh^2(beta * omega_q / 2)
      s_val = sin(theta) * lam_coupling  <-- CAREFUL with coupling definition
      c_val = cos(theta) * lam_coupling? No, theta is mixing.
      
    Wait, let's check the coupling def.
    H_I = lambda * X * B.
    f = c sigma_z - s sigma_x.
    The manuscript assumes H_I has strength included in f? Or is lambda separate?
    In results_v4.tex, theta determines the operator structure.
    The coupling strength g corresponds to lambda in the code.
    Reorganization energy E_R in paper is lambda.
    
    Let's re-read carefully:
    "The coupling strength g enters exclusively through the reorganization energy lambda propto g^2".
    
    In the code:
    H_tot = H_S + H_B + lambda_code * X * B + lambda_code^2 * Q * X^2
    
    The spectral density J(w) scales the B operators.
    The effective reorganization energy of the *coupled* system is:
    E_R_eff = (lambda_code)^2 * (Integral J/w / pi).
    So lambda_paper = (lambda_code^2) * ("Q_reorg" / pi).
    
    Also check mixing angle:
    X = cos(theta) sigma_z - sin(theta) sigma_x.
    Paper: coupling operator f = c sigma_z - s sigma_x.
    So c = cos(theta), s = sin(theta). (Matches).
    
    So:
    lambda_paper = (lam_coupling**2) * (q_reorg_integral / np.pi)
    """
    
    beta = config.beta
    omega_q = config.omega_q
    theta = config.theta
    
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Calculate paper's lambda (reorganization energy)
    # The code's q_reorg is Int J/w dw.
    # The paper's lambda is (1/pi) * Int J_eff/w, where J_eff includes the coupling scaling.
    # J_eff(w) = lambda_code^2 * J(w).
    lambda_paper = (lam_coupling**2) * (q_reorg_integral / np.pi)
    
    # Effective Kernel K(beta)
    # Eq: K(beta) = (4 * lambda / omega_q) * sinh^2(beta * omega_q / 2)
    # Check limit omega_q -> 0? No, assumes omega_q > 0.
    kernel_eff = (4.0 * lambda_paper / omega_q) * (np.sinh(beta * omega_q / 2.0)**2)
    
    # Argument sK
    # Note: in paper 's' was sin(theta).
    # "sK" usually meant s * K.
    sk_val = s * kernel_eff
    
    # Omega definition
    # cos(Omega) = cos(beta w_q / 2) cos(sK) + s sin(beta w_q / 2) sin(sK)
    # We need to solve for Omega. 
    # Use arccos, but carefully check branches?
    # Usually Omega starts near beta w_q / 2 and evolves.
    # Given the periodicity and nature, arccos should be fine for principal value 
    # if we match the sin(Omega) sign.
    
    phi = beta * omega_q / 2.0
    cos_omega = np.cos(phi) * np.cos(sk_val) + s * np.sin(phi) * np.sin(sk_val)
    
    # We need sin(Omega) for the denominators.
    # sin^2 = 1 - cos^2.
    # Sign of sin(Omega)?
    # Limit lambda -> 0 => sk -> 0 => cos_omega -> cos(phi). Omega -> phi.
    # So sin(Omega) should have same sign as sin(phi) approx?
    # Actually, we can just compute Omega = arccos(cos_omega).
    # But wait, simplified formulas use Omega/sin(Omega).
    # Let's compute h_x, h_z directly using the expressions.
    # h_x = c * (Omega / sin(Omega)) * sin(sK)
    # h_z = ...
    
    # If sin(Omega) is small, we have limits.
    
    omega_val = np.arccos(cos_omega)
    # Check consistency: does this Omega produce positive sin?
    sin_omega = np.sin(omega_val)
    
    # The derivation likely assumed Omega > 0.
    
    if abs(sin_omega) < 1e-9:
        # Limit sin(Omega) -> 0.
        # Ratio Omega/sin(Omega) -> 1? No.
        # If Omega -> 0, ratio -> 1.
        # If Omega -> pi, ratio -> singularity.
        prefactor = 1.0 # Placeholder
        if abs(omega_val) < 1e-9:
            prefactor = 1.0
        else:
             # This case shouldn't happen deep in the benchmark usually unless beta*wq = 2pi
             prefactor = omega_val / sin_omega
    else:
        prefactor = omega_val / sin_omega
        
    term_x = c * np.sin(sk_val)
    term_z = -np.sin(phi) * np.cos(sk_val) - s * np.cos(phi) * np.sin(sk_val)
    
    h_x = (1.0/beta) * prefactor * term_x
    h_z = (1.0/beta) * prefactor * term_z # Note the minus sign in H = -1/beta * nu * sigma
    
    # Wait, H_MF = -1/beta * vec(nu) * sigma
    # nu_x = prefactor * term_x
    # nu_z = prefactor * term_z
    # So h_x = -nu_x / beta.
    # Let's check signs in results_v4.tex
    # h_x = c * (Omega/sin) * sin(sK)?? 
    # Eq 158: nu'_x = c * (Omega/sin) * sin(sK).
    # Eq 154: H_MF = h_x sigma_x + h_z sigma_z.
    # Eq 182 (old): h_j = - nu_j / beta.
    # So h_x = - (1/beta) * c * (Omega/sin) * sin(sK).
    
    h_x = -(1.0/beta) * prefactor * term_x
    h_z = -(1.0/beta) * prefactor * term_z
    
    # Construct Hamiltonian
    H_mf = h_x * SIGMA_X + h_z * SIGMA_Z
    
    # Compute state
    # rho = exp(-beta H) / Z
    rho = expm(-beta * H_mf)
    rho /= np.trace(rho)
    
    return rho, h_x, h_z


def run_validation(config: BenchmarkConfig):
    print(f"Running validation for beta={config.beta}, w_q={config.omega_q}, theta={config.theta}")
    
    h_static, h_int, h_counter, h_s, x_op = build_static_operators(config)
    # q_reorg is calculated below

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
    
    for lam in lambdas:
        # Exact Calculation
        h_tot = h_static + lam * h_int + (lam**2) * h_counter
        rho_tot = thermal_state(h_tot, config.beta)
        rho_ex = partial_trace_bath(rho_tot, dim_system, dim_bath)
        
        # Analytic HMF v4
        rho_v4, hx_v4, hz_v4 = analytic_hmf_v4(config, lam, q_reorg)
        
        # Distance
        dist_v4 = trace_distance(rho_ex, rho_v4)
        
        # Extract numerical fields
        # H_eff = -1/beta log(rho_ex)
        # We need to remove the trace/constant part.
        log_rho = logm(rho_ex + 1e-12 * IDENTITY_2) # Add epsilon to avoid log(0) if pure
        h_eff_num = (-1.0/config.beta) * log_rho
        
        # Project onto Paulis
        # h_x = Tr(H sigma_x) / 2
        hx_num = np.real(np.trace(h_eff_num @ SIGMA_X)) / 2.0
        hy_num = np.real(np.trace(h_eff_num @ SIGMA_Y)) / 2.0
        hz_num = np.real(np.trace(h_eff_num @ SIGMA_Z)) / 2.0
        
        # Rotate numerical fields to the "clean" frame
        # We rotate the numeric frame by U0 = exp(-i phi/2 sigma_z) so that h_y vanishes?
        # The analytical result assumes we rotated the frame.
        # The numerical result is in the LAB frame.
        # So analytical prediction for LAB frame h_x, h_y would be:
        # h_x_lab = h_x_prime * cos(phi) - h_y_prime * sin(phi) ... ?
        # Actually, let's rotate the numerical fields back to the primed frame.
        # U0 = exp(-i phi/2 sigma_z)
        # phi = beta * omega_q / 2
        # H' = U0 H U0dag.
        # This rotates vectors about Z by angle phi.
        # vec_new = R_z(phi) vec_old.
        # x' = x cos + y sin
        # y' = -x sin + y cos (should be 0)
        
        phi = config.beta * config.omega_q / 2.0
        hx_rot = hx_num * np.cos(phi) + hy_num * np.sin(phi)
        hy_rot = -hx_num * np.sin(phi) + hy_num * np.cos(phi)
        hz_rot = hz_num 
        
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
    return df

if __name__ == "__main__":
    # Standard Config
    conf = BenchmarkConfig(
        beta=1.0, 
        omega_q=3.0, 
        theta=0.25, 
        lambda_max=4.0, 
        n_modes=4,   # Reduced from 5 to prevent OOM (Dim: 2592 vs 15552)
        n_cut=6,     # Sufficient cutoff
        omega_max=10.0
    )
    
    df = run_validation(conf)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Distance
    axes[0].semilogy(df["lambda"], df["dist_v4"], label="Trace Distance")
    axes[0].set_xlabel("Lambda")
    axes[0].set_ylabel("Distance")
    axes[0].set_title("Fidelity of HMF v4")
    axes[0].legend()
    
    # Fields (Rotated)
    axes[1].plot(df["lambda"], df["hx_v4"], 'k-', label="Analytic h_x")
    axes[1].plot(df["lambda"], df["hx_num_rot"], 'r--', label="Numeric h_x (Rot)")
    axes[1].plot(df["lambda"], df["hz_v4"], 'b-', label="Analytic h_z")
    axes[1].plot(df["lambda"], df["hz_num"], 'g--', label="Numeric h_z")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Field Strength")
    axes[1].set_title("Effective Fields (Rotated Frame)")
    axes[1].legend()

    # Check y-component removal
    axes[2].plot(df["lambda"], df["hy_num_rot"], label="Numeric h_y (Rot)")
    axes[2].plot(df["lambda"], df["hy_num_lab"], alpha=0.3, label="Numeric h_y (Lab)")
    axes[2].set_xlabel("Lambda")
    axes[2].set_title("Resulting h_y (Should be ~0)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("hmf_v4_validation.png")
    print("Validation saved to hmf_v4_validation.png")
