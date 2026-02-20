"""
Diagnostic Plot of HMF Kernels.

Visualizes the spectral density J(omega) and the imaginary-time kernel K(tau)
used in the HMF model comparison logic.
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Exact same logic as hmf_model_comparison_standalone.py
@dataclass
class BenchmarkConfig:
    beta: float
    omega_q: float
    theta: float
    n_modes: int
    n_cut: int
    omega_min: float
    omega_max: float
    q_strength: float
    tau_c: float

def _discrete_bath(config):
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)], dtype=float)
        delta_omega = float(config.omega_max - config.omega_min)
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes, dtype=float)
        delta_omega = float(omegas[1] - omegas[0])
    # J(w) = alpha * w * exp(-w * tau_c)
    j0 = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    return omegas, delta_omega, j0

def _kernel_profile(config, u_abs):
    beta = config.beta
    omegas, delta_omega, j0 = _discrete_bath(config)
    g2 = np.maximum(j0, 0.0) * delta_omega

    kernel = np.zeros_like(u_abs, dtype=float)
    for omega_k, g2_k in zip(omegas, g2):
        denom = np.sinh(0.5 * beta * omega_k)
        if abs(denom) < 1e-14:
            kernel += (2.0 * g2_k) / max(beta * omega_k, 1e-14)
        else:
            kernel += g2_k * np.cosh(omega_k * (0.5 * beta - u_abs)) / denom
    return kernel

def plot_kernels():
    # Standard parameters from comparison
    beta = 2.0
    config = BenchmarkConfig(
        beta=beta, omega_q=2.0, theta=np.pi/2,
        n_modes=50, n_cut=1, 
        omega_min=0.1, omega_max=10.0,
        q_strength=5.0, tau_c=0.5
    )
    
    # 1. Spectral Density
    omegas, dw, j0 = _discrete_bath(config)
    
    # 2. Kernel K(tau)
    taus = np.linspace(0, beta, 200)
    k_tau = _kernel_profile(config, taus)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # J(w)
    axes[0].stem(omegas, j0, basefmt=" ", markerfmt="o", label="Discrete Modes")
    # continuous curve for reference
    w_cont = np.linspace(0, 10, 100)
    j_cont = config.q_strength * config.tau_c * w_cont * np.exp(-config.tau_c * w_cont)
    axes[0].plot(w_cont, j_cont, 'r--', label=r"$J(\omega) = \alpha \omega e^{-\omega \tau_c}$")
    axes[0].set_xlabel(r"$\omega$")
    axes[0].set_ylabel(r"$J(\omega)$")
    axes[0].set_title("Spectral Density Discretization")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # K(tau)
    axes[1].plot(taus, k_tau, 'b-', linewidth=2, label=r"$K(\tau)$")
    axes[1].set_xlabel(r"$\tau$")
    axes[1].set_ylabel(r"$K(\tau)$")
    axes[1].set_title(r"Imaginary Time Kernel ($\beta={}$)".format(beta))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Response Functions (Integrands)
    omega_q = config.omega_q
    
    # v5 Integrands
    integrand_laplace = k_tau * np.exp(omega_q * taus)
    integrand_resonant = k_tau * (beta - taus) * np.exp(omega_q * taus)
    
    # Ordered/System Integrands (envelopes)
    # The system operator grows as cosh(omega_q * tau)
    envelope_system = np.cosh(omega_q * taus)
    integrand_ordered = k_tau * envelope_system
    
    # Normalized for comparison
    def norm(x): return x / np.max(np.abs(x))
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    ax2.plot(taus, norm(k_tau), 'k--', alpha=0.5, label=r"Kernel $K(\tau)$")
    ax2.plot(taus, norm(integrand_laplace), 'b-', label=r"v5 Laplace: $K(\tau) e^{\omega \tau}$")
    ax2.plot(taus, norm(integrand_resonant), 'r-', label=r"v5 Resonant: $K(\tau)(\beta-\tau)e^{\omega \tau}$")
    ax2.plot(taus, norm(integrand_ordered), 'g-.', label=r"Ordered: $K(\tau) \cosh(\omega \tau)$")
    
    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.set_title(r"Response Integrands ($\beta={}, \omega={}$)".format(beta, omega_q))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig("hmf_response_functions.png", dpi=150)
    print("Saved hmf_response_functions.png")

if __name__ == "__main__":
    plot_kernels()

