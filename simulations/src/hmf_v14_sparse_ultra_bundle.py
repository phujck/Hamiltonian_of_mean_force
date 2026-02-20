"""
Sparse Ultra-High Precision HMF Bundle (v14 - Colab Ready).
ONE-FILE VERSION: Uses scipy.sparse for large Hilbert spaces.

Method:
- Constructs Hamiltonian using scipy.sparse.kron.
- Solves for lowest N_EIG eigenvalues/vectors using scipy.sparse.linalg.eigsh.
- Reconstructs reduced density matrix from truncated thermal ensemble.
"""

import argparse
import time
import json
import itertools
import numpy as np
import pandas as pd
from scipy.sparse import kron, diags, eye, csr_matrix
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass
from typing import Any, Tuple, Dict, List

# =============================================================================
# OPERATORS & CONSTANTS
# =============================================================================
SX = csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = csr_matrix([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = csr_matrix([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = eye(2, dtype=complex, format='csr')

# =============================================================================
# DATA STRUCTURES
# =============================================================================
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

# =============================================================================
# SPARSE HMF LOGIC
# =============================================================================
def build_sparse_operators(config: BenchmarkConfig):
    n_m = config.n_modes
    n_c = config.n_cut
    
    # Bath frequencies and couplings
    omegas = np.linspace(config.omega_min, config.omega_max, n_m)
    delta_w = omegas[1] - omegas[0] if n_m > 1 else config.omega_max - config.omega_min
    j_vals = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    g_k = np.sqrt(j_vals * delta_w)
    
    # System Operators
    h_s = 0.5 * config.omega_q * SZ
    v_q = np.cos(config.theta) * SZ - np.sin(config.theta) * SX
    
    # Bath Operators (Kronecker sums)
    dim_bath = n_c**n_m
    
    # Single mode building blocks
    a = np.zeros((n_c, n_c), dtype=complex)
    for i in range(n_c - 1):
        a[i, i+1] = np.sqrt(i + 1)
    x_c = csr_matrix(a + a.T.conj())
    h_c = csr_matrix(a.T.conj() @ a)
    
    h_b_sparse = csr_matrix((dim_bath, dim_bath), dtype=complex)
    v_b_sparse = csr_matrix((dim_bath, dim_bath), dtype=complex)
    
    for k in range(n_m):
        pre_dim = n_c**k
        post_dim = n_c**(n_m - k - 1)
        
        # H_b = sum omega_k n_k
        h_mode = kron(eye(pre_dim), kron(omegas[k] * h_c, eye(post_dim)), format='csr')
        h_b_sparse += h_mode
        
        # V_b = sum g_k (a_k + a_k^dag)
        v_mode = kron(eye(pre_dim), kron(g_k[k] * x_c, eye(post_dim)), format='csr')
        v_b_sparse += v_mode
        
    h_static = kron(h_s, eye(dim_bath)) + kron(I2, h_b_sparse)
    h_int = kron(v_q, v_b_sparse)
    
    return h_static, h_int, dim_bath

def compute_reduced_density_sparse(h_tot, config, n_eig):
    dim_tot = h_tot.shape[0]
    n_eig = min(n_eig, dim_tot - 2)
    
    # Solve for lowest eigenvalues
    # Using 'SA' (Smallest Algebraic) for Hermitian matrices
    evals, evecs = eigsh(h_tot, k=n_eig, which='SA')
    
    # Boltzmann weights
    shift = np.min(evals)
    weights = np.exp(-config.beta * (evals - shift))
    z_sum = np.sum(weights)
    
    # Reduced density matrix construction via partial trace over eigenvectors
    # Tr_B(|psi><psi|) = 2x2 matrix
    # psi is 2*dim_bath vector. Split into psi_0 and psi_1.
    dim_bath = dim_tot // 2
    rho_q = np.zeros((2, 2), dtype=complex)
    
    for i in range(n_eig):
        w = weights[i] / z_sum
        psi = evecs[:, i]
        psi0 = psi[:dim_bath]
        psi1 = psi[dim_bath:]
        
        rho_q[0, 0] += w * np.vdot(psi0, psi0)
        rho_q[1, 1] += w * np.vdot(psi1, psi1)
        rho_q[0, 1] += w * np.vdot(psi0, psi1)
        rho_q[1, 0] += w * np.vdot(psi1, psi0)
        
    return rho_q

def run_sparse_sweep(args):
    out_dir = Path(".").resolve()
    scan_csv = out_dir / f"{args.output_prefix}_scan.csv"
    
    betas = np.linspace(args.beta_min, args.beta_max, args.beta_points)
    n_m_list = [int(x) for x in args.n_modes_list.split(",")]
    n_c_list = [int(x) for x in args.n_cut_list.split(",")]
    
    # High g and theta for the test
    G = 0.5
    THETA = np.pi / 2.0
    
    results = []
    print(f"[SPARSE-ULTRA] Starting Sweep. n_eig={args.n_eig}")
    
    for m in n_m_list:
        for c in n_c_list:
            # Build Hamiltonian once per (m, c)
            cfg_base = BenchmarkConfig(0.1, 2.0, THETA, m, c, 0.1, 8.0, 5.0, 0.5)
            h_stat, h_int, dim_b = build_sparse_operators(cfg_base)
            h_tot = h_stat + G * h_int
            
            for beta in betas:
                t0 = time.perf_counter()
                cfg = BenchmarkConfig(beta, 2.0, THETA, m, c, 0.1, 8.0, 5.0, 0.5)
                
                rho_q = compute_reduced_density_sparse(h_tot, cfg, args.n_eig)
                
                dt = time.perf_counter() - t0
                print(f"  m={m} c={c} b={beta:.2f} | p00={rho_q[0,0].real:.4f} | {dt:.1f}s")
                
                results.append({
                    "n_modes": m, "n_cut": c, "beta": beta,
                    "p00": rho_q[0,0].real, "p11": rho_q[1,1].real,
                    "re01": rho_q[0,1].real, "im01": rho_q[0,1].imag,
                    "elapsed_s": dt
                })
                
    pd.DataFrame(results).to_csv(scan_csv, index=False)
    print(f"[DONE] Saved to {scan_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes-list", type=str, default="5,6")
    parser.add_argument("--n-cut-list", type=str, default="10,12")
    parser.add_argument("--n-eig", type=int, default=50)
    parser.add_argument("--beta-min", type=float, default=1.0) # Truncation works best at low-mid T
    parser.add_argument("--beta-max", type=float, default=6.0)
    parser.add_argument("--beta-points", type=int, default=6)
    parser.add_argument("--output-prefix", type=str, default="hmf_sparse_ultra_results")
    args = parser.parse_args()
    run_sparse_sweep(args)

if __name__ == "__main__":
    main()
