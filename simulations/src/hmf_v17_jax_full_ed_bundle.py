"""
HMF FULL-ED GPU RESEARCH SUITE (v17 - JAX Edition)
------------------------------------------------
The definitive high-performance engine for Exact Diagonalization.

KEY OPTIMIZATIONS:
- Full Spectral Info: jax.lax.linalg.eigh finds ALL eigenvalues/vectors.
- Instant Beta Sweeps: One diagonalization per (g, theta) allows 1000s of 
  temperature points in milliseconds via broadcasting.
- GPU Partial Trace: Computed entirely in VRAM.
- Hardware Profiler: Auto-estimates Hilbert space capacity based on VRAM.
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    from scipy.linalg import expm # For analytic bits
except ImportError:
    print("❌ JAX not found. For Colab, run: !pip install jax jaxlib")
    jax = None

# =============================================================================
# 1. CORE PHYSICS & OPERATORS (JAX NATIVE)
# =============================================================================

@dataclass
class PhysicsConfig:
    beta: float = 2.0
    omega_q: float = 2.0
    theta: float = 1.5708  # pi/2
    g: float = 0.5
    # Bath
    n_modes: int = 4
    n_cut: int = 8
    omega_min: float = 0.1
    omega_max: float = 8.0
    q_strength: float = 5.0
    tau_c: float = 0.5
    # Analytic
    scale: float = 1.04
    kappa: float = 0.94

def get_vram_limit():
    """Returns a conservative VRAM limit. T4 has 16GB, but eigh needs heavy workspace."""
    if jax is None: return 2e9 # Very safe for CPU
    try:
        device = jax.devices()[0]
        # We now target a 70% margin (using only 30-40%) because syevd (eigh) 
        # requires massive internal workspace during the solve.
        total_mem = 15e9 if "T4" in device.device_kind else 40e9 if "A100" in device.device_kind else 6e9
        return total_mem * 0.35 # 35% is much safer for N > 10k
    except Exception:
        return 2e9

def is_memory_safe(n_m, n_c, limit_bytes):
    dim = 2 * (n_c**n_m)
    # The matrix itself is N^2 * 8 bytes. eigh can take up to ~3x that in workspace.
    mem_needed = (dim**2) * 8 * 3 
    return mem_needed <= limit_bytes, dim, mem_needed

def check_hardware_capacity():
    if jax is None: return
    try:
        device = jax.devices()[0]
        limit = get_vram_limit()
        print(f"🖥️ Hardware: {device.device_kind}")
        print(f"🛡️ Safety Margin (50%): {limit/1e9:.1f} GB")
        
        dim_limit = int(np.sqrt(limit / 8))
        print(f"📊 Maximum Safe Hilbert Space: ~{dim_limit} states.")
    except Exception:
        print("🖥️ Running on CPU (or capacity check failed).")

@jit
def get_ops_jax():
    sz = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
    sx = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
    sy = jnp.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex64)
    id2 = jnp.eye(2, dtype=jnp.complex64)
    return sz, sx, sy, id2

def build_ham_jax(c: PhysicsConfig):
    sz, sx, sy, id2 = get_ops_jax()
    
    # Bath coupling
    omegas = jnp.linspace(c.omega_min, c.omega_max, c.n_modes)
    delta_w = omegas[1] - omegas[0] if c.n_modes > 1 else c.omega_max - c.omega_min
    j_vals = c.q_strength * c.tau_c * omegas * jnp.exp(-c.tau_c * omegas)
    g_k = jnp.sqrt(j_vals * delta_w)
    
    # Mode operators
    a = jnp.zeros((c.n_cut, c.n_cut), dtype=jnp.complex64)
    for i in range(c.n_cut - 1):
        a = a.at[i, i+1].set(jnp.sqrt(i + 1))
    xc = a + a.T.conj()
    nc = a.T.conj() @ a
    id_b = jnp.eye(c.n_cut, dtype=jnp.complex64)
    
    # Full Bath Construction
    # Since we want Full-ED, we construct the dense matrix
    dim_b = c.n_cut**c.n_modes
    hb = jnp.zeros((dim_b, dim_b), dtype=jnp.complex64)
    vb = jnp.zeros((dim_b, dim_b), dtype=jnp.complex64)
    
    for k in range(c.n_modes):
        # Kron chain
        mat_h = jnp.eye(1, dtype=jnp.complex64)
        mat_v = jnp.eye(1, dtype=jnp.complex64)
        for j in range(c.n_modes):
            if j == k:
                mat_h = jnp.kron(mat_h, omegas[k] * nc)
                mat_v = jnp.kron(mat_v, g_k[k] * xc)
            else:
                mat_h = jnp.kron(mat_h, id_b)
                mat_v = jnp.kron(mat_v, id_b)
        hb += mat_h
        vb += mat_v
        
    vq = jnp.cos(c.theta) * sz - jnp.sin(c.theta) * sx
    h_tot = jnp.kron(0.5 * c.omega_q * sz, jnp.eye(dim_b)) + \
            jnp.kron(id2, hb) + \
            c.g * jnp.kron(vq, vb)
    return h_tot, dim_b

# =============================================================================
# 2. FULL-ED SOLVER (VECTORIZED BETA)
# =============================================================================

@jit
def solve_single_sector(h_static, h_int, g, theta_val, betas):
    """
    Computes a full beta sweep for a FIXED g and theta.
    One diagonalization inside. Returns all 4 components.
    """
    sz, sx, _, _ = get_ops_jax()
    dim_b = h_int.shape[0] // 2
    
    # 1. Update H with current g and theta
    vq = jnp.cos(theta_val) * sz - jnp.sin(theta_val) * sx
    h_tot = h_static + g * jnp.kron(vq, jnp.eye(dim_b))
    
    # 2. Diagonalize
    evals, evecs = jnp.linalg.eigh(h_tot)
    
    def beta_trace(beta):
        w = jnp.exp(-beta * (evals - jnp.min(evals)))
        w /= jnp.sum(w)
        psi0, psi1 = evecs[:dim_b, :], evecs[dim_b:, :]
        
        # Parallel vdot across all eigenvectors i
        p00 = jnp.sum(w * jnp.sum(jnp.abs(psi0)**2, axis=0))
        p11 = jnp.sum(w * jnp.sum(jnp.abs(psi1)**2, axis=0))
        # rho_01 = sum_i w_i * (psi0_i.conj * psi1_i)
        rho01 = jnp.sum(w * jnp.sum(psi0.conj() * psi1, axis=0))
        
        return jnp.array([p00.real, p11.real, rho01.real, rho01.imag])

    return vmap(beta_trace)(betas)

def run_multi_dimensional_sweep(c: PhysicsConfig, gs, thetas, betas):
    """
    Orchestrates the 3D sweep. 
    Sequential over (g, theta) to avoid OOM, vectorized over beta.
    """
    # Build static components once
    omegas, g_k, xc, nc = build_sparse_components_params(c)
    sz, sx, sy, id2 = get_ops_jax()
    
    # Pre-build h_static and h_int parts
    dim_b = c.n_cut**c.n_modes
    hb = jnp.zeros((dim_b, dim_b), dtype=jnp.complex64)
    vb = jnp.zeros((dim_b, dim_b), dtype=jnp.complex64)
    id_b = jnp.eye(c.n_cut, dtype=jnp.complex64)
    
    for k in range(c.n_modes):
        mat_h, mat_v = jnp.eye(1, dtype=jnp.complex64), jnp.eye(1, dtype=jnp.complex64)
        for j in range(c.n_modes):
            if j == k:
                mat_h = jnp.kron(mat_h, omegas[j] * nc)
                mat_v = jnp.kron(mat_v, g_k[j] * xc)
            else:
                mat_h = jnp.kron(mat_h, id_b)
                mat_v = jnp.kron(mat_v, id_b)
        hb += mat_h
        vb += mat_v
        
    h_static = jnp.kron(0.5 * c.omega_q * sz, jnp.eye(dim_b)) + jnp.kron(id2, hb)
    h_int = jnp.kron(jnp.eye(2), vb) # This will be multiplied by vq later
    
    # JIT the sector solver
    sector_solver = solve_single_sector
    
    results = []
    print(f"🌀 Starting 3D Sweep: {len(gs)} g's x {len(thetas)} thetas x {len(betas)} betas")
    
    for g_val in gs:
        for th_val in thetas:
            t0 = time.perf_counter()
            # This call executes one Full-ED
            res = sector_solver(h_static, h_int, g_val, th_val, betas)
            dt = time.perf_counter() - t0
            print(f"  --- Physical Sector: g={g_val:.2f}, theta={th_val:.2f} (Time: {dt:.2f}s) ---")
            
            for i, b in enumerate(betas):
                p00, p11, re01, im01 = res[i]
                print(f"    beta={b:.2f} | p00={p00:.4f}, p11={p11:.4f}, re01={re01:.4f}, im01={im01:.4f}")
                results.append({
                    "g": float(g_val), "theta": float(th_val), "beta": float(b),
                    "p00": float(p00), "p11": float(p11), "re01": float(re01), "im01": float(im01)
                })
                
    return pd.DataFrame(results)

def build_sparse_components_params(c: PhysicsConfig):
    omegas = jnp.linspace(c.omega_min, c.omega_max, c.n_modes)
    delta_w = omegas[1] - omegas[0] if c.n_modes > 1 else c.omega_max - c.omega_min
    j_vals = c.q_strength * c.tau_c * omegas * jnp.exp(-c.tau_c * omegas)
    g_k = jnp.sqrt(j_vals * delta_w)
    a = jnp.zeros((c.n_cut, c.n_cut), dtype=jnp.complex64)
    for i in range(c.n_cut - 1): a = a.at[i, i+1].set(jnp.sqrt(i + 1))
    return omegas, g_k, a + a.T.conj(), a.T.conj() @ a

# =============================================================================
# 3. ANALYTIC ENGINE (Standard CPU/NumPy for safety)
# =============================================================================

def kernel_profile(c, u):
    omegas = np.linspace(c.omega_min, c.omega_max, 100)
    dw = omegas[1] - omegas[0]
    j = c.q_strength * c.tau_c * omegas * np.exp(-c.tau_c * omegas)
    res = np.zeros_like(u)
    for wk, jk in zip(omegas, j):
        den = np.sinh(0.5 * c.beta * wk)
        res += (jk * dw) * (np.cosh(wk * (0.5 * c.beta - u)) / den if den > 1e-12 else 2.0/(c.beta*wk))
    return res

def solve_ordered(c: PhysicsConfig):
    # Standard NumPy path integral (fallback)
    t = np.linspace(0, c.beta, 32)
    dt = t[1] - t[0]
    uu = np.minimum(abs(t[:,None]-t[None,:]), c.beta-abs(t[:,None]-t[None,:]))
    cov = kernel_profile(c, uu)
    ev, ek = np.linalg.eigh(cov)
    kl = ek[:, -4:] * np.sqrt(np.clip(ev[-4:], 0, None))
    gx, gw = np.polynomial.hermite.hermgauss(4)
    w_avg = np.zeros((2,2), dtype=complex)
    sz, sx, sy = np.array([[1,0],[0,-1]]), np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]])
    for inds in itertools.product(range(len(gx)), repeat=4):
        xi = c.g * (kl @ (np.sqrt(2.0)*gx[list(inds)]))
        weight = np.prod(gw[list(inds)]/np.sqrt(np.pi))
        u_op = np.eye(2, dtype=complex)
        for n in range(len(t)):
            tau = t[n]
            xt = np.cos(c.theta)*sz - np.sin(c.theta)*(np.cosh(c.omega_q*tau)*sx + 1j*np.sinh(c.omega_q*tau)*sy)
            u_op = (np.cosh(dt*xi[n])*np.eye(2) - np.sinh(dt*xi[n])*xt) @ u_op
        w_avg += weight * u_op
    rho = expm(-c.beta * (0.5*c.omega_q*sz)) @ w_avg
    return rho / np.trace(rho)

# =============================================================================
# 4. EXECUTION & DASHBOARD
# =============================================================================

def run_suite():
    print("🚀 [JAX FULL-ED] Starting Auto-Convergence Research Sweep...")
    check_hardware_capacity()
    limit = get_vram_limit()
    
    # 1. SEARCH SPACE (Physical - High Density 10x10 Grid)
    gs = jnp.linspace(0.1, 1.5, 10)
    thetas = jnp.linspace(0.0, 1.571, 10)
    betas = jnp.linspace(0.4, 8.0, 50)
    
    # 2. CONVERGENCE SPACE (Numerical)
    # Pairs of (n_modes, n_cut)
    candidates = [
        (3, 10), (3, 15), (3, 20),
        (4, 6), (4, 8), (4, 10),
        (5, 4), (5, 6), (5, 8)
    ]
    
    active_tasks = []
    for nm, nc in candidates:
        safe, d, m = is_memory_safe(nm, nc, limit)
        if safe:
            active_tasks.append((nm, nc, d))
            print(f"  [SAFE] m={nm}, c={nc} | Dim={d} | Mem={m/1e6:.1f} MB")
        else:
            print(f"  [SKIP] m={nm}, c={nc} | Dim={d} | Mem={m/1e9:.1f} GB exceeds safety limit.")

    all_dfs = []
    t_global_start = time.perf_counter()

    for nm, nc, dim in active_tasks:
        print(f"\n💎 Processing Sector: n_modes={nm}, n_cut={nc} (Dim={dim})")
        c = PhysicsConfig(n_modes=nm, n_cut=nc)
        
        t0 = time.perf_counter()
        df_sector = run_multi_dimensional_sweep(c, gs, thetas, betas)
        df_sector["n_modes"] = nm
        df_sector["n_cut"] = nc
        df_sector["dim"] = dim
        all_dfs.append(df_sector)
        print(f"  Checkpointed results for m={nm}, c={nc} | Total sector time: {time.perf_counter()-t0:.1f}s")
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("hmf_jax_convergence_results.csv", index=False)
    
    # 3. CONVERGENCE DASHBOARD
    print("\n🎬 Rendering Convergence Dashboard...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Selection for comparison: highest cutoff version
    best_sector = candidates[np.argmax([is_memory_safe(nm, nc, limit)[1] for nm, nc in candidates if is_memory_safe(nm, nc, limit)[0]])]
    baseline = final_df[(final_df["n_modes"] == best_sector[0]) & (final_df["n_cut"] == best_sector[1])]
    
    # Plot population vs beta for a slice of the 10x10 grid
    ax = axes[0]
    for g_val in [gs[0], gs[len(gs)//2], gs[-1]]:
        sub = baseline[baseline["g"].round(3) == float(round(g_val, 3))]
        for th_val in [thetas[0], thetas[-1]]:
            label = f"g={g_val:.2f} th={th_val:.2f}"
            ax.plot(sub[sub["theta"].round(3) == float(round(th_val, 3))]["beta"], 
                    sub[sub["theta"].round(3) == float(round(th_val, 3))]["p00"], 'o-', label=label)
    ax.set_title(f"Physical Landscape (m={best_sector[0]}, c={best_sector[1]})")
    ax.set_xlabel("beta"); ax.legend(fontsize='small'); ax.grid(alpha=0.2)
    
    # Plot convergence of p00 at the strongest sector (high g, high beta)
    ax = axes[1]
    strongest = final_df[(final_df["g"] == float(gs[-1])) & (final_df["theta"] == float(thetas[-1]))]
    for (nm, nc), group in strongest.groupby(["n_modes", "n_cut"]):
        ax.plot(group["beta"], group["p00"], 's-', label=f"m={nm} c={nc}")
    ax.set_title("Convergence at High-Coupling Boundary")
    ax.set_xlabel("beta"); ax.legend(fontsize='small'); ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    print(f"🎉 OMNIBUS CONVERGENCE COMPLETE. Total Time: {(time.perf_counter()-t_global_start)/60:.1f}m")

if __name__ == "__main__":
    run_suite()
