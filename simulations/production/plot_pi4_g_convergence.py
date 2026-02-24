
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set style
mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.usetex": True, "figure.dpi": 200, "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
})

# Path setup
PROD_DIR = Path(__file__).parent
SRC_DIR = PROD_DIR.parent / "src"
sys.path.append(str(SRC_DIR))

# Import HMF logic
try:
    from hmf_model_comparison_standalone_codex_v1 import (
        BenchmarkConfig, RenormConfig, laplace_k0, resonant_r0
    )
except ImportError:
    print("Warning: Could not import hmf_model_comparison_standalone_codex_v1 from src.")
    # Fallback to local definitions if needed, but let's assume it works.

def calculate_hmf_p11(g, beta, omega_q, theta, config, renorm):
    # Calculate raw continuum channels
    w = omega_q
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Continuum kernels
    k0_0 = laplace_k0(config, 0.0, 2001)
    k0_p = laplace_k0(config, w, 2001)
    k0_m = laplace_k0(config, -w, 2001)
    r_p = resonant_r0(config, w, 2001)
    r_m = resonant_r0(config, -w, 2001)

    sp0 = (c * s / w) * ((1.0 + np.exp(beta * w)) * k0_0 - 2.0 * k0_p)
    sm0 = (c * s / w) * ((1.0 + np.exp(-beta * w)) * k0_0 - 2.0 * k0_m)
    dz0 = (s**2) * 0.5 * (r_p - r_m)
    
    g2 = g*g
    sigma_plus = g2 * sp0
    sigma_minus = g2 * sm0
    delta_z = g2 * dz0
    
    chi_raw = np.sqrt(max(delta_z**2 + sigma_plus*sigma_minus, 0.0))
    
    # Apply running renormalization
    a = 0.5 * beta * omega_q
    chi_cap = max(renorm.kappa * abs(a), renorm.eps)
    run = 1.0 / (1.0 + chi_raw / chi_cap)
    
    chi_eff = run * chi_raw
    dz_eff = run * delta_z
    gamma = np.tanh(chi_eff) / chi_eff if chi_eff > 1e-12 else 1.0
    u = gamma * dz_eff
    
    zq = 2.0 * (np.cosh(a) - u * np.sinh(a))
    p11 = np.exp(a) * (1.0 - u) / zq
    return p11

# Data loading
DATA_FILE = PROD_DIR / "data" / "pi4_ncut_convergence.csv"
df = pd.read_csv(DATA_FILE)

# Select a representative beta
# We saw beta=1.972... has full counts
TARGET_BETA = df["beta"].iloc[np.argmin(np.abs(df["beta"] - 1.972))]
print(f"Plotting for beta = {TARGET_BETA:.3f}")

sub_beta = df[np.isclose(df['beta'], TARGET_BETA)].copy()

fig, ax = plt.subplots(figsize=(5.5, 4))

# Selection of lines to plot
lines = [
    (2, 6, "n=2, c=6 (Low)", "#e41a1c"),
    (2, 20, "n=2, c=20", "#377eb8"),
    (2, 60, "n=2, c=60", "#4daf4a"),
    (4, 4, "n=4, c=4", "#984ea3"),
    (4, 6, "n=4, c=6", "#ff7f00"),
]

groups = sub_beta.groupby(['n_modes', 'n_cut'])

for n_m, n_c, label, col in lines:
    if (n_m, n_c) in groups.groups:
        sub = groups.get_group((n_m, n_c)).sort_values("g")
        ax.plot(sub["g"], sub["ed_p11"], label=f"ED ({label})", color=col, marker='o', markersize=3, alpha=0.7)

# Theoretical Prediction Calculation
# Parameters from CSV
OMEGA_Q = 2.0
THETA = np.pi/4
Q_STRENGTH = 10.0
TAU_C = 1.0

# Setup continuum config
# For the continuum theory, we use large n_modes to approximate the kernel correctly
# but technically laplace_k0 in the imported file uses the discrete bath defined in the config.
# To get the "True Continuum", we should use a very large n_modes in the config.
theory_cfg = BenchmarkConfig(
    beta=TARGET_BETA, omega_q=OMEGA_Q, theta=THETA,
    n_modes=1000, n_cut=1, omega_min=0.01, omega_max=15.0,
    q_strength=Q_STRENGTH, tau_c=TAU_C
)
renorm = RenormConfig(scale=1.04, kappa=0.94)

g_range = np.linspace(0.01, 1.8, 50)
theory_p11 = [calculate_hmf_p11(g, TARGET_BETA, OMEGA_Q, THETA, theory_cfg, renorm) for g in g_range]

ax.plot(g_range, theory_p11, 'k--', lw=2.0, label="HMF Theory (Running)", zorder=10)

# Also plot raw HMF theory (no running) for comparison?
renorm_none = RenormConfig(scale=1.0, kappa=1e10) # Effectivly disables running
theory_p11_raw = [calculate_hmf_p11(g, TARGET_BETA, OMEGA_Q, THETA, theory_cfg, renorm_none) for g in g_range]
ax.plot(g_range, theory_p11_raw, 'gray', ls=':', lw=1.2, label="HMF Theory (Raw)", zorder=1)

ax.set_xlabel(r"Coupling Strength $g$")
ax.set_ylabel(r"Excited State Population $p_{11}$")
ax.set_title(rf"Coupling Sweep Convergence ($\beta \approx {TARGET_BETA:.1f}, \theta=\pi/4$)")
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(alpha=0.3, ls=":")

plt.tight_layout()
plt.savefig("../../manuscript/figures/hmf_g_convergence.png", dpi=300)
plt.savefig("../../manuscript/figures/hmf_g_convergence.pdf")
print(f"Generated coupling convergence plot for beta={TARGET_BETA:.3f}")
