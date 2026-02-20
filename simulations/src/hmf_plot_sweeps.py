import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
csv_path = "c:/Users/gerar/VScodeProjects/Hamiltonian_of_mean_force/simulations/src/hmf_jax_convergence_results.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit()

df = pd.read_csv(csv_path)

# Rounding for clean alignment
df['g'] = df['g'].round(3)
df['theta'] = df['theta'].round(3)
df['beta'] = df['beta'].round(2)

# Global physical anchors for convergence study
max_g = df['g'].max()
max_th = df['theta'].max()
max_beta = df['beta'].max()

print(f"Rendering HEAVY Convergence Dashboard (Focus: g={max_g}, theta={max_th}, beta={max_beta})")

fig, axes = plt.subplots(3, 2, figsize=(18, 20))
sectors = df.groupby(['n_modes', 'n_cut'])

# --- ROW 1: TEMPERATURE CONVERGENCE (vs Beta) ---
# Left: p00, Right: re01
for (nm, nc), group in sectors:
    sub = group[(group['g'] == max_g) & (group['theta'] == max_th)].sort_values('beta')
    label = f"m={nm}, c={nc}"
    axes[0, 0].plot(sub['beta'], sub['p00'], 'o-', label=label, alpha=0.6, markersize=4)
    axes[0, 1].plot(sub['beta'], sub['re01'], 's-', label=label, alpha=0.6, markersize=4)

axes[0, 0].set_title(f"Population vs Beta (g={max_g}, theta={max_th})")
axes[0, 1].set_title(f"Coherence Re(rho01) vs Beta (g={max_g}, theta={max_th})")

# --- ROW 2: COUPLING CONVERGENCE (vs g) ---
# Left: p00, Right: re01
for (nm, nc), group in sectors:
    sub = group[(group['beta'] == max_beta) & (group['theta'] == max_th)].sort_values('g')
    label = f"m={nm}, c={nc}"
    axes[1, 0].plot(sub['g'], sub['p00'], 'o-', label=label, alpha=0.6, markersize=4)
    axes[1, 1].plot(sub['g'], sub['re01'], 's-', label=label, alpha=0.6, markersize=4)

axes[1, 0].set_title(f"Population vs Coupling g (beta={max_beta}, theta={max_th})")
axes[1, 1].set_title(f"Coherence vs Coupling g (beta={max_beta}, theta={max_th})")

# --- ROW 3: GEOMETRY CONVERGENCE (vs theta) ---
# Left: p00, Right: re01
for (nm, nc), group in sectors:
    sub = group[(group['beta'] == max_beta) & (group['g'] == max_g)].sort_values('theta')
    label = f"m={nm}, c={nc}"
    axes[2, 0].plot(sub['theta'], sub['p00'], 'o-', label=label, alpha=0.6, markersize=4)
    axes[2, 1].plot(sub['theta'], sub['re01'], 's-', label=label, alpha=0.6, markersize=4)

axes[2, 0].set_title(f"Population vs Angle theta (beta={max_beta}, g={max_g})")
axes[2, 1].set_title(f"Coherence vs Angle theta (beta={max_beta}, g={max_g})")

# Formatting
for ax in axes.flatten():
    ax.legend(fontsize='x-small', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Physical Parameter")

plt.tight_layout()
plt.savefig("hmf_definitive_convergence_suite.png", dpi=200)
print("Omnibus Dashboard saved to hmf_definitive_convergence_suite.png")
