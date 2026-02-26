import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# ── Style Configuration ───────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# ── Physics Functions ────────────────────────────────────────────────────────
THETA_VAL = np.pi/4.0
OMEGA_Q   = 1.0

def get_chi0(beta, theta=np.pi/4.0):
    k0 = 1.0 / (2.0 * np.pi) 
    c, s = np.cos(theta), np.sin(theta)
    
    # Simple Ohmic kernels for demonstration
    k_0 = k0 * (2.0/beta)
    k_w = k0 * OMEGA_Q * (np.cosh(beta*OMEGA_Q/2.0)/np.sinh(beta*OMEGA_Q/2.0))
    r_w = k0 * OMEGA_Q 
    
    s_z = r_w 
    s_p = (4.0/OMEGA_Q) * (np.cosh(beta*OMEGA_Q/2.0)*k_0 - np.exp(-beta*OMEGA_Q/2.0)*k_w)
    
    chi0 = s * np.sqrt(c**2 * s_p**2 + s**2 * s_z**2)
    return chi0, s_z, s_p

def bloch_branch(g_arr, beta, theta=np.pi/4.0, flip=False):
    chi0, s_z, s_p = get_chi0(beta, theta)
    chi = g_arr**2 * chi0
    gamma = np.where(chi > 1e-9, np.tanh(chi)/chi, 1.0)
    
    c, s = np.cos(theta), np.sin(theta)
    delta_z = s**2 * s_z
    delta_p = -c * s * s_p
    
    theta_mf = np.arctanh(np.clip(gamma * g_arr**2 * delta_z, -0.999, 0.999)) - beta*OMEGA_Q/2.0
    kappa = (gamma * g_arr**2 * delta_p) / np.sqrt(np.clip(1.0 - (gamma * g_arr**2 * delta_z)**2, 1e-9, None))
    
    r_z = np.tanh(theta_mf)
    r_p = kappa / np.cosh(theta_mf)
    
    mx_std = r_p
    mz_std = r_z
    
    if not flip:
        return mx_std, mz_std
    
    # Geometric Flip: Reflection across theta/2
    # In polar coordinates: angle alpha relative to -z axis
    alpha = np.arctan2(mx_std, -mz_std)
    r_mag = np.sqrt(mx_std**2 + mz_std**2)
    
    # Pivot is theta/2. alpha -> theta - alpha
    alpha_flipped = theta - alpha
    mx_flip = r_mag * np.sin(alpha_flipped)
    mz_flip = -r_mag * np.cos(alpha_flipped)
    
    return mx_flip, mz_flip

# ── Main Plotting ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(3.4, 3.8))
gs = gridspec.GridSpec(1, 1, left=0.18, right=0.95, top=0.90, bottom=0.15)
ax = fig.add_subplot(gs[0])

# Coordinate Grid
ax.axvline(0, color="k", ls=":", lw=0.4, alpha=0.3)
ax.axhline(0, color="k", ls=":", lw=0.4, alpha=0.3)

# Parameters
BETA = 2.0
G_VALS = np.linspace(0.001, 8.0, 500)

# Colors
C_STD = "#2c7bb6" # Professional blue
C_FLP = "#d7191c" # Professional red/coral
C_PIV = "#636363" # Neutral gray

# Standard Manifold
mx_s, mz_s = bloch_branch(G_VALS, BETA, THETA_VAL, flip=False)
ax.plot(mx_s, mz_s, color=C_STD, lw=2.0, label=r"$\text{Standard Branch}$", zorder=5)

# Flipped Manifold
mx_f, mz_f = bloch_branch(G_VALS, BETA, THETA_VAL, flip=True)
ax.plot(mx_f, mz_f, color=C_FLP, lw=2.0, ls="--", label=r"$\text{Aligned Branch}$", zorder=5)

# Reference Vectors
# Coupling direction f
f_x, f_z = np.sin(THETA_VAL), -np.cos(THETA_VAL)
ax.annotate("", xy=(1.1*f_x, 1.1*f_z), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C_PIV, lw=0.8, alpha=0.6))
ax.text(1.15*f_x, 1.15*f_z, r"$\hat{\mathbf{f}}$ (Interaction)", 
        color=C_PIV, fontsize=9, ha="center", va="center")

# Geometric Pivot (theta/2)
p_x, p_z = np.sin(THETA_VAL/2), -np.cos(THETA_VAL/2)
ax.plot([0, 1.2*p_x], [0, 1.2*p_z], color=C_PIV, ls="-.", lw=0.6, alpha=0.4)
ax.text(1.2*p_x, 1.2*p_z, r"$\theta/2$ (Pivot)", 
        color=C_PIV, fontsize=8, ha="left", va="top", rotation=-22)

# Bare Axis
ax.text(-0.02, -1.05, r"$\hat{\mathbf{n}}_s$ (Bare)", 
        color=C_PIV, fontsize=9, ha="right", va="center")

# Disk Boundary
circ_phi = np.linspace(-np.pi/2, 0.2, 200)
ax.plot(np.cos(circ_phi), np.sin(circ_phi), "k", lw=0.8, alpha=0.15, zorder=1)

# Highlights: Attraction point
ax.plot(mx_s[-1], mz_s[-1], "o", color=C_STD, ms=4, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
ax.plot(mx_f[-1], mz_f[-1], "o", color=C_FLP, ms=4, markeredgecolor="white", markeredgewidth=0.5, zorder=6)

# Labels and Limits
ax.set_xlabel(r"Magnetisation $m_x$")
ax.set_ylabel(r"Magnetisation $m_z$")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-1.1, 0.1)

# Elegant Legend
ax.legend(loc="upper right", framealpha=0.9, borderpad=0.4)

ax.set_title(r"\textbf{Geometric Branch-Flip Alignment}", pad=12)

plt.savefig("../../manuscript/figures/hmf_branch_alignment.pdf")
plt.savefig("../../manuscript/figures/hmf_branch_alignment.png")
print("Saved figure to hmf_branch_alignment.pdf")
