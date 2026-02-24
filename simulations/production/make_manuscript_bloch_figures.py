import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set stylistic parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"
})

# Parameters for "worked point"
theta = 3 * np.pi / 8
beta_wq = 4
a = beta_wq / 2  # 2.0
g_gstar = 1.0

# Qubit geometry vectors in (x,z) plane = (r_perp_hat, n_s)
n_s = np.array([0, 1.0])
r_perp_hat = np.array([1.0, 0])
r_vec = np.array([np.sin(theta), np.cos(theta)])

# Bath channels for Ohmic bath (approximate ratio for beta*wq=4)
# Usually b_nr > b_r at finite temperature. 
# Let's pick b_nr = 1.2, b_r = 0.8 as a representative Ohmic-like pair.
b_nr = 1.2
b_r = 0.8
# The magnitude chi must be 1.0
# chi = r_perp * |a_b| = sin(theta) * sqrt((b_r*sin(theta))^2 + (b_nr*cos(theta))^2)
# With theta=pi/4: chi = 0.707 * 0.707 * sqrt(b_r^2 + b_nr^2) = 0.5 * sqrt(b_r^2 + b_nr^2)
# To get chi=1, we need sqrt(b_r^2 + b_nr^2) = 2.
# Let's scale our test values:
scale = 1.0 / (np.sin(theta) * np.sqrt((b_r * np.sin(theta))**2 + (b_nr * np.cos(theta))**2))
b_nr *= scale
b_r *= scale

# Vectors
a_b = np.array([b_r * np.sin(theta), b_nr * np.cos(theta)])
# Quarter turn N x a_b = (-a_bz, a_bx)
h_tilde = np.array([-a_b[1], a_b[0]])
h_eff = np.sin(theta) * h_tilde
chi = np.linalg.norm(h_eff) # Should be 1.0
gamma = np.tanh(chi) / chi
u_vec = gamma * h_eff

# Final state vector v
u_perp = u_vec[0]
u_par = u_vec[1]
denom = np.cosh(a) - u_par * np.sinh(a)
v_perp = u_perp / denom
v_par = (u_par * np.cosh(a) - np.sinh(a)) / denom
v_vec = np.array([v_perp, v_par])

# Reference states
v_th = np.array([0, -np.tanh(a)])

# Colors
c_sys = '#333333'
c_coupling = '#2ca02c'
c_channel = '#ff7f0e'
c_influence = '#1f77b4'
c_state = '#d62728'
c_thermal = '#9467bd'

def setup_ax(ax, title=None):
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='none', linestyle='-', linewidth=1.5, zorder=1)
    ax.add_patch(circle)
    ax.plot([-1.1, 1.1], [0, 0], 'k:', alpha=0.3, zorder=0)
    ax.plot([0, 0], [-1.1, 1.1], 'k:', alpha=0.3, zorder=0)
    ax.set_xlim([-1.2, 1.2]) # Slightly tighter
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal')
    ax.axis('off')
    # Move labels slightly inward to ensure they stay mapped inside the graphic
    ax.text(0.05, 1.02, r'$\mathbf{n}_s$', ha='left', va='bottom', fontsize=18)
    ax.text(1.02, 0.05, r'$\hat{\mathbf{r}}_\perp$', ha='left', va='bottom', fontsize=18)
    if title:
        ax.set_title(title, pad=10)

def draw_vec(ax, v, color, label, linestyle='-', lw=3, zorder=3, alpha=1.0, text_offset=None, fs=22):
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-5: return
    ax.arrow(0, 0, v[0], v[1], head_width=0.045, head_length=0.065, fc=color, ec=color, 
             linestyle=linestyle, linewidth=lw, zorder=zorder, alpha=alpha, length_includes_head=True)
    if label:
        if text_offset is None:
            text_offset = v * (1.0 + 0.18/v_norm)
        ax.text(text_offset[0], text_offset[1], label, color=color, ha='center', va='center', fontsize=fs, zorder=10)

out_dir = "../../manuscript/figures/"
os.makedirs(out_dir, exist_ok=True)

# 1. hmf_bloch_overview.pdf
fig, ax = plt.subplots(figsize=(6, 6))
setup_ax(ax)
# Shade the coupling plane (the disk itself)
coupling_plane = patches.Circle((0, 0), radius=1.0, facecolor='gray', alpha=0.08, zorder=0)
ax.add_patch(coupling_plane)

draw_vec(ax, n_s, c_sys, r'$\mathbf{n}_s$', linestyle='--')
draw_vec(ax, r_vec, c_coupling, r'$\mathbf{r}$')
draw_vec(ax, v_vec, c_state, r'$\mathbf{v}$', text_offset=[-0.4, -0.7])

# Mark the polar angle theta between n_s and r
# n_s is at 90 deg, r is at 90-theta deg
theta_deg = np.degrees(theta)
arc_theta = patches.Arc((0, 0), 0.5, 0.5, theta1=90-theta_deg, theta2=90, color='black', alpha=0.6, lw=1.5)
ax.add_patch(arc_theta)
# Position for the theta label
ax.text(0.12, 0.35, r'$\theta$', fontsize=20, ha='center', va='center')

# Note the angle between v and the transverse coupling component (horizontal)
v_angle_deg = np.degrees(np.arctan2(v_vec[1], v_vec[0]))
arc_v = patches.Arc((0, 0), 0.4, 0.4, theta1=v_angle_deg, theta2=0, color=c_state, alpha=0.4, lw=1.5)
ax.add_patch(arc_v)

plt.savefig(os.path.join(out_dir, "hmf_bloch_overview.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_bloch_overview.png"), bbox_inches='tight', dpi=300)
plt.close()

# 2. hmf_bloch_decompose_weighting.pdf
fig, ax = plt.subplots(figsize=(6, 6))
setup_ax(ax)
draw_vec(ax, r_vec, c_coupling, r'$\mathbf{r}$', alpha=0.4)
# Projections
ax.plot([r_vec[0], r_vec[0]], [0, r_vec[1]], color=c_coupling, linestyle=':', alpha=0.5)
ax.plot([0, r_vec[0]], [r_vec[1], r_vec[1]], color=c_coupling, linestyle=':', alpha=0.5)
draw_vec(ax, a_b, c_channel, r'$\mathbf{a}_b$')
# Bath weights labels
ax.text(r_vec[0]/2, -0.15, r'$b_{\rm r} r_\perp$', color=c_channel, ha='center', fontsize=16)
ax.text(-0.25, r_vec[1]/2, r'$b_{\rm nr} r_\parallel$', color=c_channel, ha='right', fontsize=16)
plt.savefig(os.path.join(out_dir, "hmf_bloch_decompose_weighting.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_bloch_decompose_weighting.png"), bbox_inches='tight', dpi=300)
plt.close()

# 3. hmf_bloch_symmetrised_map.pdf
fig, ax = plt.subplots(figsize=(6, 6))
setup_ax(ax)
draw_vec(ax, a_b, c_channel, r'$\mathbf{a}_b$', alpha=0.3)
draw_vec(ax, h_tilde, c_influence, r'$\tilde{\mathbf{h}}$')
# Rotation arc
theta_a = np.arctan2(a_b[1], a_b[0])
theta_h = np.arctan2(h_tilde[1], h_tilde[0])
if theta_h < theta_a: theta_h += 2*np.pi
arc = patches.Arc((0,0), 0.5, 0.5, theta1=np.degrees(theta_a), theta2=np.degrees(theta_h), color='black', alpha=0.6)
ax.add_patch(arc)
ax.text(0.35, 0.35, r'$\mathbf{N}\times$', fontsize=16)
# Full vectors
draw_vec(ax, h_eff, c_influence, r'$\mathbf{h}_{\mathrm{eff}}$', linestyle='--', alpha=0.6, text_offset=[-0.8, 0.6])
draw_vec(ax, u_vec, c_state, r'$\mathbf{u}$', text_offset=[-0.35, 0.4])
plt.savefig(os.path.join(out_dir, "hmf_bloch_symmetrised_map.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_bloch_symmetrised_map.png"), bbox_inches='tight', dpi=300)
plt.close()

# 4. hmf_bloch_thermal_recombination.pdf
fig, ax = plt.subplots(figsize=(6, 6))
setup_ax(ax)
draw_vec(ax, u_vec, c_state, r'$\mathbf{u}$', alpha=0.4, linestyle='--')
draw_vec(ax, v_th, c_thermal, r'$\mathbf{v}_{\mathrm{th}}$', text_offset=[-0.25, -0.6])
draw_vec(ax, v_vec, c_state, r'$\mathbf{v}$')
# Link
ax.plot([v_th[0], v_vec[0]], [v_th[1], v_vec[1]], 'k-', alpha=0.2, lw=1)
# Attractors inset logic? Skip inset for simplicity or add a small one.
# For now just clear vectors.
plt.savefig(os.path.join(out_dir, "hmf_bloch_thermal_recombination.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_bloch_thermal_recombination.png"), bbox_inches='tight', dpi=300)
plt.close()

print("Generated 4 manuscript Bloch figures.")
