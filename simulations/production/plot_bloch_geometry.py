import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set stylistic parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
})

# Parameters
r_par = 0.5
r_perp = 0.866
b_r = 0.8
b_nr = 0.8  # Make non-resonant coupling larger to visually distinguish a_b from r
a = 0.4 # beta omega_q / 2

# Vectors defined in the (x, z) = (\hat{r}_\perp, n_s) plane
n_s = np.array([0, 1.0])
r_perp_hat = np.array([1.0, 0])
r_vec = np.array([r_perp, r_par])

a_b = np.array([b_r * r_perp, b_nr * r_par])
# Quarter turn! N x a_b = (-\hat{y}) x (x, z) = (-z, x)
h_tilde = np.array([-a_b[1], a_b[0]])
h_eff = r_perp * h_tilde

chi = np.linalg.norm(h_eff)
gamma = np.tanh(chi) / chi if chi > 1e-5 else 1.0
u_vec = gamma * h_eff

u_perp = u_vec[0]
u_par = u_vec[1]
denom = np.cosh(a) - u_par * np.sinh(a)
v_perp = u_perp / denom
v_par = (u_par * np.cosh(a) - np.sinh(a)) / denom
v_vec = np.array([v_perp, v_par])

# Bare thermal state
v_bare = np.array([0, -np.tanh(a)])

# Start plot
fig, ax = plt.subplots(figsize=(6.5, 6.5))

# Draw Bloch circle
circle = patches.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='none', linestyle='-', linewidth=1.5, zorder=1)
ax.add_patch(circle)
ax.plot([-1.1, 1.1], [0, 0], 'k:', alpha=0.3, zorder=0)
ax.plot([0, 0], [-1.1, 1.1], 'k:', alpha=0.3, zorder=0)

def draw_vector(ax, v, color, label, linestyle='-', lw=2, zorder=3, alpha=1.0, text_offset=None, ha='center', va='center'):
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-5: return
    ax.arrow(0, 0, v[0], v[1], head_width=0.035, head_length=0.05, fc=color, ec=color, 
             linestyle=linestyle, linewidth=lw, zorder=zorder, alpha=alpha, length_includes_head=True)
    if text_offset is None:
        text_offset = v * (1.0 + 0.15/v_norm)
    ax.text(text_offset[0], text_offset[1], label, color=color, ha=ha, va=va, fontsize=20, zorder=10)

# Colors
c_sys = '#333333'
c_coupling = '#2ca02c'
c_channel = '#ff7f0e'
c_influence = '#1f77b4'
c_state = '#d62728'

# Draw vectors
draw_vector(ax, n_s, c_sys, r'$\mathbf{n}_s$', linestyle='--', lw=1, text_offset=[-0.08, 1.08])
draw_vector(ax, r_perp_hat, c_sys, r'$\hat{\mathbf{r}}_\perp$', linestyle='--', lw=1, text_offset=[1.08, -0.08])
draw_vector(ax, -n_s, c_sys, r'$-\mathbf{n}_s$', linestyle='--', lw=1, text_offset=[-0.12, -1.08])

draw_vector(ax, r_vec, c_coupling, r'$\mathbf{r}$', linestyle=':', lw=2, alpha=0.7)
draw_vector(ax, a_b, c_channel, r'$\mathbf{a}_b$', linestyle='-', lw=2.5)
draw_vector(ax, h_tilde, c_influence, r'$\tilde{\mathbf{h}}$', linestyle='-', lw=2.5)
draw_vector(ax, h_eff, c_influence, r'$\mathbf{h}_{\mathrm{eff}}$', linestyle='--', lw=2, alpha=0.8, text_offset=[-0.35, -0.15])
draw_vector(ax, v_vec, c_state, r'$\mathbf{v}$', linestyle='-', lw=3, zorder=5)
draw_vector(ax, v_bare, c_state, r'$\mathbf{v}_{\mathrm{th}}$', linestyle=':', lw=2, alpha=0.6, text_offset=[0.12, -0.4])

# Draw quarter turn arc from a_b to h_tilde
theta1 = np.arctan2(a_b[1], a_b[0])
theta2 = np.arctan2(h_tilde[1], h_tilde[0])
if theta2 < theta1: theta2 += 2*np.pi
radius = 0.25
arc_x = [radius * np.cos(t) for t in np.linspace(theta1, theta2, 50)]
arc_y = [radius * np.sin(t) for t in np.linspace(theta1, theta2, 50)]
ax.plot(arc_x, arc_y, color='black', alpha=0.6, linewidth=1.5, zorder=2)
# Arrow head for arc
dx = arc_x[-1] - arc_x[-2]
dy = arc_y[-1] - arc_y[-2]
ax.arrow(arc_x[-1], arc_y[-1], 0.001*dx, 0.001*dy, head_width=0.03, head_length=0.04, fc='black', ec='black', alpha=0.6, zorder=2)

ax.text(radius*np.cos((theta1+theta2)/2) + 0.05, radius*np.sin((theta1+theta2)/2) + 0.05, 
        r'$\mathbf{N} \times$', color='black', fontsize=14, alpha=0.8)

# Limits and display
ax.set_xlim([-1.15, 1.15])
ax.set_ylim([-1.15, 1.15])
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out_dir = "../../manuscript/figures/"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "hmf_fig5_geometry.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_fig5_geometry.png"), bbox_inches='tight', dpi=300)
print("Saved geometry plots to figures directory")
