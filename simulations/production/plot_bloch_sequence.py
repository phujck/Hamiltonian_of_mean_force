import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Set stylistic parameters
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
})

# Parameters
r_par = 0.5
r_perp = 0.866
b_r = 1.0
b_nr = 1.5
a = 0.5 # beta omega_q / 2

# Vectors defined in the (x, z) = (\hat{r}_\perp, n_s) plane
n_s = np.array([0, 1.0])
r_perp_hat = np.array([1.0, 0])
r_vec = np.array([r_perp, r_par])

a_b = np.array([b_r * r_perp, b_nr * r_par])
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

v_bare = np.array([0, -np.tanh(a)])

# Start plot
fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))
fig.subplots_adjust(wspace=0.1)

# Colors
c_sys = '#333333'
c_coupling = '#2ca02c'
c_channel = '#ff7f0e'
c_influence = '#1f77b4'
c_state = '#d62728'

for ax in axes:
    circle = patches.Circle((0, 0), radius=1.0, edgecolor='black', facecolor='none', linestyle='-', linewidth=1.5, zorder=1)
    ax.add_patch(circle)
    ax.plot([-1.1, 1.1], [0, 0], 'k:', alpha=0.3, zorder=0)
    ax.plot([0, 0], [-1.1, 1.1], 'k:', alpha=0.3, zorder=0)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal')
    ax.axis('off')
    # add axis labels
    ax.text(0, 1.08, r'$\mathbf{n}_s$', ha='center', va='bottom', fontsize=14, alpha=0.6)
    ax.text(1.08, 0, r'$\hat{\mathbf{r}}_\perp$', ha='left', va='center', fontsize=14, alpha=0.6)

def draw_vector(ax, v, color, label, linestyle='-', lw=2.5, zorder=3, alpha=1.0, text_offset=None, ha='center', va='center', fs=20):
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-5: return
    ax.arrow(0, 0, v[0], v[1], head_width=0.04, head_length=0.06, fc=color, ec=color, 
             linestyle=linestyle, linewidth=lw, zorder=zorder, alpha=alpha, length_includes_head=True)
    if text_offset is None:
        text_offset = v * (1.0 + 0.15/v_norm)
    if label:
        ax.text(text_offset[0], text_offset[1], label, color=color, ha=ha, va=va, fontsize=fs, zorder=10)

# Panel 1
ax = axes[0]
ax.set_title(r"(a) Decompose", pad=20)
draw_vector(ax, r_vec, c_coupling, r'$\mathbf{r}$')
ax.plot([r_perp, r_perp], [0, r_par], color=c_coupling, linestyle=':', alpha=0.6, lw=2)
ax.plot([0, r_perp], [r_par, r_par], color=c_coupling, linestyle=':', alpha=0.6, lw=2)
ax.text(r_perp/2, -0.1, r'$r_\perp$', color=c_coupling, ha='center', fontsize=16)
ax.text(-0.1, r_par/2, r'$r_\parallel$', color=c_coupling, ha='right', fontsize=16)
draw_vector(ax, np.array([r_perp, 0]), c_coupling, '', linestyle='-', lw=3, alpha=0.4)
draw_vector(ax, np.array([0, r_par]), c_coupling, '', linestyle='-', lw=3, alpha=0.4)

# Panel 2
ax = axes[1]
ax.set_title(r"(b) $\xrightarrow{\mathbf{b}}$", pad=20)
draw_vector(ax, r_vec, c_coupling, r'', linestyle=':', alpha=0.3, lw=2)
draw_vector(ax, a_b, c_channel, r'$\mathbf{a}_b$')
ax.plot([a_b[0], a_b[0]], [0, a_b[1]], color=c_channel, linestyle=':', alpha=0.6, lw=2)
ax.plot([0, a_b[0]], [a_b[1], a_b[1]], color=c_channel, linestyle=':', alpha=0.6, lw=2)
ax.text(a_b[0]/2, -0.1, r'$b_{\mathrm{r}} r_\perp$', color=c_channel, ha='center', fontsize=16)
ax.text(-0.1, a_b[1]/2, r'$b_{\mathrm{nr}} r_\parallel$', color=c_channel, ha='right', fontsize=16)
draw_vector(ax, np.array([a_b[0], 0]), c_channel, '', linestyle='-', lw=3, alpha=0.4)
draw_vector(ax, np.array([0, a_b[1]]), c_channel, '', linestyle='-', lw=3, alpha=0.4)

# Panel 3
ax = axes[2]
ax.set_title(r"(c) $\xrightarrow{\mathbf{N}\times}$", pad=20)
draw_vector(ax, a_b, c_channel, r'$\mathbf{a}_b$', alpha=0.4)
draw_vector(ax, h_tilde, c_influence, r'$\tilde{\mathbf{h}}$')
# arc
theta1 = np.arctan2(a_b[1], a_b[0])
theta2 = np.arctan2(h_tilde[1], h_tilde[0])
if theta2 < theta1: theta2 += 2*np.pi
radius = 0.35
arc_x = [radius * np.cos(t) for t in np.linspace(theta1, theta2, 50)]
arc_y = [radius * np.sin(t) for t in np.linspace(theta1, theta2, 50)]
ax.plot(arc_x, arc_y, color='black', alpha=0.6, linewidth=1.5, zorder=2)
dx = arc_x[-1] - arc_x[-2]
dy = arc_y[-1] - arc_y[-2]
ax.arrow(arc_x[-1], arc_y[-1], 0.001*dx, 0.001*dy, head_width=0.04, head_length=0.06, fc='black', ec='black', alpha=0.6, zorder=2)
ax.text(radius*np.cos((theta1+theta2)/2) + 0.05, radius*np.sin((theta1+theta2)/2) + 0.05, 
        r'$90^\circ$', color='black', fontsize=14, alpha=0.8)

# Panel 4
ax = axes[3]
ax.set_title(r"(d) $\xrightarrow{\times r_\perp, \gamma(\chi)}$", pad=20)
draw_vector(ax, h_tilde, c_influence, r'$\tilde{\mathbf{h}}$', linestyle=':', alpha=0.4, lw=2)
draw_vector(ax, h_eff, c_influence, r'$\mathbf{h}_{\mathrm{eff}}$', lw=2, linestyle='--', text_offset=[h_eff[0] - 0.25, h_eff[1] + 0.15])
draw_vector(ax, u_vec, c_state, r'$\mathbf{u}$')
ax.text(h_eff[0]/2 - 0.1, h_eff[1]/2 + 0.15, r'$\times r_\perp$', color=c_influence, ha='right', fontsize=16)

# Panel 5
ax = axes[4]
ax.set_title(r"(e) $\xrightarrow{\Pi}$", pad=20)
draw_vector(ax, u_vec, c_state, r'$\mathbf{u}$', linestyle='--', alpha=0.5, lw=2)
draw_vector(ax, v_bare, 'purple', r'$\mathbf{v}_{\mathrm{th}}$', linestyle=':')
draw_vector(ax, v_vec, c_state, r'$\mathbf{v}$', lw=3)
ax.plot([v_bare[0], v_vec[0]], [v_bare[1], v_vec[1]], color='gray', linestyle='-', alpha=0.4, lw=1.5)
ax.text((v_bare[0]+v_vec[0])/2 + 0.1, (v_bare[1]+v_vec[1])/2, r'shift', color='gray', fontsize=14, rotation=np.degrees(np.arctan2(v_vec[1]-v_bare[1], v_vec[0]-v_bare[0])))

plt.tight_layout()
out_dir = "../../manuscript/figures/"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "hmf_fig5_sequence.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_fig5_sequence.png"), bbox_inches='tight', dpi=300)
print("Saved sequence plots to figures directory")
