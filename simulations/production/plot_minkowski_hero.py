"""
Two-panel hero figure for the introduction.
(a) Bloch-disk cross-section showing anisotropic bath influence and
    the orientability symmetry breaking that generates duality.
(b) Minkowski causal diagram in thermal coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

# ══════════════════════════════════════════════════════════════════════════════
# PANEL (a): Bloch disk cross-section
# ══════════════════════════════════════════════════════════════════════════════
ax = ax1

# Bloch circle
theta_circle = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', lw=1.2, alpha=0.25)
ax.plot(0.5*np.cos(theta_circle), 0.5*np.sin(theta_circle), 'k--', lw=0.5, alpha=0.15)

# Axes
ax.axhline(0, color='k', lw=0.4, alpha=0.2)
ax.axvline(0, color='k', lw=0.4, alpha=0.2)

# Axis labels
ax.text(0.0, 1.18, r'$m_z$', fontsize=13, ha='center', va='center', color='k', alpha=0.6)
ax.text(1.18, 0.0, r'$m_\perp$', fontsize=13, ha='center', va='center', color='k', alpha=0.6)

# ── Bare Gibbs vector (on negative z-axis) ───────────────────────────────────
r0 = 0.45  # bare radius
ax.annotate('', xy=(0, -r0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0))
ax.text(0.08, -r0/2, r'$r_0$', fontsize=13, ha='left', va='center',
        color='#555555')
ax.text(0.08, -r0-0.07, r'bare', fontsize=10, ha='left', va='top',
        color='#888888', style='italic')

# ── Coupling direction (at angle theta from z) ──────────────────────────────
theta_coup = np.pi/4  # coupling angle
coup_len = 1.05
cx = np.sin(theta_coup) * coup_len
cz = -np.cos(theta_coup) * coup_len
ax.plot([0, cx], [0, cz], '--', color='#999999', lw=1.0, alpha=0.5)
ax.text(cx+0.06, cz-0.06, r'$\hat{f}$', fontsize=12, ha='left', va='top',
        color='#999999')

# ── Show theta angle ────────────────────────────────────────────────────────
arc_r = 0.3
arc_angles = np.linspace(-np.pi/2, -np.pi/2 + theta_coup, 50)
ax.plot(arc_r*np.cos(arc_angles), arc_r*np.sin(arc_angles), '-', color='#999999', lw=0.8)
ax.text(0.12, -0.32, r'$\theta$', fontsize=11, color='#999999')

# ── Bath influence arrows ────────────────────────────────────────────────────
# Mean-force state: tilted and with modified radius
phi_Q = np.pi/4 + np.pi/2 + 0.15  # tilt angle from +x axis (measuring from negative z)
rQ = 0.72  # renormalised radius
# In Bloch coordinates: state angle from -z axis is phi_Q,
# so Bloch vector = rQ * (sin(tilt), -cos(tilt))
tilt_from_negz = 0.45  # angle away from -z axis toward transverse
mx = rQ * np.sin(tilt_from_negz)
mz = -rQ * np.cos(tilt_from_negz)

# Draw the mean-force Bloch vector
ax.annotate('', xy=(mx, mz), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
ax.text(mx+0.08, mz+0.06, r'$\mathbf{r}_Q$', fontsize=14, ha='left', va='bottom',
        color='#d62728', fontweight='bold')

# Longitudinal pull: along -z, showing temperature renormalization
# Arrow from bare endpoint to the projection of rQ on z-axis
ax.annotate('', xy=(0, mz), xytext=(0, -r0),
            arrowprops=dict(arrowstyle='->', color='#2266aa', lw=2.0,
                            connectionstyle='arc3,rad=0.0'))
ax.text(-0.28, (mz - r0)/2 - 0.02, r'$\Sigma_z$', fontsize=13, ha='center',
        color='#2266aa')
ax.text(-0.28, (mz - r0)/2 - 0.15, r'\footnotesize rapidity', fontsize=9, ha='center',
        color='#2266aa', alpha=0.7)

# Transverse pull: horizontal, showing tilt
ax.annotate('', xy=(mx, mz), xytext=(0, mz),
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2.0,
                            connectionstyle='arc3,rad=0.0'))
ax.text(mx/2, mz - 0.13, r'$\Sigma_\perp$', fontsize=13, ha='center',
        color='#9b59b6')
ax.text(mx/2, mz - 0.25, r'\footnotesize tilt', fontsize=9, ha='center',
        color='#9b59b6', alpha=0.7)

# ── Dual state (orientability flip: tilt reflected) ──────────────────────────
mx_dual = -mx
mz_dual = mz  # same longitudinal, reflected transverse

ax.annotate('', xy=(mx_dual, mz_dual), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.8,
                            linestyle='dashed'))
ax.text(mx_dual-0.08, mz_dual+0.06, r'$\mathbf{r}_Q^{\,\vee}$', fontsize=14,
        ha='right', va='bottom', color='#d62728', alpha=0.7)

# Dashed arc showing the flip
arc_angles2 = np.linspace(
    np.arctan2(mz, mx),
    np.arctan2(mz_dual, mx_dual),
    50
)
arc_r2 = rQ * 0.55
ax.plot(arc_r2*np.cos(arc_angles2), arc_r2*np.sin(arc_angles2), ':',
        color='#d62728', lw=1.2, alpha=0.5)
ax.text(-0.05, -0.62, r'$\psi \leftrightarrow -\psi$', fontsize=10,
        ha='center', va='center', color='#d62728', alpha=0.7)

# Panel label
ax.text(-1.1, 1.1, r'\textbf{(a)}', fontsize=15, ha='left', va='top')

ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL (b): Minkowski diagram
# ══════════════════════════════════════════════════════════════════════════════
ax = ax2

L = 1.6
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)

# Light cone fills
cone_x = np.linspace(-L, L, 300)
ax.fill_between(cone_x, np.abs(cone_x), L, color='#e8d4f0', alpha=0.35, zorder=0)
ax.fill_between(cone_x, -L, -np.abs(cone_x), color='#e8d4f0', alpha=0.35, zorder=0)
ax.fill_between(cone_x, -np.abs(cone_x), np.abs(cone_x), color='#d4e8f0', alpha=0.35, zorder=0)

# Light cone lines
ax.plot([-L, 0, L], [-L, 0, L], 'k-', lw=1.2, alpha=0.5, zorder=2)
ax.plot([-L, 0, L], [L, 0, -L], 'k-', lw=1.2, alpha=0.5, zorder=2)

# Axes
ax.axhline(0, color='k', lw=0.6, alpha=0.3)
ax.axvline(0, color='k', lw=0.6, alpha=0.3)

# Self-dual fixed point
ax.plot(0, 0, 'ko', ms=10, zorder=10)
ax.annotate(r'self-dual', xy=(0, 0), xytext=(0.22, -0.22),
            fontsize=11, ha='left', va='top',
            arrowprops=dict(arrowstyle='-', lw=0.8, color='k'),
            zorder=10)

# Dual pairs
pairs = [
    (0.9, 0.35),
    (0.5, 0.9),
    (-0.3, -1.05),
]
pair_colors = ['#d62728', '#1f77b4', '#2ca02c']

for (x, y), col in zip(pairs, pair_colors):
    xd, yd = -x, -y
    ax.plot(x, y, 'o', color=col, ms=8, zorder=8)
    ax.plot(xd, yd, 's', color=col, ms=8, zorder=8)
    ax.annotate('', xy=(xd, yd), xytext=(x, y),
                arrowprops=dict(arrowstyle='<->', color=col, lw=1.4,
                                connectionstyle='arc3,rad=0.0',
                                shrinkA=5, shrinkB=5),
                zorder=7)

# Region labels
ax.text(0.0, 1.35, r'\textit{spacelike}', fontsize=12, ha='center', va='center',
        color='#7b3f9e', style='italic')
ax.text(0.0, -1.35, r'\textit{spacelike}', fontsize=12, ha='center', va='center',
        color='#7b3f9e', style='italic')
ax.text(0.0, 1.12, r'tilt-dominated', fontsize=9, ha='center', va='center',
        color='#7b3f9e', alpha=0.8)
ax.text(0.0, -1.12, r'tilt-dominated', fontsize=9, ha='center', va='center',
        color='#7b3f9e', alpha=0.8)

ax.text(1.15, 0.0, r'\textit{timelike}', fontsize=12, ha='center', va='center',
        color='#2266aa', style='italic')
ax.text(-1.15, 0.0, r'\textit{timelike}', fontsize=12, ha='center', va='center',
        color='#2266aa', style='italic')
ax.text(1.15, -0.18, r'$T$-renorm.', fontsize=9, ha='center', va='center',
        color='#2266aa', alpha=0.8)
ax.text(-1.15, -0.18, r'$T$-renorm.', fontsize=9, ha='center', va='center',
        color='#2266aa', alpha=0.8)

# Light cone label
ax.text(1.15, 1.35, r'light cone', fontsize=9, ha='center', va='center',
        color='#555555', rotation=45, alpha=0.7)

# Axis labels
ax.set_xlabel(r'$\frac{\omega_q}{2}\,\delta\bar\beta$ \quad (rapidity)', fontsize=14)
ax.set_ylabel(r'$\psi = \varphi_Q - \pi/2$ \quad (tilt deviation)', fontsize=14)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('#888888')

# Legend
circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                     ms=8, label=r'state')
square = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                     ms=8, label=r'dual')
ax.legend(handles=[circle, square], fontsize=10, loc='lower right',
          framealpha=0.88, edgecolor='#cccccc')

# Panel label
ax.text(-L+0.05, L-0.05, r'\textbf{(b)}', fontsize=15, ha='left', va='top')

# ══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(w_pad=1.5)

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.abspath(os.path.join(script_dir, "../../manuscript/figures/"))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "hmf_minkowski_hero.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_minkowski_hero.png"), bbox_inches='tight', dpi=200)
print(f"Saved hero figure to {out_dir}")
