import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
})

def dual_r(r_bar, beta_omegaq):
    """Dual radius: arctanh(r) + arctanh(r_vee) = beta*omega_q"""
    a = np.arctanh(np.clip(r_bar, 1e-9, 1 - 1e-9))
    b = beta_omegaq - a
    return np.where(b > 0, np.tanh(b), np.nan)

# Vertical layout
fig, axes = plt.subplots(2, 1, figsize=(4.8, 8.5))

# ── Panel (a): dual radius (hyperbolas) ─────────────────────────────────────
ax = axes[0]

beta_omegaqs = [0.6, 1.2, 2.0]
colors = ['#1f77b4', '#d62728', '#2ca02c']
labels = [r'$\beta\omega_q = 0.6$', r'$\beta\omega_q = 1.2$', r'$\beta\omega_q = 2.0$']

r_vals = np.linspace(0.001, 0.999, 800)

for bw, col, lbl in zip(beta_omegaqs, colors, labels):
    r0 = np.tanh(bw / 2)
    rdual = dual_r(r_vals, bw)
    mask = (rdual > 0) & (rdual < 1)
    ax.plot(r_vals[mask], rdual[mask], color=col, lw=2.4, label=lbl)
    # self-dual fixed point on diagonal
    ax.plot(r0, r0, 'o', color=col, ms=8, zorder=5)

# diagonal (identity)
ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.45)
ax.text(0.12, 0.17, r'$\bar{r}^{\vee}=\bar{r}$', fontsize=11, alpha=0.6,
        rotation=45, ha='center', va='center')

# Label the three r_0 dots cleanly
offsets = [(-0.13, -0.04), (0.05, -0.13), (0.04, -0.13)]
for bw, col, (dx, dy) in zip(beta_omegaqs, colors, offsets):
    r0 = np.tanh(bw / 2)
    ax.annotate(r'$r_0$', xy=(r0, r0), xytext=(r0 + dx, r0 + dy),
                color=col, fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='-', color=col, lw=0.8, shrinkA=3, shrinkB=2))



ax.set_xlabel(r'$\bar{r}$')
ax.set_ylabel(r'$\bar{r}^{\,\vee}$')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(fontsize=11, loc='upper right', framealpha=0.88)
ax.set_title(r'(a)\ Dual radius', fontsize=14, pad=8)

# ── Panel (b): dual angle (straight line) ───────────────────────────────────
ax2 = axes[1]

phi_vals = np.linspace(0, np.pi, 400)
phi_dual = np.pi - phi_vals

ax2.plot(phi_vals, phi_dual, color='#9467bd', lw=2.5,
         label=r'$\varphi_Q^{\,\vee} = \pi - \varphi_Q$')
ax2.plot([0, np.pi], [0, np.pi], 'k--', lw=1.0, alpha=0.45)
ax2.text(2.0, 2.1, r'$\varphi_Q^{\vee}=\varphi_Q$', fontsize=11, alpha=0.6,
         rotation=45, ha='center', va='center')

# fixed point
ax2.plot(np.pi/2, np.pi/2, 'o', color='#9467bd', ms=9, zorder=5)
ax2.annotate(r'$\varphi_Q = \pi/2$', xy=(np.pi/2, np.pi/2),
             xytext=(np.pi/2 + 0.35, np.pi/2 + 0.35),
             color='#9467bd', fontsize=11,
             arrowprops=dict(arrowstyle='-', color='#9467bd', lw=0.8))

pi_ticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
pi_labels = [r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
ax2.set_xticks(pi_ticks)
ax2.set_xticklabels(pi_labels)
ax2.set_yticks(pi_ticks)
ax2.set_yticklabels(pi_labels)
ax2.set_xlabel(r'$\varphi_Q$')
ax2.set_ylabel(r'$\varphi_Q^{\,\vee}$')
ax2.set_xlim(0, np.pi)
ax2.set_ylim(0, np.pi)
ax2.set_aspect('equal')
ax2.legend(fontsize=11, loc='upper right', framealpha=0.88)
ax2.set_title(r'(b)\ Dual tilt angle', fontsize=14, pad=8)

plt.tight_layout(pad=1.8)

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.abspath(os.path.join(script_dir, "../../manuscript/figures/"))
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "hmf_bloch_dual_mapping.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "hmf_bloch_dual_mapping.png"), bbox_inches='tight', dpi=200)
print(f"Saved dual mapping figure to {out_dir}")
