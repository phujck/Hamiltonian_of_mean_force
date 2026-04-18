import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set stylistic parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
})

def plot_bloch_3d_v2():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # Geometric parameters (chosen for readability)
    theta = np.pi / 4.5      # shallower f hat (~40 deg from +z)
    beta_wq = 1.3
    r0_mag = np.tanh(beta_wq / 2)
    rS_mag = 0.78
    rQ_mag = 0.96
    varphi_S = np.pi / 4.8   # added ~5 deg more arc
    varphi_Q = np.pi / 2.6   # added ~9 deg more arc

    # Vectors (n_s is +z)
    n_s = np.array([0, 0, 1.0])

    # Bare state vector (pointing South)
    r0_vec = np.array([0, 0, -r0_mag])

    # Interaction axis f
    f_axis = np.array([np.sin(theta), 0, np.cos(theta)])

    # Symmetrised influence state and final reduced state (in coupling plane)
    rS_vec = rS_mag * np.array([np.sin(varphi_S), 0, -np.cos(varphi_S)])
    rQ_vec = rQ_mag * np.array([np.sin(varphi_Q), 0, -np.cos(varphi_Q)])

    # Sphere Surface (Ultra-Frosted)
    u_s, v_s = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = np.cos(u_s)*np.sin(v_s)
    y_s = np.sin(u_s)*np.sin(v_s)
    z_s = np.cos(v_s)
    ax.plot_surface(x_s, y_s, z_s, color="skyblue", alpha=0.04, shade=True, zorder=-1)
    
    # Grid Rims
    t_rim = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t_rim), np.sin(t_rim), 0, color='black', alpha=0.1, linewidth=0.5, zorder=0)
    ax.plot(np.cos(t_rim), 0, np.sin(t_rim), color='black', alpha=0.08, linewidth=0.5, zorder=0)
    ax.plot(0, np.cos(t_rim), np.sin(t_rim), color='black', alpha=0.05, linewidth=0.5, zorder=0)

    # THE COUPLING PLANE
    r_range = np.linspace(0, 1.0, 10)
    theta_range = np.linspace(0, 2*np.pi, 80)
    R, T = np.meshgrid(r_range, theta_range)
    X_p = R * np.sin(T)
    Y_p = R * 0 
    Z_p = R * np.cos(T)
    ax.plot_surface(X_p, Y_p, Z_p, color='gray', alpha=0.08, shade=False, zorder=1)
    ax.plot(np.sin(t_rim), np.zeros_like(t_rim), np.cos(t_rim), color='black', alpha=0.3, lw=0.8, zorder=2)

    # Drawing functions
    def draw_arrow(vec, color, label, lw=4.5, label_offset=1.1, fs=26, arrow_ratio=0.15):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=lw, arrow_length_ratio=arrow_ratio, zorder=10)
        if label:
            lpos = vec * label_offset
            ax.text(lpos[0], lpos[1], lpos[2], label, color=color, fontsize=fs, ha='center', va='center', fontweight='bold', zorder=15)

    # Drawing the key vectors
    draw_arrow(n_s, '#222222', r'$\hat{\mathbf{n}}_s$', label_offset=1.15)
    draw_arrow(f_axis, '#2ca02c', r'$\hat{\mathbf{f}}$', label_offset=1.2)

    # Bare / influence / reduced Bloch vectors
    draw_arrow(r0_vec, '#1f77b4', r'', lw=5, label_offset=1.1, arrow_ratio=0.2)
    ax.text(r0_vec[0], r0_vec[1], r0_vec[2]*1.22, r'$r_0$', color='#1f77b4', fontsize=24, ha='center', va='top', zorder=20)

    draw_arrow(rS_vec, '#9467bd', r'', lw=5, label_offset=1.08, arrow_ratio=0.17)
    ax.text(rS_vec[0]*1.25, 0, rS_vec[2]*1.20, r'$r_S$', color='#9467bd', fontsize=24, ha='left', va='bottom', zorder=20)

    draw_arrow(rQ_vec, '#d62728', r'', lw=5, label_offset=1.08, arrow_ratio=0.15)
    ax.text(rQ_vec[0]*1.20, 0, rQ_vec[2]*1.22, r'$r_Q$', color='#d62728', fontsize=24, ha='left', va='top', zorder=20)

    # Angle labels
    # varphi_S and varphi_Q are measured from the bare south axis (-z)
    vals_S = np.linspace(np.pi, np.pi - varphi_S, 40)
    vals_Q = np.linspace(np.pi, np.pi - varphi_Q, 40)
    ax.plot(0.38*np.sin(vals_S), 0, 0.38*np.cos(vals_S), color='#9467bd', alpha=0.9, lw=2.2, zorder=6)
    ax.plot(0.54*np.sin(vals_Q), 0, 0.54*np.cos(vals_Q), color='#d62728', alpha=0.9, lw=2.2, zorder=6)
    
    # Shift labels: phi_S towards south (0.35 offset), phi_Q towards vector (0.65 offset)
    ax.text(0.48*np.sin(np.pi - 0.35*varphi_S), 0, 0.48*np.cos(np.pi - 0.35*varphi_S), 
            r'$\varphi_S$', color='#9467bd', fontsize=21, ha='center', zorder=11)
    ax.text(0.68*np.sin(np.pi - 0.65*varphi_Q), 0, 0.68*np.cos(np.pi - 0.65*varphi_Q), 
            r'$\varphi_Q$', color='#d62728', fontsize=21, ha='center', zorder=11)

    # Relative angle between S and Q: moved inward and shortened to avoid arrow tips
    vals_rel = np.linspace(np.pi - varphi_Q + 0.08, np.pi - varphi_S - 0.08, 36)
    ax.plot(0.70*np.sin(vals_rel), 0, 0.70*np.cos(vals_rel), color='black', alpha=0.7, lw=2, zorder=6)
    ax.text(0.78*np.sin(np.pi - (varphi_S + varphi_Q)/2), 0, 0.78*np.cos(np.pi - (varphi_S + varphi_Q)/2),
            r'$\Delta\varphi$', fontsize=19, zorder=11)

    # Theta angle: arc between n_s (+z) and f_axis in the xz-plane
    vals_theta = np.linspace(np.pi/2, np.pi/2 - theta, 40)  # from +z down to f_axis
    arc_r = 0.45
    ax.plot(arc_r*np.cos(vals_theta), 0, arc_r*np.sin(vals_theta),
            color='#2ca02c', alpha=0.85, lw=2.2, zorder=6)
    theta_mid = np.pi/2 - theta/2
    ax.text(0.54*np.cos(theta_mid), 0, 0.54*np.sin(theta_mid),
            r'$\theta$', color='#2ca02c', fontsize=22, ha='center', va='center', zorder=12)

    # Axes limits and style
    scale_lim = 1.15
    ax.set_xlim([-scale_lim, scale_lim])
    ax.set_ylim([-scale_lim, scale_lim])
    ax.set_zlim([-scale_lim, scale_lim])
    ax.axis('off')
    
    # View angle: Tilted off-center for 3D perspective
    # azim=-90 is exactly face-on to xz. 
    # Rotating to -70 gives a ~20 degree tilt of the plane.
    # elev=12 looks slightly down onto the equator.
    ax.view_init(elev=12, azim=-70)
    
    plt.tight_layout()
    # Use script absolute path to find figures directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(os.path.join(script_dir, "../../manuscript/figures/"))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "hmf_bloch_3d_overview.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "hmf_bloch_3d_overview.png"), bbox_inches='tight', dpi=300)
    print(f"Saved upgraded 3D Bloch overview to {out_dir}")

if __name__ == "__main__":
    plot_bloch_3d_v2()
