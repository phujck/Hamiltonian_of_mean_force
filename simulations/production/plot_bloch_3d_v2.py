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

    # Parameters for visualization (Exaggerated for clarity)
    theta = np.pi/3     # Exaggerated theta
    phi_f = 0           
    beta_wq = 1.3       # Lowered beta to make bare vector shorter
    r_bare_mag = np.tanh(beta_wq / 2)
    
    # Dressed parameters
    r_dressed_mag = 0.98 # Exaggerated length
    phi_tilt = np.pi/4   # Exaggerated tilde phi
    
    # Vectors (n_s is +z)
    n_s = np.array([0, 0, 1.0])
    
    # Bare state vector (pointing South)
    r_bare_vec = np.array([0, 0, -r_bare_mag])
    
    # Interaction axis f
    f_axis = np.array([np.sin(theta), 0, np.cos(theta)])
    
    # Dressed state vector
    r_dressed_vec = r_dressed_mag * np.array([np.sin(phi_tilt), 0, -np.cos(phi_tilt)])

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
    
    # Bare state vector (South)
    draw_arrow(r_bare_vec, '#1f77b4', r'', lw=5, label_offset=1.1, arrow_ratio=0.2)
    ax.text(r_bare_vec[0], r_bare_vec[1], r_bare_vec[2]*1.2, r'$r_{\rm bare}(\beta)$', color='#1f77b4', fontsize=24, ha='center', va='top', zorder=20)
    
    # Dressed state vector
    draw_arrow(r_dressed_vec, '#d62728', r'', lw=5, label_offset=1.1, arrow_ratio=0.15)
    ax.text(r_dressed_vec[0]*1.1, r_dressed_vec[1], r_dressed_vec[2]*1.1, r'$r(\beta) = r_{\rm bare}(\tilde{\beta})$', color='#d62728', fontsize=24, ha='left', va='top', zorder=20)

    # Angle labels
    # 2. tilde phi (between bare and dressed)
    phi_vals = np.linspace(np.pi, np.pi - phi_tilt, 30)
    ax.plot(0.5*np.sin(phi_vals), 0, 0.5*np.cos(phi_vals), color='black', alpha=0.7, lw=2, zorder=5)
    ax.text(0.6*np.sin(np.pi - phi_tilt/2), 0, 0.6*np.cos(np.pi - phi_tilt/2), r'$\tilde{\varphi}$', fontsize=24, zorder=10)

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
