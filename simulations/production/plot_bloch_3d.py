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

def plot_bloch_3d():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    # Parameters
    theta = np.pi/3
    phi_r = np.pi/4
    beta_wq = 4
    a = beta_wq / 2
    
    # Representative kernels
    Sigma_z = 0.8
    Sigma_perp = 1.2
    
    # Vectors
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Coupling vector r
    r_vec = np.array([s * np.cos(phi_r), s * np.sin(phi_r), c])
    
    # System axis n_s
    n_s = np.array([0, 0, 1.0])
    
    # Transverse coupling component r_perp_vec
    r_perp_vec = np.array([s * np.cos(phi_r), s * np.sin(phi_r), 0])
    
    # Calculate v in the coupling plane (phi_v = phi_r or phi_r + pi)
    # We use the previous logic but keep it rotated
    # Delta_z = s^2 * Sigma_z
    # Delta_perp = -c * s * Sigma_perp
    # Assume chi=1 for visualization scale
    chi = 1.0
    gamma = np.tanh(chi)/chi
    
    # h_eff = -Sigma_perp * c * r_perp_hat + Sigma_z * s^2 * n_s
    # where r_perp_hat = (cos phi, sin phi, 0)
    h_eff = -Sigma_perp * c * np.array([np.cos(phi_r), np.sin(phi_r), 0]) + Sigma_z * s**2 * n_s
    chi_val = np.linalg.norm(h_eff)
    h_eff = h_eff / chi_val * 0.8 # Scale for plot
    
    u_vec = gamma * h_eff
    
    # Recombine to v
    u_par = u_vec[2]
    u_perp_mag = np.linalg.norm(u_vec[:2])
    denom = np.cosh(a) - u_par * np.sinh(a)
    v_z = (u_par * np.cosh(a) - np.sinh(a)) / denom
    v_perp_mag = u_perp_mag / denom
    
    # v phase is phi_r (assuming kappa > 0) or phi_r + pi
    v_vec = np.array([v_perp_mag * np.cos(phi_r), v_perp_mag * np.sin(phi_r), v_z])

    # Sphere Surface (Ultra-Frosted)
    u_s, v_s = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x_s = np.cos(u_s)*np.sin(v_s)
    y_s = np.sin(u_s)*np.sin(v_s)
    z_s = np.cos(v_s)
    ax.plot_surface(x_s, y_s, z_s, color="skyblue", alpha=0.03, shade=True, zorder=-1)
    
    # Grid Rims
    t_rim = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(t_rim), np.sin(t_rim), 0, color='black', alpha=0.1, linewidth=0.5, zorder=0)
    ax.plot(np.cos(t_rim), 0, np.sin(t_rim), color='black', alpha=0.05, linewidth=0.5, zorder=0)
    ax.plot(0, np.cos(t_rim), np.sin(t_rim), color='black', alpha=0.05, linewidth=0.5, zorder=0)

    # THE COUPLING PLANE (The restricted arena)
    # This is a vertical circular disk at angle phi_r
    r_range = np.linspace(0, 1.0, 10)
    theta_range = np.linspace(0, 2*np.pi, 40)
    R, T = np.meshgrid(r_range, theta_range)
    X_p = R * np.cos(phi_r) * np.sin(T)
    Y_p = R * np.sin(phi_r) * np.sin(T)
    Z_p = R * np.cos(T)
    # Plot the half-disk containing r and n_s more prominently? 
    # Actually a full disk shows the 'cut' through the sphere.
    ax.plot_surface(X_p, Y_p, Z_p, color='gray', alpha=0.07, shade=False, zorder=1)
    # The rim of the coupling plane
    ax.plot(np.sin(t_rim)*np.cos(phi_r), np.sin(t_rim)*np.sin(phi_r), np.cos(t_rim), color='black', alpha=0.3, lw=0.8, zorder=2)

    # Drawing functions
    def draw_arrow(vec, color, label, lw=4.5, label_offset=1.1, fs=26):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=lw, arrow_length_ratio=0.15, zorder=10)
        lpos = vec * label_offset
        ax.text(lpos[0], lpos[1], lpos[2], label, color=color, fontsize=fs, ha='center', va='center', fontweight='bold', zorder=15)

    draw_arrow(n_s, '#222222', r'$\hat{\mathbf{n}}_s$', label_offset=1.15)
    draw_arrow(r_vec, '#2ca02c', r'$\hat{\mathbf{f}}$', label_offset=1.2)
    draw_arrow(v_vec, '#d62728', r'$\mathbf{r}$', label_offset=1.25)

    # Angle labels (inside the plane)
    theta_vals = np.linspace(0, theta, 30)
    ax.plot(0.5*np.sin(theta_vals)*np.cos(phi_r), 0.5*np.sin(theta_vals)*np.sin(phi_r), 0.5*np.cos(theta_vals), color='black', alpha=0.7, lw=2, zorder=5)
    ax.text(0.6*np.sin(theta/2)*np.cos(phi_r), 0.6*np.sin(theta/2)*np.sin(phi_r), 0.6*np.cos(theta/2), r'$\theta$', fontsize=24, zorder=10)

    # (Phi angle and labels dropped as requested)

    # Axes limits and style
    scale_lim = 1.15
    ax.set_xlim([-scale_lim, scale_lim])
    ax.set_ylim([-scale_lim, scale_lim])
    ax.set_zlim([-scale_lim, scale_lim])
    ax.axis('off')
    
    # View angle - Dramatic oblique view to emphasize planarity
    ax.view_init(elev=18, azim=5)
    
    plt.tight_layout()
    out_dir = "../../manuscript/figures/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "hmf_bloch_3d_overview.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "hmf_bloch_3d_overview.png"), bbox_inches='tight', dpi=300)
    print("Saved 3D Bloch overview.")

if __name__ == "__main__":
    plot_bloch_3d()
