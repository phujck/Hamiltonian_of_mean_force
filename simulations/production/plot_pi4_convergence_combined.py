
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# -- STYLE --
mpl.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.usetex": True, "figure.dpi": 250, "lines.linewidth": 1.1,
    "axes.linewidth": 0.7,
})

PROD_DIR = Path(__file__).parent
DATA_FILE = PROD_DIR / "data" / "pi4_ncut_convergence.csv"

# -- PHYSICAL CONSTANTS --
OMEGA_Q = 2.0
THETA = np.pi/4
Q_STRENGTH = 10.0
TAU_C = 1.0
OMIN = 0.5
OMAX = 8.0

def calculate_hmf_rho_robust(g, beta, n_modes=1000, renorm=None):
    """
    Robust HMF normalization using scaled integrals to prevent overflow.
    """
    oms = np.linspace(OMIN, OMAX, n_modes)
    d_om = (OMAX - OMIN) / (n_modes - 1)
    g2k = Q_STRENGTH * TAU_C * oms * np.exp(-TAU_C * oms) * d_om
    b = beta
    
    # Scale integrals by e^(beta*omega_q)
    W = np.exp(b * OMEGA_Q)
    
    def mode_lap_scaled(wk, s):
        # Integral Int_0^b 0.5(exp(wk(b/2-t)) + exp(-wk(b/2-t))) exp(st) dt / [sinh(b*wk/2) * W]
        # Leading term in Int is exp((s + wk/2)*b) or exp(wk*b/2)
        c1, c2 = s - wk, s + wk
        t1 = (np.exp(c1*b)-1.0)/c1 if abs(c1)>1e-11 else b
        t2 = (np.exp(c2*b)-1.0)/c2 if abs(c2)>1e-11 else b
        den = (np.exp(wk*b*0.5) - np.exp(-wk*b*0.5))
        # Scaled result:
        res = (np.exp(wk*b*0.5)*t1 + np.exp(-wk*b*0.5)*t2) / max(den, 1e-25)
        return res / W

    def mode_res_scaled(wk, s):
        def d_term(c):
            if abs(c) < 1e-11: return 0.5 * b**2
            return (b * np.exp(c * b)) / c - (np.exp(c * b) - 1.0) / (c*c)
        c1, c2 = s - wk, s + wk
        den = (np.exp(wk*b*0.5) - np.exp(-wk*b*0.5))
        res = (np.exp(wk*b*0.5)*d_term(c1) + np.exp(-wk*b*0.5)*d_term(c2)) / max(den, 1e-25)
        return res / W

    k00_s, k0p_s, k0m_s = 0.0, 0.0, 0.0
    r0p_s, r0m_s = 0.0, 0.0
    for wk, g_k in zip(oms, g2k):
        # We also need K(0) scaled by W?
        # No, just scale everything by W.
        k00_s += g_k * mode_lap_scaled(wk, 0.0)
        k0p_s += g_k * mode_lap_scaled(wk, OMEGA_Q)
        k0m_s += g_k * mode_lap_scaled(wk, -OMEGA_Q)
        r0p_s += g_k * mode_res_scaled(wk, OMEGA_Q)
        r0m_s += g_k * mode_res_scaled(wk, -OMEGA_Q)

    c, s = np.cos(THETA), np.sin(THETA)
    # sigma_plus = cs/w * [ (1+W) k00_s*W - 2 k0p_s*W ] / W? No.
    # Sigma_plus / W = (cs/w) * [ (1/W + 1) k00_s - 2 k0p_s ]
    sp_s = (c * s / OMEGA_Q) * ((1.0/W + 1.0) * k00_s - 2.0 * k0p_s)
    sm_s = (c * s / OMEGA_Q) * ((1.0/W + np.exp(-2*b*OMEGA_Q)) * k00_s - 2.0 * k0m_s)
    dz_s = s**2 * 0.5 * (r0p_s - r0m_s)
    
    # chi_scaled = sqrt(dz_s^2 + sp_s*sm_s)
    # chi = g^2 * chi_scaled * W
    chi_sc = np.sqrt(max(dz_s**2 + sp_s*sm_s, 1e-30))
    chi_raw = g*g * chi_sc * W
    
    run = 1.0
    if renorm:
        a_s = 0.5 * b * OMEGA_Q
        chi_cap = max(renorm.get('kappa', 0.94) * abs(a_s), 1e-6)
        run = 1.0 / (1.0 + chi_raw / chi_cap)
    
    chi = run * chi_raw
    # dz/chi is independent of g and W: dz/chi = dz_s / chi_sc
    # gamma * dz = [tanh(chi)/chi] * dz = tanh(chi) * (dz_s / chi_sc)
    direction_z = dz_s / chi_sc
    gamma_dz = np.tanh(chi) * direction_z
    gamma_dz = np.clip(gamma_dz, -1.0 + 1e-15, 1.0 - 1e-15)
    
    # p11 = (1 - gamma_dz) / [ (1 - gamma_dz) + exp(-2a)(1 + gamma_dz) ]
    a = 0.5 * b * OMEGA_Q
    t1 = 1.0 - gamma_dz
    t0 = np.exp(-2.0 * a) * (1.0 + gamma_dz)
    return t1 / (t1 + t0 + 1e-30)

def us_limit_fn(beta):
    a = 0.5 * beta * OMEGA_Q
    c = np.cos(THETA)
    return 0.5 * (1.0 + c * np.tanh(a * c))

df = pd.read_csv(DATA_FILE)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.18)

lines = [
    (2, 60, "2m, 60c", "#4daf4a"), 
    (3, 15, "3m, 15c", "#377eb8"),
    (4, 6, "4m, 6c", "#ff7f00"),
]

# (a) Thermal Sweep
sub_g = df[np.isclose(df['g'], 0.05)].copy()
for n_m, n_c, lbl, col in lines:
    sub = sub_g[(sub_g['n_modes']==n_m) & (sub_g['n_cut']==n_c)].sort_values("beta")
    if not sub.empty:
        ax1.plot(sub["beta"], sub["ed_p11"], label=lbl, color=col, lw=0.75, marker='.', ms=3, alpha=0.9)

b_gr = np.linspace(0.3, 10.0, 50)
p11_ex = [calculate_hmf_rho_robust(0.05, b, renorm=None) for b in b_gr]
ax1.plot(b_gr, p11_ex, 'k--', label="Exact HMF", zorder=10)
p11_us = [us_limit_fn(b) for b in b_gr]
ax1.plot(b_gr, p11_us, 'gray', ls=':', lw=0.9, label="US Limit", alpha=0.8)

ax1.set_xlabel(r"Inv. Temp. $\beta$")
ax1.set_ylabel(r"Population $p_{11}$")
ax1.set_title(r"(a) Thermal Convergence ($g=0.05$)")
ax1.legend(loc="lower right", ncol=2)
ax1.grid(alpha=0.1, ls=":")

# (b) Coupling Sweep
TARGET_BETA = 1.972
sub_b = df[np.isclose(df['beta'], TARGET_BETA, atol=0.05)].copy()
for n_m, n_c, lbl, col in lines:
    sub = sub_b[(sub_b['n_modes']==n_m) & (sub_b['n_cut']==n_c)].sort_values("g")
    if not sub.empty:
        ax2.plot(sub["g"], sub["ed_p11"], label=lbl, color=col, lw=0.75, marker='.', ms=3, alpha=0.9)

g_gr = np.linspace(0.0, 1.8, 50)
p11_raw_b = [calculate_hmf_rho_robust(g, TARGET_BETA, renorm=None) for g in g_gr]
p11_run_b = [calculate_hmf_rho_robust(g, TARGET_BETA, renorm={'kappa': 0.94}) for g in g_gr]
us_l = us_limit_fn(TARGET_BETA)

ax2.plot(g_gr, p11_raw_b, 'gray', ls='--', lw=0.8, label="HMF (Raw)", alpha=0.6)
ax2.plot(g_gr, p11_run_b, 'k--', label="HMF (Running)", zorder=10)
ax2.axhline(us_l, color="#AA3377", ls=":", lw=1.2, label=f"US Limit ({us_l:.3f})")

ax2.set_xlabel(r"Coupling $g$")
ax2.set_ylabel(r"Population $p_{11}$")
ax2.set_title(rf"(b) Coupling Convergence ($\beta \approx {TARGET_BETA:.1f}$)")
ax2.legend(loc="best", ncol=2)
ax2.grid(alpha=0.1, ls=":")

plt.savefig("../../manuscript/figures/hmf_fig6_convergence_details.png", dpi=300)
plt.savefig("../../manuscript/figures/hmf_fig6_convergence_details.pdf")
print("Saved final convergence figure with robust scaling.")
