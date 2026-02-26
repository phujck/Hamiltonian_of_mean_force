import numpy as np
import scipy.integrate as quad
import matplotlib.pyplot as plt

# Parameters
ALPHA = 1.0
OMEGA_C = 5.0
OMEGA_Q = 1.0
THETA = 3 * np.pi / 8 # The value used in the manuscript

def J_ohmic(w):
    return ALPHA * w * np.exp(-w / OMEGA_C)

# Re-using the logic from fig1_chi_theory.py for F_z and F_x
def _F_z(Omega, beta):
    b = float(beta); oq = float(OMEGA_Q); Om = float(Omega); a = b * oq / 2.0
    eps = abs(Om - oq)
    if eps < 1e-6 * oq:
        bOm2 = b * oq / 2.0
        sh = np.sinh(np.clip(bOm2, 1e-14, 500))
        ch_a = np.cosh(np.clip(a, 0, 500)); sh_a = np.sinh(np.clip(a, 0, 500))
        val = ch_a / sh * (b**2 / 4.0 * ch_a + b / (2.0 * oq) * sh_a) - b * np.cosh(np.clip(bOm2, 0, 500)) / sh * ch_a / oq
        return float(val)
    Om_plus = Om + oq; Om_minus = oq - Om
    bOm2 = b * Om / 2.0; sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500)); ch_Om = np.cosh(np.clip(bOm2, 0, 500))
    ch_a = np.cosh(np.clip(a, 0, 500)); sh_a = np.sinh(np.clip(a, 0, 500))
    denom = oq**2 - Om**2
    term1 = -b * oq * ch_Om / denom
    term2 = (ch_a / sh_Om) * (np.sinh(np.clip(Om_plus * b / 2, 0, 500)) / Om_plus**2 + np.sinh(np.clip(Om_minus * b / 2, 0, 500)) / Om_minus**2)
    return float(term1 + term2)

def _F_x(Omega, beta):
    b = float(beta); oq = float(OMEGA_Q); Om = float(Omega); a = b * oq / 2.0
    eps = abs(Om - oq); bOm2 = b * Om / 2.0; ch_a = np.cosh(np.clip(a, 0, 500)); sh_a = np.sinh(np.clip(a, 0, 500))
    if abs(Om) < 1e-12:
        part1 = 2.0 * ch_a**2 * b
    else:
        sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500))
        part1 = 2.0 * ch_a**2 * 2.0 * sh_Om / Om
    if eps < 1e-6 * oq:
        dOm = 1e-5 * oq
        return float((_F_x(oq + dOm, beta) + _F_x(oq - dOm, beta)) / 2.0)
    sh_Om = np.sinh(np.clip(bOm2, 1e-14, 500)); ch_Om = np.cosh(np.clip(bOm2, 0, 500)); denom = Om**2 - oq**2
    part2 = 4.0 * ch_a * (Om * sh_Om * ch_a - oq * ch_Om * sh_a) / denom
    return float(part1 - part2)

def get_direction(beta):
    s = np.sin(THETA)
    c = np.cos(THETA)
    
    def integrand_z(Om):
        return J_ohmic(Om) * _F_z(Om, beta)
    def integrand_x(Om):
        return J_ohmic(Om) * _F_x(Om, beta)
    
    dz0_val, _ = quad.quad(integrand_z, 0, 20.0) # Larger range for Ohmic
    sx0_val, _ = quad.quad(integrand_x, 0, 20.0)
    
    dz0 = (s**2 / np.pi) * dz0_val
    sx0 = (c * s / (np.pi * OMEGA_Q)) * sx0_val
    
    # n = (sx0, -dz0)
    angle = np.arctan2(sx0, -dz0) 
    return angle, dz0, sx0

# Coupling angle f
# f_hat = (sin(theta), cos(theta)) -> angle theta relative to z-axis?
# Or angle relative to x-axis?
# If we use arctan2(y, x), n_x = sx0, n_y = -dz0.
# The coupling angle in this coordinate system (x=r_perp, z=n_s)?
# Wait, f_hat in the plane is (sin(theta), cos(theta)).
# arctan2(sin(theta), cos(theta)) = theta.

f_angle = THETA
print(f"Coupling angle f: {f_angle:.6f} rad ({np.degrees(f_angle):.2f} deg)")

for b in [1000.0, 100.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01]:
    angle, dz0, sx0 = get_direction(b)
    print(f"beta={b:8.3f}: n_angle={angle:8.6f} rad ({np.degrees(angle):6.2f} deg), n_x={sx0:8.6f}, n_z={-dz0:8.6f}, ratio={sx0/(-dz0):8.6f}")

