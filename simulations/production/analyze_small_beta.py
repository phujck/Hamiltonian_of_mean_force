import numpy as np
import scipy.integrate as quad

OMEGA_Q = 1.0

def _F_z_small_beta(Omega, beta):
    # Taylor expansion for small beta
    # F_z = int_0^beta (beta-u) * cosh(Omega(u-b/2))/sinh(bO/2) * sinh(oq u) du
    # sinh(oq u) ~ oq u
    # cosh / sinh ~ (1)/(bO/2)
    # F_z ~ (2/bO) * oq * int_0^beta (beta-u) u du
    # int_0^beta (beta u - u^2) du = beta^3/2 - beta^3/3 = beta^3/6
    # F_z ~ (2/bO) * oq * beta^3/6 = beta^2 oq / (3 Omega)
    return (beta**2 * OMEGA_Q) / (3.0 * Omega)

def _F_x_small_beta(Omega, beta):
    # F_x = int_0^beta cosh(Omega(u-b/2))/sinh(bO/2) * [cosh(bo)+1 - cosh(ou) - cosh(o(b-u))] du
    # cosh(bo)+1 ~ 2 + (bo)^2/2
    # cosh(ou) ~ 1 + (ou)^2/2
    # [ ] ~ 2 + b^2o^2/2 - (1 + o^2u^2/2) - (1 + o^2(b-u)^2/2)
    #     = b^2o^2/2 - o^2/2 [ u^2 + (b-u)^2 ]
    #     = b^2o^2/2 - o^2/2 [ u^2 + b^2 - 2bu + u^2 ]
    #     = b^2o^2/2 - o^2/2 [ 2u^2 + b^2 - 2bu ]
    #     = b^2o^2/2 - o^2u^2 - o^2b^2/2 + o^2bu
    #     = o^2 (bu - u^2)
    # F_x ~ (2/bO) * o^2 * int_0^beta (bu - u^2) du
    #     = (2/bO) * o^2 * (beta^3/2 - beta^3/3) = (2/bO) * o^2 * beta^3/6
    #     = beta^2 o^2 / (3 Omega)
    return (beta**2 * OMEGA_Q**2) / (3.0 * Omega)

beta = 1e-4
for Om in [0.5, 1.0, 2.0]:
    fz = _F_z_small_beta(Om, beta)
    fx = _F_x_small_beta(Om, beta)
    print(f"Om={Om}, fz={fz:.2e}, fx={fx:.2e}, fx/fz={fx/fz:.2e}")

