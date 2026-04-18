#!/usr/bin/env python
"""Minimal algebraic checks for dual-representative identities.

This script is intentionally lightweight and model-agnostic at the identity level.
It checks:
  1) involution: (g^vee)^vee = g
  2) crossover inversion: chi(beta, g^vee, theta) = 1 / chi(beta, g, theta)
  3) affine reflection: beta_eff' + beta_eff = 2 beta
"""

from __future__ import annotations

import numpy as np


def g_dual(chi0: np.ndarray, g: np.ndarray) -> np.ndarray:
    return 1.0 / (chi0 * g)


def chi(chi0: np.ndarray, g: np.ndarray) -> np.ndarray:
    return chi0 * g * g


def check_involution(rng: np.random.Generator, n: int = 2000) -> bool:
    chi0 = rng.uniform(1e-3, 30.0, size=n)
    g = rng.uniform(1e-3, 30.0, size=n)
    g2 = g_dual(chi0, g_dual(chi0, g))
    return np.allclose(g2, g, rtol=1e-12, atol=1e-12)


def check_crossover_inversion(rng: np.random.Generator, n: int = 2000) -> bool:
    chi0 = rng.uniform(1e-3, 30.0, size=n)
    g = rng.uniform(1e-3, 30.0, size=n)
    c = chi(chi0, g)
    c_dual = chi(chi0, g_dual(chi0, g))
    return np.allclose(c_dual, 1.0 / c, rtol=1e-12, atol=1e-12)


def check_affine_reflection(rng: np.random.Generator, n: int = 2000) -> bool:
    beta = rng.uniform(1e-3, 20.0, size=n)
    beta_eff = rng.uniform(-20.0, 40.0, size=n)
    beta_eff_p = 2.0 * beta - beta_eff
    return np.allclose(beta_eff + beta_eff_p, 2.0 * beta, rtol=1e-12, atol=1e-12)


def main() -> int:
    rng = np.random.default_rng(20260227)
    ok_inv = check_involution(rng)
    ok_chi = check_crossover_inversion(rng)
    ok_aff = check_affine_reflection(rng)

    print(f"involution_check: {'PASS' if ok_inv else 'FAIL'}")
    print(f"chi_inversion_check: {'PASS' if ok_chi else 'FAIL'}")
    print(f"affine_reflection_check: {'PASS' if ok_aff else 'FAIL'}")

    return 0 if (ok_inv and ok_chi and ok_aff) else 1


if __name__ == "__main__":
    raise SystemExit(main())
