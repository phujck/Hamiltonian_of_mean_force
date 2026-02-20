"""
Core utilities for v5 qubit regime simulations.

This module provides:
- ED context/state evaluation reusing benchmark operators,
- compact v5 channel construction from response kernels,
- analytic reduced-state constructor,
- gauge-fixed Bloch/coherence observables,
- finite-difference susceptibility helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh

from prl127_qubit_benchmark import (
    BenchmarkConfig,
    IDENTITY_2,
    SIGMA_X,
    SIGMA_Z,
    build_static_operators,
    partial_trace_bath,
    thermal_state,
    trace_distance,
)

SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


@dataclass
class EDContext:
    config: BenchmarkConfig
    h_static: np.ndarray
    h_int: np.ndarray
    dim_system: int
    dim_bath: int


@dataclass
class V5BaseChannels:
    k0_zero: float
    k0_plus: float
    k0_minus: float
    r0_plus: float
    r0_minus: float
    sigma_plus0: float
    sigma_minus0: float
    delta_z0: float
    chi0: float
    kms_residual: float


@dataclass
class V5CouplingChannels:
    sigma_plus: float
    sigma_minus: float
    delta_z: float
    chi: float
    gamma: float


def build_ed_context(config: BenchmarkConfig) -> EDContext:
    h_static, h_int, _, h_s, _ = build_static_operators(config)
    dim_system = h_s.shape[0]
    dim_bath = h_static.shape[0] // dim_system
    return EDContext(
        config=config,
        h_static=h_static,
        h_int=h_int,
        dim_system=dim_system,
        dim_bath=dim_bath,
    )


def exact_reduced_state(ctx: EDContext, g: float) -> np.ndarray:
    # v5 comparison uses H_tot = H_S + H_B + g H_I (no counterterm term).
    # For the qubit coupling X = c sigma_z - s sigma_x, X^2 = I, so the dropped
    # lambda^2 Q X^2 contribution is a scalar energy shift only.
    h_tot = ctx.h_static + g * ctx.h_int
    rho_tot = thermal_state(h_tot, ctx.config.beta)
    rho_q = partial_trace_bath(rho_tot, dim_system=ctx.dim_system, dim_bath=ctx.dim_bath)
    return 0.5 * (rho_q + rho_q.conj().T)


def _discrete_bath(config: BenchmarkConfig) -> Tuple[np.ndarray, float, np.ndarray]:
    if config.n_modes == 1:
        omegas = np.array([0.5 * (config.omega_min + config.omega_max)], dtype=float)
        delta_omega = float(config.omega_max - config.omega_min)
    else:
        omegas = np.linspace(config.omega_min, config.omega_max, config.n_modes, dtype=float)
        delta_omega = float(omegas[1] - omegas[0])
    j0 = config.q_strength * config.tau_c * omegas * np.exp(-config.tau_c * omegas)
    return omegas, delta_omega, j0


def kernel_profile_base(config: BenchmarkConfig, u_abs: np.ndarray) -> np.ndarray:
    """
    Base kernel profile K0(u) for g=1, using the same bath discretization as ED.
    """
    beta = config.beta
    omegas, delta_omega, j0 = _discrete_bath(config)
    g2 = np.maximum(j0, 0.0) * delta_omega

    kernel = np.zeros_like(u_abs, dtype=float)
    for omega_k, g2_k in zip(omegas, g2):
        denom = np.sinh(0.5 * beta * omega_k)
        if abs(denom) < 1e-14:
            kernel += (2.0 * g2_k) / max(beta * omega_k, 1e-14)
        else:
            kernel += g2_k * np.cosh(omega_k * (0.5 * beta - u_abs)) / denom
    return kernel


def _integration_grid(beta: float, n_kernel_grid: int) -> np.ndarray:
    n_points = max(401, int(n_kernel_grid))
    if n_points % 2 == 0:
        n_points += 1
    return np.linspace(0.0, beta, n_points, dtype=float)


def laplace_k0(config: BenchmarkConfig, omega: float, n_kernel_grid: int) -> float:
    u = _integration_grid(config.beta, n_kernel_grid)
    k0 = kernel_profile_base(config, u)
    return float(np.trapezoid(k0 * np.exp(omega * u), u))


def resonant_r0(config: BenchmarkConfig, omega: float, n_kernel_grid: int) -> float:
    u = _integration_grid(config.beta, n_kernel_grid)
    k0 = kernel_profile_base(config, u)
    return float(np.trapezoid((config.beta - u) * k0 * np.exp(omega * u), u))


def compute_v5_base_channels(config: BenchmarkConfig, n_kernel_grid: int = 4001) -> V5BaseChannels:
    if abs(config.omega_q) < 1e-14:
        raise ValueError("omega_q must be nonzero for v5 channel construction.")

    c = float(np.cos(config.theta))
    s = float(np.sin(config.theta))
    beta = float(config.beta)
    omega_q = float(config.omega_q)

    k0_zero = laplace_k0(config, 0.0, n_kernel_grid)
    k0_plus = laplace_k0(config, omega_q, n_kernel_grid)
    k0_minus = laplace_k0(config, -omega_q, n_kernel_grid)
    r0_plus = resonant_r0(config, omega_q, n_kernel_grid)
    r0_minus = resonant_r0(config, -omega_q, n_kernel_grid)

    sigma_plus0 = (c * s / omega_q) * ((1.0 + np.exp(beta * omega_q)) * k0_zero - 2.0 * k0_plus)
    sigma_minus0 = (c * s / omega_q) * ((1.0 + np.exp(-beta * omega_q)) * k0_zero - 2.0 * k0_minus)
    delta_z0 = (s**2) * 0.5 * (r0_plus - r0_minus)

    chi0_sq = float(delta_z0 * delta_z0 + sigma_plus0 * sigma_minus0)
    chi0 = float(np.sqrt(max(chi0_sq, 0.0)))
    kms_residual = float(sigma_plus0 - np.exp(beta * omega_q) * sigma_minus0)

    return V5BaseChannels(
        k0_zero=float(k0_zero),
        k0_plus=float(k0_plus),
        k0_minus=float(k0_minus),
        r0_plus=float(r0_plus),
        r0_minus=float(r0_minus),
        sigma_plus0=float(sigma_plus0),
        sigma_minus0=float(sigma_minus0),
        delta_z0=float(delta_z0),
        chi0=chi0,
        kms_residual=kms_residual,
    )


def gamma_of_chi(chi: float) -> float:
    if abs(chi) < 1e-8:
        chi2 = chi * chi
        return float(1.0 - chi2 / 3.0 + 2.0 * chi2 * chi2 / 15.0)
    return float(np.tanh(chi) / chi)


def coupling_channels(base: V5BaseChannels, g: float) -> V5CouplingChannels:
    g2 = float(g * g)
    sigma_plus = g2 * base.sigma_plus0
    sigma_minus = g2 * base.sigma_minus0
    delta_z = g2 * base.delta_z0
    chi_sq = float(delta_z * delta_z + sigma_plus * sigma_minus)
    chi = float(np.sqrt(max(chi_sq, 0.0)))
    gamma = gamma_of_chi(chi)
    return V5CouplingChannels(
        sigma_plus=sigma_plus,
        sigma_minus=sigma_minus,
        delta_z=delta_z,
        chi=chi,
        gamma=gamma,
    )


def _project_density(rho: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (rho + rho.conj().T)
    evals, evecs = eigh(hermitian)
    evals = np.clip(np.real(evals), 0.0, None)
    if float(np.sum(evals)) <= 0.0:
        return 0.5 * IDENTITY_2.copy()
    rho_psd = (evecs * evals) @ evecs.conj().T
    rho_psd /= np.trace(rho_psd)
    return 0.5 * (rho_psd + rho_psd.conj().T)


def v5_theory_state(config: BenchmarkConfig, channels: V5CouplingChannels) -> np.ndarray:
    a = 0.5 * config.beta * config.omega_q
    gamma = channels.gamma
    delta_z = channels.delta_z
    sigma_plus = channels.sigma_plus
    sigma_minus = channels.sigma_minus

    z_q = 2.0 * (np.cosh(a) - gamma * delta_z * np.sinh(a))
    if abs(z_q) < 1e-14:
        return 0.5 * IDENTITY_2.copy()

    rho = np.array(
        [
            [np.exp(-a) * (1.0 + gamma * delta_z), np.exp(-a) * gamma * sigma_plus],
            [np.exp(a) * gamma * sigma_minus, np.exp(a) * (1.0 - gamma * delta_z)],
        ],
        dtype=complex,
    )
    rho /= z_q
    return _project_density(rho)


def apply_real_gauge(rho: np.ndarray) -> Tuple[np.ndarray, float]:
    rho01 = complex(rho[0, 1])
    phase = float(np.angle(rho01))
    u_z = np.array(
        [
            [np.exp(-0.5j * phase), 0.0],
            [0.0, np.exp(0.5j * phase)],
        ],
        dtype=complex,
    )
    rho_rot = u_z @ rho @ u_z.conj().T
    if np.real(rho_rot[0, 1]) < 0.0:
        phase += np.pi
        u_z = np.array(
            [
                [np.exp(-0.5j * phase), 0.0],
                [0.0, np.exp(0.5j * phase)],
            ],
            dtype=complex,
        )
        rho_rot = u_z @ rho @ u_z.conj().T
    rho_rot = 0.5 * (rho_rot + rho_rot.conj().T)
    return rho_rot, phase


def bloch_components(rho: np.ndarray) -> Tuple[float, float, float]:
    mx = float(np.real(np.trace(rho @ SIGMA_X)))
    my = float(np.real(np.trace(rho @ SIGMA_Y)))
    mz = float(np.real(np.trace(rho @ SIGMA_Z)))
    return mx, my, mz


def state_observables(rho: np.ndarray) -> Dict[str, float]:
    rho_gauge, phase = apply_real_gauge(rho)
    mx, my, mz = bloch_components(rho_gauge)
    if abs(mx) < 1e-14:
        mx = 0.0
    if abs(my) < 1e-14:
        my = 0.0
    if abs(mz) < 1e-14:
        mz = 0.0

    phi = float(np.arctan2(mx, mz))
    if phi < 0.0:
        phi += np.pi
    r = float(np.sqrt(max(mx * mx + my * my + mz * mz, 0.0)))
    coherence = float(2.0 * abs(rho[0, 1]))
    rho01_abs = float(abs(rho[0, 1]))

    return {
        "mx": mx,
        "my": my,
        "mz": mz,
        "phi": phi,
        "r": r,
        "coherence": coherence,
        "rho01_abs": rho01_abs,
        "gauge_phase": phase,
    }


def central_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([0.0], dtype=float)

    out = np.zeros(n, dtype=float)
    out[0] = (y[1] - y[0]) / max(x[1] - x[0], 1e-14)
    out[-1] = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-14)
    for i in range(1, n - 1):
        out[i] = (y[i + 1] - y[i - 1]) / max(x[i + 1] - x[i - 1], 1e-14)
    return out


def state_trace_distance(rho_ed: np.ndarray, rho_th: np.ndarray) -> float:
    return float(trace_distance(rho_ed, rho_th))
