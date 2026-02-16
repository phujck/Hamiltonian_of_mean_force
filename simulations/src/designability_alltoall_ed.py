"""
Exact finite-temperature baseline (small system sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from time import perf_counter
from typing import Dict, Sequence

import numpy as np
import psutil
from scipy.linalg import eigh

from designability_alltoall_models import AllToAllModel


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)
SIGMAS = (SIGMA_X, SIGMA_Y, SIGMA_Z)


@dataclass
class EDResult:
    method: str
    runtime_s: float
    peak_mem_mb: float
    energy_density: float
    mx: float
    my: float
    mz: float
    two_body: Dict[str, float]


@lru_cache(maxsize=None)
def _site_operator(n_sites: int, site: int, axis: int) -> np.ndarray:
    ops = []
    for idx in range(n_sites):
        if idx == site:
            ops.append(SIGMAS[axis])
        else:
            ops.append(IDENTITY_2)
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _build_hamiltonian(model: AllToAllModel) -> np.ndarray:
    n = model.n_sites
    dim = 2**n
    h = np.zeros((dim, dim), dtype=complex)

    for i in range(n):
        h = h + model.omega_x[i] * _site_operator(n, i, 0)
        h = h + model.omega_z[i] * _site_operator(n, i, 2)

    for term, j in zip(model.pair_terms, model.couplings):
        op_i = _site_operator(n, term.i, term.a)
        op_j = _site_operator(n, term.j, term.b)
        h = h + j * (op_i @ op_j)

    return 0.5 * (h + h.conj().T)


def _thermal_state(hamiltonian: np.ndarray, beta: float) -> np.ndarray:
    evals, evecs = eigh(hamiltonian)
    evals_shifted = evals - np.min(evals)
    weights = np.exp(-beta * evals_shifted)
    z = np.sum(weights)
    probs = weights / z
    rho = (evecs * probs) @ evecs.conj().T
    return 0.5 * (rho + rho.conj().T)


def _reduced_density_matrix(rho: np.ndarray, n_sites: int, keep: Sequence[int]) -> np.ndarray:
    keep_sorted = sorted(keep)
    keep_set = set(keep_sorted)
    trace_out = [idx for idx in range(n_sites) if idx not in keep_set]
    tensor = rho.reshape([2] * (2 * n_sites))
    current_n = n_sites
    for site in sorted(trace_out, reverse=True):
        tensor = np.trace(tensor, axis1=site, axis2=site + current_n)
        current_n -= 1
    d_keep = 2 ** len(keep_sorted)
    return tensor.reshape((d_keep, d_keep))


def run_exact_ed(model: AllToAllModel) -> EDResult:
    process = psutil.Process()
    mem0 = process.memory_info().rss
    t0 = perf_counter()

    h = _build_hamiltonian(model)
    rho = _thermal_state(h, model.beta)
    n = model.n_sites

    energy_density = float(np.real(np.trace(rho @ h)) / n)

    mags = np.zeros(3, dtype=float)
    for i in range(n):
        rho_i = _reduced_density_matrix(rho, n_sites=n, keep=[i])
        for a in range(3):
            mags[a] += float(np.real(np.trace(rho_i @ SIGMAS[a])))
    mags /= n

    n_pairs = n * (n - 1) // 2
    pair_ops = {(a, b): np.kron(SIGMAS[a], SIGMAS[b]) for a in range(3) for b in range(3)}
    pair_acc = {(a, b): 0.0 for a in range(3) for b in range(3)}
    for i, j in combinations(range(n), 2):
        rho_ij = _reduced_density_matrix(rho, n_sites=n, keep=[i, j])
        for a in range(3):
            for b in range(3):
                pair_acc[(a, b)] += float(np.real(np.trace(rho_ij @ pair_ops[(a, b)])))

    two_body = {}
    for a, a_lab in enumerate("xyz"):
        for b, b_lab in enumerate("xyz"):
            two_body[f"{a_lab}{b_lab}"] = pair_acc[(a, b)] / max(n_pairs, 1)

    runtime_s = perf_counter() - t0
    mem1 = process.memory_info().rss
    peak_mem_mb = max(mem0, mem1) / (1024.0 * 1024.0)

    return EDResult(
        method="ed",
        runtime_s=float(runtime_s),
        peak_mem_mb=float(peak_mem_mb),
        energy_density=energy_density,
        mx=float(mags[0]),
        my=float(mags[1]),
        mz=float(mags[2]),
        two_body=two_body,
    )

