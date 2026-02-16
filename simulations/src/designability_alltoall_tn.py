"""
Tensor-network baseline using quimb MPO + DMRG.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Dict, Sequence

import numpy as np
import psutil
import quimb.tensor as qtn

from designability_alltoall_models import AllToAllModel


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)
SIGMAS = (SIGMA_X, SIGMA_Y, SIGMA_Z)


@dataclass
class TNResult:
    method: str
    runtime_s: float
    peak_mem_mb: float
    energy_density: float
    mx: float
    my: float
    mz: float
    two_body: Dict[str, float]
    converged: bool
    max_bond_observed: int


def _pure_rdm(psi_vec: np.ndarray, n_sites: int, keep: Sequence[int]) -> np.ndarray:
    keep_sorted = sorted(keep)
    other = [i for i in range(n_sites) if i not in keep_sorted]
    perm = keep_sorted + other
    psi_tensor = psi_vec.reshape([2] * n_sites)
    psi_perm = np.transpose(psi_tensor, perm)
    d_keep = 2 ** len(keep_sorted)
    psi_mat = psi_perm.reshape(d_keep, -1)
    rho = psi_mat @ psi_mat.conj().T
    return 0.5 * (rho + rho.conj().T)


def _build_mpo(model: AllToAllModel, tn_max_bond: int, cutoff: float, compress_every: int = 128) -> qtn.MatrixProductOperator:
    n = model.n_sites
    mpo = qtn.MPO_zeros(n, phys_dim=2, dtype="complex128")

    # one-body fields
    for i in range(n):
        if abs(model.omega_x[i]) > 0.0:
            arrays = [IDENTITY_2] * n
            arrays[i] = SIGMA_X
            mpo = mpo + float(model.omega_x[i]) * qtn.MPO_product_operator(arrays)
        if abs(model.omega_z[i]) > 0.0:
            arrays = [IDENTITY_2] * n
            arrays[i] = SIGMA_Z
            mpo = mpo + float(model.omega_z[i]) * qtn.MPO_product_operator(arrays)

    for idx, (term, j) in enumerate(zip(model.pair_terms, model.couplings), start=1):
        if abs(j) < 1e-15:
            continue
        arrays = [IDENTITY_2] * n
        arrays[term.i] = SIGMAS[term.a]
        arrays[term.j] = SIGMAS[term.b]
        mpo = mpo + float(j) * qtn.MPO_product_operator(arrays)
        if idx % compress_every == 0:
            mpo.compress(max_bond=tn_max_bond, cutoff=cutoff)

    mpo.compress(max_bond=tn_max_bond, cutoff=cutoff)
    return mpo


def run_tn_dmrg(
    model: AllToAllModel,
    tn_max_bond: int,
    cutoff: float,
    max_sweeps: int = 8,
    compute_full_observables: bool = False,
) -> TNResult:
    process = psutil.Process()
    mem0 = process.memory_info().rss
    t0 = perf_counter()

    mpo = _build_mpo(model, tn_max_bond=tn_max_bond, cutoff=cutoff)
    dmrg = qtn.DMRG2(
        mpo,
        bond_dims=[max(16, tn_max_bond // 2), tn_max_bond],
        cutoffs=cutoff,
    )
    converged = bool(dmrg.solve(max_sweeps=max_sweeps, tol=1e-5, verbosity=0))
    energy_density = float(np.real(dmrg.energy) / model.n_sites)
    max_bond_observed = int(dmrg.state.max_bond())

    mx = np.nan
    my = np.nan
    mz = np.nan
    two_body = {f"{a}{b}": np.nan for a in "xyz" for b in "xyz"}

    if compute_full_observables:
        psi_dense = np.asarray(dmrg.state.to_dense()).reshape(-1)
        n = model.n_sites

        mags = np.zeros(3, dtype=float)
        for i in range(n):
            rho_i = _pure_rdm(psi_dense, n_sites=n, keep=[i])
            for a in range(3):
                mags[a] += float(np.real(np.trace(rho_i @ SIGMAS[a])))
        mags /= n
        mx, my, mz = map(float, mags)

        n_pairs = n * (n - 1) // 2
        pair_ops = {(a, b): np.kron(SIGMAS[a], SIGMAS[b]) for a in range(3) for b in range(3)}
        pair_acc = {(a, b): 0.0 for a in range(3) for b in range(3)}
        for i, j in combinations(range(n), 2):
            rho_ij = _pure_rdm(psi_dense, n_sites=n, keep=[i, j])
            for a in range(3):
                for b in range(3):
                    pair_acc[(a, b)] += float(np.real(np.trace(rho_ij @ pair_ops[(a, b)])))
        for a, a_lab in enumerate("xyz"):
            for b, b_lab in enumerate("xyz"):
                two_body[f"{a_lab}{b_lab}"] = pair_acc[(a, b)] / max(n_pairs, 1)

    runtime_s = perf_counter() - t0
    mem1 = process.memory_info().rss
    peak_mem_mb = max(mem0, mem1) / (1024.0 * 1024.0)

    return TNResult(
        method="tn_dmrg",
        runtime_s=float(runtime_s),
        peak_mem_mb=float(peak_mem_mb),
        energy_density=energy_density,
        mx=float(mx),
        my=float(my),
        mz=float(mz),
        two_body=two_body,
        converged=converged,
        max_bond_observed=max_bond_observed,
    )
