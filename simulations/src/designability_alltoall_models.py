"""
All-to-all 2-local model construction utilities.

The free-spin core is:
    H_S = sum_i (Omega_i^x X_i + Omega_i^z Z_i)

Target interacting Hamiltonian family:
    H_target = H_S + sum_{i<j} sum_{a,b in {x,y,z}} J_{ij}^{ab} sigma_i^a sigma_j^b

For the pair-resolved channel basis, each coefficient J_{ij}^{ab} is directly
controlled by one channel parameter, so the design matrix is the identity and
has full rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


PAULI_LABELS: Tuple[str, str, str] = ("x", "y", "z")
PAULI_TO_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(PAULI_LABELS)}


@dataclass(frozen=True)
class PairTerm:
    """Single 2-local Pauli-pair coefficient descriptor."""

    i: int
    j: int
    a: int
    b: int


@dataclass
class AllToAllModel:
    """Container for model data at a fixed system size."""

    n_sites: int
    beta: float
    omega_x: np.ndarray
    omega_z: np.ndarray
    pair_terms: List[PairTerm]
    couplings: np.ndarray

    @property
    def n_pairs(self) -> int:
        return self.n_sites * (self.n_sites - 1) // 2

    @property
    def n_two_body_coeffs(self) -> int:
        return len(self.pair_terms)

    def coupling_tensor(self) -> np.ndarray:
        """
        Returns J tensor with shape (n, n, 3, 3).
        Only i<j entries are populated by construction.
        """
        out = np.zeros((self.n_sites, self.n_sites, 3, 3), dtype=float)
        for term, value in zip(self.pair_terms, self.couplings):
            out[term.i, term.j, term.a, term.b] = value
        return out


def enumerate_pair_terms(n_sites: int) -> List[PairTerm]:
    terms: List[PairTerm] = []
    for i, j in combinations(range(n_sites), 2):
        for a in range(3):
            for b in range(3):
                terms.append(PairTerm(i=i, j=j, a=a, b=b))
    return terms


def sample_model(
    n_sites: int,
    beta: float,
    rng: np.random.Generator,
    field_scale: float = 0.35,
    coupling_scale: float = 0.08,
) -> AllToAllModel:
    """
    Sample a dense all-to-all 2-local target model.

    Fields are restricted to X/Z, consistent with the free-spin core.
    """
    pair_terms = enumerate_pair_terms(n_sites)
    omega_x = rng.normal(loc=0.0, scale=field_scale, size=n_sites)
    omega_z = rng.normal(loc=0.0, scale=field_scale, size=n_sites)
    couplings = rng.normal(loc=0.0, scale=coupling_scale, size=len(pair_terms))
    return AllToAllModel(
        n_sites=n_sites,
        beta=beta,
        omega_x=omega_x,
        omega_z=omega_z,
        pair_terms=pair_terms,
        couplings=couplings,
    )


def build_design_matrix(n_sites: int) -> np.ndarray:
    """
    Pair-resolved channel family -> direct control of each J_{ij}^{ab}.

    The flattened map is J = A theta with A = I.
    """
    n_coeff = len(enumerate_pair_terms(n_sites))
    return np.eye(n_coeff, dtype=float)


def solve_design_parameters(design_matrix: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
    theta, *_ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    return theta


def relative_l2_error(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-15) -> float:
    ref_norm = float(np.linalg.norm(reference))
    denom = ref_norm if ref_norm > eps else eps
    return float(np.linalg.norm(reference - estimate) / denom)


def design_rank_info(n_sites: int) -> Tuple[int, int]:
    design_matrix = build_design_matrix(n_sites)
    rank = int(np.linalg.matrix_rank(design_matrix))
    return rank, design_matrix.shape[1]


def model_to_term_arrays(model: AllToAllModel) -> Dict[str, np.ndarray]:
    """
    Convert pair-term structure into compact numpy arrays used by simulators.
    """
    n_terms = len(model.pair_terms)
    i_idx = np.empty(n_terms, dtype=np.int32)
    j_idx = np.empty(n_terms, dtype=np.int32)
    a_idx = np.empty(n_terms, dtype=np.int32)
    b_idx = np.empty(n_terms, dtype=np.int32)
    j_val = np.asarray(model.couplings, dtype=float)
    signs = np.sign(j_val)
    signs[signs == 0.0] = 1.0
    amps = np.sqrt(np.abs(j_val))

    for t, term in enumerate(model.pair_terms):
        i_idx[t] = term.i
        j_idx[t] = term.j
        a_idx[t] = term.a
        b_idx[t] = term.b

    return {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "a_idx": a_idx,
        "b_idx": b_idx,
        "j_val": j_val,
        "j_sign": signs,
        "j_amp_sqrt": amps,
    }


def summarize_couplings(model: AllToAllModel) -> Dict[str, float]:
    j = np.asarray(model.couplings, dtype=float)
    return {
        "j_mean": float(np.mean(j)),
        "j_std": float(np.std(j)),
        "j_max_abs": float(np.max(np.abs(j))) if j.size else 0.0,
    }


def flatten_two_body_couplings(model: AllToAllModel) -> np.ndarray:
    return np.asarray(model.couplings, dtype=float).copy()


def reconstruct_couplings(n_sites: int, target_vector: np.ndarray) -> Tuple[np.ndarray, int, int]:
    design_matrix = build_design_matrix(n_sites)
    theta = solve_design_parameters(design_matrix, target_vector)
    reconstructed = design_matrix @ theta
    rank, n_cols = design_rank_info(n_sites)
    return reconstructed, rank, n_cols

