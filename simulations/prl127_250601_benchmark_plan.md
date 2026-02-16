# Simulation Plan: PRL 127, 250601 (Cresser and Anders, 2021)

## Goal
Build a benchmark suite that directly tests the predictions of
`Phys. Rev. Lett. 127, 250601 (2021)` in the HMF-v2 workflow:
- weak-coupling corrections (their Eq. (3), including V-system coherence),
- ultrastrong-coupling limit state (their Eq. (7)),
- explicit qubit and two-qubit ultrastrong formulas (their Eqs. (8) and (10)),
- crossover behavior for all coupling strengths using exact finite-bath numerics.

## Why This Is a Good Follow-Up Target
The PRL gives closed-form asymptotic predictions (weak and ultrastrong), while
our project focuses on operator-level construction and representability. This
lets us compare:
1. asymptotic analytic states,
2. exact finite-bath trace-out states at intermediate coupling,
3. basis-rotation/coherence predictions across coupling strength.

## Primary Models
1. Single qubit + bosonic reservoir
   - `H_S = (omega_q/2) sigma_z`
   - `X = cos(theta) sigma_z - sin(theta) sigma_x`
   - target checks: Eq. (8), energetic coherence persistence, basis crossover.
2. Three-level V-system + bosonic reservoir
   - target checks: Eq. (6) structure and nonzero equilibrium coherence `g(beta)`.
3. Two coupled qubits + two reservoirs
   - target checks: Eq. (10), `omega_q` independence in ultrastrong limit.

## Parallel Numerical Implementations
1. Exact finite-bath trace-out (implemented first)
   - discretize spectral density `J(omega)` to finite modes,
   - truncate each bosonic mode to finite Fock cutoff,
   - construct full Hamiltonian and reduced equilibrium state by partial trace.
2. Independent stochastic/influence implementation (planned next)
   - imaginary-time quenched-field sampling with kernel from `J(omega)`,
   - estimate reduced state and compare against method (1).

Using two independent pipelines protects against implementation bias and is
manuscript-friendly for a "parallel numerical implementation" claim.

## Spectral Density and Discretization
Start with the PRL Figure-1 choice:
- `J(omega) = Q * tau_c * omega * exp(-tau_c * omega)`,
- finite interval `[0, omega_max]`, uniform quadrature for mode weights.

Sensitivity studies:
1. vary number of modes,
2. vary Fock cutoff,
3. vary `omega_max`.

## Observables and Comparison Metrics
For each `(beta, lambda)`:
1. `rho_S^exact(lambda)` from finite-bath trace-out.
2. `tau_S` (bare Gibbs).
3. `rho_US` from Eq. (7)/(8) projection formula.
4. Coherence in energy basis of `H_S`:
   - `C_HS = sum_{i != j} |<e_i| rho_S |e_j>|`.
5. Coherence in coupling basis of `X`:
   - `C_X = sum_{i != j} |<x_i| rho_S |x_j>|`.
6. Distances:
   - `D_tau = (1/2)||rho_S^exact - tau_S||_1`,
   - `D_US = (1/2)||rho_S^exact - rho_US||_1`.

Expected trend:
- small `lambda`: `D_tau` small, weak-coupling correction scaling appears,
- large `lambda`: `D_US` decreases and coupling-basis structure dominates.

## Deliverables
1. `simulations/src/prl127_qubit_benchmark.py`
   - scan over `lambda`,
   - compute exact reduced states and Eq. (8) comparison,
   - save CSV + figure.
2. `simulations/results/data/prl127_qubit_scan.csv`
3. `simulations/results/figures/prl127_qubit_scan.png`
4. Follow-up scripts:
   - `prl127_vsystem_benchmark.py`,
   - `prl127_twoqubit_benchmark.py`.

## Acceptance Criteria
1. Numerical scan reproduces qualitative PRL predictions:
   - energetic coherence persists away from weak-coupling Gibbs limit,
   - approach to interaction-basis ultrastrong state.
2. Ultrastrong formula (Eq. (8)) agrees with large-`lambda` finite-bath numerics
   within truncation error.
3. Convergence tests (mode count/cutoff) reported for at least one parameter set.

## Notes on Scope
- Weak-coupling closed expressions for V-system coefficients `f_p(beta), g(beta)`
  are cited in the PRL Supplemental equations; implementing exact symbolic
  versions is a separate step and should be added after the qubit baseline.
