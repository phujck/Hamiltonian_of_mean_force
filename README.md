# Hamiltonian of Mean Force (v2)

## Overview
This repository is set up as the follow-up manuscript to:
- arXiv:2602.13146, `Mean-Force Hamiltonians from Influence Functionals`

Paper I established the quenched-density representation and the commuting
Gaussian benchmark. This v2 project is scoped to the noncommuting Gaussian
representability problem and closed-algebra constructions.

## Key Project Files
- `followup_framing.md`: explicit "paper I -> paper II" framing and scope boundaries.
- `task.md`: active execution checklist for theory, numerics, and submission prep.
- `manuscript/tex/main.tex`: current draft entry point.
- `simulations/prl127_250601_benchmark_plan.md`: simulation roadmap anchored to PRL 127, 250601.

## Directory Structure
- `theory/`: derivations and algebraic notes.
- `simulations/`: benchmark code and numerical checks.
- `manuscript/`: RevTeX manuscript source and build artifacts.
- `literature/`: bibliography and literature maps.

## Usage
Use PowerShell from the repository root:

```powershell
./manage.ps1 install   # Install dependencies
./manage.ps1 sim       # Run simulations
./manage.ps1 paper     # Compile manuscript
./manage.ps1 clean     # Clean build artifacts
```

Current simulation baseline:
- `./manage.ps1 sim` runs the PRL-anchored single-qubit benchmark and writes
  - `simulations/results/data/prl127_qubit_scan.csv`
  - `simulations/results/figures/prl127_qubit_scan.png`

All-to-all designability benchmark:
- `py simulations/src/main.py --benchmark designability_alltoall`
- Primary outputs:
  - `simulations/results/data/designability_alltoall_metrics.csv`
  - `simulations/results/data/designability_alltoall_reconstruction.csv`
  - `simulations/results/figures/designability_alltoall_reconstruction.png`
  - `simulations/results/figures/designability_alltoall_ed_agreement.png`
  - `simulations/results/figures/designability_alltoall_scaling.png`
