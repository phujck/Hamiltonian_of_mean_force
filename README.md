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

PRL qubit analytic-bridge benchmark:
- `py simulations/src/main.py --benchmark prl127_qubit_analytic_bridge`
- `powershell -ExecutionPolicy Bypass -File .\\run_safe.ps1 simulations/src/main.py --benchmark prl127_qubit_analytic_bridge` (thread-limited safe mode)
- Uses ordered-kernel constructions for the noncommuting Gaussian comparator.
- Ordered comparator controls (optional): `--ordered-time-slices 80 --ordered-kl-rank 5 --ordered-gh-order 3`
- Primary outputs:
  - `simulations/results/data/prl127_qubit_analytic_bridge_scan.csv`
  - `simulations/results/data/prl127_qubit_analytic_bridge_summary.csv`
  - `simulations/results/figures/prl127_qubit_analytic_bridge_alignment.png`

v5 regime pipeline (ED vs compact `H_MF` from `04_results_v5.tex`):
- Data generation (safe mode, CPU-limited):
  - `powershell -ExecutionPolicy Bypass -File .\\run_safe.ps1 simulations/src/hmf_v5_regime_generate.py`
  - comparator options: `--theory-model ordered` (default) or `compact`
- Plotting from saved CSVs only:
  - `py simulations/src/hmf_v5_regime_plot.py`
- Model/kernels writeup:
  - `simulations/src/hmf_v5_ordered_model_writeup.md`
- Primary outputs:
  - `simulations/results/data/hmf_v5_regime_scan.csv`
  - `simulations/results/data/hmf_v5_regime_summary.csv`
  - `simulations/results/data/hmf_v5_regime_halfstep_validation.csv`
  - `simulations/results/figures/hmf_v5_regime_panels.png`
  - `simulations/results/figures/hmf_v5_regime_panel_a_geometry.png`
  - `simulations/results/figures/hmf_v5_regime_panel_b_coherence.png`
  - `simulations/results/figures/hmf_v5_regime_panel_c_susceptibility.png`
  - `simulations/results/figures/hmf_v5_regime_panel_d_trace_distance.png`
  - manuscript copies in `manuscript/figures/` with matching filenames

v5 truncation/mode convergence probe:
- `powershell -ExecutionPolicy Bypass -File .\\run_safe.ps1 simulations/src/hmf_v5_cutoff_modes_study.py`
- Outputs:
  - `simulations/results/data/hmf_v5_cutoff_modes_scan.csv`
  - `simulations/results/data/hmf_v5_cutoff_modes_summary.csv`
  - `simulations/results/figures/hmf_v5_cutoff_modes_convergence.png`
  - `manuscript/figures/hmf_v5_cutoff_modes_convergence.png`

Safe-mode note:
- `run_safe.ps1` now defaults to 2 threads.
- You can reduce/increase manually via `SAFE_THREADS`, e.g.:
  - `$env:SAFE_THREADS='1'; powershell -ExecutionPolicy Bypass -File .\\run_safe.ps1 ...`

All-to-all designability benchmark:
- `py simulations/src/main.py --benchmark designability_alltoall`
- Primary outputs:
  - `simulations/results/data/designability_alltoall_metrics.csv`
  - `simulations/results/data/designability_alltoall_reconstruction.csv`
  - `simulations/results/figures/designability_alltoall_reconstruction.png`
  - `simulations/results/figures/designability_alltoall_ed_agreement.png`
  - `simulations/results/figures/designability_alltoall_scaling.png`
