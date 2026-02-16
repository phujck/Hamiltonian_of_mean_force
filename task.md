# v2 Execution Plan (Hamiltonian_of_mean_force)

## Objective
Frame and execute this manuscript as the follow-up to arXiv:2602.13146.

## Workstream A: Framing and Narrative
- [x] Add explicit sequel framing in manuscript entry points.
- [x] Add a citation entry for arXiv:2602.13146.
- [x] Document contribution boundaries in `followup_framing.md`.
- [ ] Tighten section-level claims so each section states "new here" in first paragraph.

## Workstream B: Theory Core
- [ ] Verify closure criterion statement is hypothesis-complete and internally consistent.
- [ ] Ensure normalization conventions are consistent (`Z_B`, additive identity shifts, and sign conventions).
- [ ] Add one compact theorem-style statement for commuting baseline recovery.
- [ ] Add one compact theorem-style statement for closed-algebra noncommuting construction.

## Workstream C: Numerical Backing
- [x] Add a minimal benchmark script for a noncommuting qubit case.
- [x] Report comparison metric between direct trace-out and derived HMF form (trace distance or operator norm).
- [x] Add one figure/table summarizing agreement and parameter sweep.
- [ ] Extend the benchmark to the V-system case (weak + ultrastrong asymptotics).
- [ ] Extend the benchmark to the two-qubit/two-reservoir case.

## Workstream D: Submission Readiness
- [ ] Compile with no BibTeX warnings for new citations.
- [ ] Run a pass for notation consistency (`H_Q`/`H_S`, `K`/`\kappa_\ell`, `H_{\mathrm{eff}}`/`H_{\mathrm{MF}}`).
- [ ] Create arXiv flatten package and verify standalone build.
