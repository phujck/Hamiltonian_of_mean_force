# HMF Multi-Run Handoff Summary (Codex v46)

This note is a compact map of the `codex_v36` to `codex_v45` files, what each run tested, and the current conclusions to use for paper figures.

## Execution and safety conventions
- All expensive scripts were intended to be run via `run_safe.ps1`.
- Existing CSV outputs are now sufficient for crossover and turnover analysis; no new ED data is required for those conclusions.
- Fair comparison rule used in late-stage runs:
`ED` and `analytic` should be evaluated on the same spectral window, and often on the same discrete node set.

## Run-by-run map

| Run | Main script | What was tested | Key outputs |
|---|---|---|---|
| v36 | `simulations/src/hmf_narrowband_convergence_codex_v36.py` | Bandwidth scan around `omega_q=2`, fixed `theta=pi/2`, `g=0.5`, `n_modes=3`, `n_cut={4,5,6}`; looked for RMSE minima and cutoff stability | `hmf_narrowband_convergence_summary_codex_v36.csv`, `hmf_narrowband_convergence_cutoff_summary_codex_v36.csv`, `hmf_narrowband_convergence_codex_v36.png`, `hmf_narrowband_convergence_log_codex_v36.md` |
| v37 | `simulations/src/hmf_omega1_diagnostic_suite_codex_v37.py` | First full `omega_q=1` diagnostic suite with checkpointing and PDF packaging | Script only retained as baseline implementation (superseded by v38 discrete-kernel-focused outputs) |
| v38 | `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_codex_v38.py` | Fair `omega_q=1` suite comparing ED against analytic-discrete and analytic-continuum over windows, modes, cutoffs | `hmf_omega1_diagnostic_suite_discretekernel_summary_codex_v38.csv`, `hmf_omega1_diagnostic_suite_discretekernel_casebest_codex_v38.csv`, `hmf_omega1_diagnostic_suite_discretekernel_codex_v38.png`, `hmf_omega1_diagnostic_suite_discretekernel_report_fallback_codex_v38.pdf`, `hmf_omega1_diagnostic_suite_discretekernel_log_codex_v38.md` |
| v39 | `simulations/src/hmf_omega1_twomode_cutoff12_codex_v39.py` | Two-mode cutoff sweep to `n_cut=12` at fixed best window `[0.2,1.8]` | `hmf_omega1_twomode_cutoff12_summary_codex_v39.csv`, `hmf_omega1_twomode_cutoff12_conv_codex_v39.csv`, `hmf_omega1_twomode_cutoff12_codex_v39.png`, `hmf_omega1_twomode_cutoff12_log_codex_v39.md` |
| v40 | `simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.py` | Three-mode cutoff sweep to `n_cut=12` at same window `[0.2,1.8]` | `hmf_omega1_threemode_cutoff12_summary_codex_v40.csv`, `hmf_omega1_threemode_cutoff12_conv_codex_v40.csv`, `hmf_omega1_threemode_cutoff12_codex_v40.png`, `hmf_omega1_threemode_cutoff12_log_codex_v40.md` |
| v41 | `simulations/src/hmf_omega1_resonant_nodes_threemode_codex_v41.py` | Custom nonuniform 3-mode node sets (`[0.05,1,2]`, `[0.05,1,1.8]`) vs uniform `[0.2,1,1.8]` | `hmf_omega1_resonant_nodes_threemode_summary_codex_v41.csv`, `hmf_omega1_resonant_nodes_threemode_codex_v41.png`, `hmf_omega1_resonant_nodes_threemode_log_codex_v41.md` |
| v42 | `simulations/src/hmf_omega1_bestkernel_sweeps_codex_v42.py` | Parameter sweeps (`g`, `beta`, `theta`) for selected fair kernel (`omega_q=1`, `m=2`, `c=6`, window `[0.2,1.8]`), analytic-discrete only | `hmf_omega1_bestkernel_sweeps_summary_codex_v42.csv`, `hmf_omega1_bestkernel_sweeps_codex_v42.png`, `hmf_omega1_bestkernel_sweeps_log_codex_v42.md` |
| v43 | `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.py` | Same sweeps as v42 but overlays analytic-discrete and analytic-continuous | `hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_summary.csv`, `hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.png`, `hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_log.md` |
| v44 | `simulations/src/hmf_crossover_effects_explainer_codex_v44.py` | Re-analysis from existing CSVs only: crossover minima, slope sign flip, cutoff divergence, branch tradeoffs | `hmf_crossover_effects_explainer_codex_v44.png`, `hmf_crossover_effects_explainer_summary_codex_v44.csv`, `hmf_crossover_effects_explainer_log_codex_v44.md` |
| v45 | `simulations/src/hmf_turnover_magicline_codex_v45.py` | Closed-form turnover condition from scan parameterization + plotted magic line against observed turnover | `hmf_turnover_magicline_codex_v45.png`, `hmf_turnover_magicline_codex_v45_summary.csv`, `hmf_turnover_magicline_codex_v45_log.md` |

## Hard conclusions from current data

1. Best fair agreement point identified in the `omega_q=1` suite is:
- window `center_d0.80` (`omega_min=0.2`, `omega_max=1.8`, bandwidth `1.6`)
- `n_modes=2`, `n_cut=6`
- `rmse_ed_vs_disc_p00=0.011036` (from v38 log)

2. Increasing mode count did not improve fairness match at this window:
- Two-mode sweep (v39): best `rmse_ed_vs_disc_p00=0.011046` at `n_cut=6`
- Three-mode sweep (v40): best `rmse_ed_vs_disc_p00=0.027387` at `n_cut=9`

3. Custom resonant 3-mode nodes were not better (v41):
- Best tested set was still uniform `[0.2,1.0,1.8]` with `rmse_ed_vs_disc_p00=0.114395`
- Resonant sets were worse (`0.132529`, `0.141308`)

4. Crossover behavior is real in the bandwidth scan (v36/v44):
- For `n_cut=4,5,6`, RMSE minima all occur at bandwidth `3.2`
- RMSE slope changes sign around bandwidth `~3.38` to `~3.55`
- Internal ED cutoff mismatch increases strongly past crossover

5. No single analytic branch dominates all sweeps (v43):
- Coupling sweep: analytic-continuous has lower population RMSE than analytic-discrete
- Temperature and angle sweeps: analytic-discrete has lower population RMSE
- Coherence can be good for both branches depending on sweep

6. Turnover condition from analytical scan parameters (v45):
- With lower-window clamp `omega_min = max(omega_floor, omega_q - delta)`
- Magic bandwidth line:
`B* = 2(omega_q - omega_floor)`
- In v36 data (`omega_q=2`, `omega_floor=0.1`):
`B*=3.8`, observed RMSE minimum at `B=3.2`, slope-zero at `B~=3.377`
- Interpretation: turnover region is where resonance pinning breaks and detuning/cutoff effects take over.

## Recommended figure set for paper drafting

1. Fair benchmark map and global best:
- `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_codex_v38.png`
- Optional packaged report: `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_report_fallback_codex_v38.pdf`

2. Cutoff convergence, two-mode vs three-mode:
- `simulations/src/hmf_omega1_twomode_cutoff12_codex_v39.png`
- `simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.png`

3. Final parameter sweeps with branch comparison:
- `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.png`

4. Regime/crossover evidence figure:
- `simulations/src/hmf_crossover_effects_explainer_codex_v44.png`

5. Analytic turnover condition ("magic line") figure:
- `simulations/src/hmf_turnover_magicline_codex_v45.png`

## Practical guidance for next agents

- If a run compares ED and analytic, keep spectral window alignment explicit in logs.
- Keep `n_modes=2`, `n_cut=6`, window `[0.2,1.8]` as the baseline reference point.
- Treat broad-window improvements cautiously unless cutoff-convergence metrics are shown together.
- For narrative consistency:
Use v44 and v45 together.
v44 shows empirical crossover in errors.
v45 gives the compact analytical condition that predicts where turnover starts.
