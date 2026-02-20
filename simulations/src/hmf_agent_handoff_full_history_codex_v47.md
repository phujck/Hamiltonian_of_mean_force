# HMF Full Experiment History (Codex v47)

This note records the experimentation sequence from `codex_v8` onward, including purpose, generated artifacts, and key takeaways when available.

## Scope and conventions
- Scope: versions `v8` through `v45` plus prior handoff `v46`.
- Source of truth: files in `simulations/src` with `codex_v*` in the name.
- Safety convention used during heavy runs: `run_safe.ps1`.
- If a version has no dedicated log, purpose is inferred from the script docstring and filename.

## Version Index (v8+)
| Version | Files | Primary script |
|---|---:|---|
| 8 | 6 | `hmf_delta_factor_coherence_diagnostic_codex_v8.py` |
| 9 | 5 | `hmf_rho00_ratio_functional_diff_codex_v9.py` |
| 10 | 4 | `hmf_chi_vs_delta_mismatch_codex_v10.py` |
| 11 | 5 | `hmf_kernel_factor_similarity_codex_v11.py` |
| 12 | 5 | `hmf_bestmodel_ordered_ed_sweeps_codex_v12.py` |
| 13 | 4 | `hmf_population_turning_diagnostic_codex_v13.py` |
| 14 | 4 | `hmf_tex_exact_consistency_codex_v14.py` |
| 15 | 3 | `hmf_ordered_convergence_point_codex_v15.py` |
| 16 | 4 | `hmf_stable_exact_eval_codex_v16.py` |
| 17 | 5 | `hmf_population_sweep_stable_compare_codex_v17.py` |
| 18 | 5 | `hmf_diagonal_rapidity_mismatch_codex_v18.py` |
| 19 | 5 | `hmf_diagonal_rapidity_renorm_compare_codex_v19.py` |
| 20 | 5 | `hmf_beta_rate_bound_diagnostic_codex_v20.py` |
| 21 | 5 | `hmf_ed_beta_rate_convergence_codex_v21.py` |
| 22 | 5 | `hmf_multimode_ed_vs_exact_codex_v22.py` |
| 22b | 5 | `hmf_multimode_ed_vs_exact_extended_codex_v22b.py` |
| 23 | 6 | `hmf_bestmodel_betaflow_edfit_codex_v23.py` |
| 24 | 5 | `hmf_bestmodel_betaflow_blended_codex_v24.py` |
| 25 | 9 | `hmf_v12_ed_convergence_checkpoint_codex_v25.py` |
| 26 | 1 | `hmf_ed_beta_rate_with_analytic_codex_v26.py` |
| 27 | 5 | `hmf_ed_beta_rate_overlay_from_v21_codex_v27.py` |
| 28 | 1 | `hmf_ed_modesafe_convergence_codex_v28.py` |
| 29 | 5 | `hmf_ed_convergence_highbeta_lomodes_codex_v29.py` |
| 30 | 5 | `hmf_ed_convergence_highbeta_lomodes_codex_v30.py` |
| 31 | 5 | `hmf_ed_convergence_highbeta_scalebound_overlay_codex_v31.py` |
| 32 | 1 | `hmf_plot_sweeps_ultrastrong_ref_codex_v32.py` |
| 33 | 5 | `hmf_ed_convergence_bestanalytic_usref_codex_v33.py` |
| 34 | 5 | `hmf_spectral_window_characterization_codex_v34.py` |
| 35 | 5 | `hmf_omegaq_scaled_window_scan_codex_v35.py` |
| 36 | 6 | `hmf_narrowband_convergence_codex_v36.py` |
| 37 | 1 | `hmf_omega1_diagnostic_suite_codex_v37.py` |
| 38 | 11 | `hmf_omega1_diagnostic_suite_discretekernel_codex_v38.py` |
| 39 | 6 | `hmf_omega1_twomode_cutoff12_codex_v39.py` |
| 40 | 6 | `hmf_omega1_threemode_cutoff12_codex_v40.py` |
| 41 | 5 | `hmf_omega1_resonant_nodes_threemode_codex_v41.py` |
| 42 | 5 | `hmf_omega1_bestkernel_sweeps_codex_v42.py` |
| 43 | 5 | `hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.py` |
| 44 | 4 | `hmf_crossover_effects_explainer_codex_v44.py` |
| 45 | 4 | `hmf_turnover_magicline_codex_v45.py` |
| 46 | 1 | `-` |

## Detailed Chronology
### v8
- Purpose: Delta-Factor and Coherence Diagnostic
- Primary script: `simulations/src/hmf_delta_factor_coherence_diagnostic_codex_v8.py`
- Main log: `simulations/src/hmf_delta_factor_log_codex_v8.md`
- Artifacts:
  - `simulations/src/hmf_delta_coherence_variants_codex_v8.csv`
  - `simulations/src/hmf_delta_factor_coherence_diag_codex_v8.png`
  - `simulations/src/hmf_delta_factor_coherence_diagnostic_codex_v8.py`
  - `simulations/src/hmf_delta_factor_log_codex_v8.md`
  - `simulations/src/hmf_delta_factor_scan_codex_v8.csv`
  - `simulations/src/hmf_delta_factor_summary_codex_v8.csv`

### v9
- Purpose: Functional Difference Diagnostic
- Primary script: `simulations/src/hmf_rho00_ratio_functional_diff_codex_v9.py`
- Main log: `simulations/src/hmf_rho00_ratio_functional_log_codex_v9.md`
- Artifacts:
  - `simulations/src/hmf_rho00_ratio_functional_diff_codex_v9.png`
  - `simulations/src/hmf_rho00_ratio_functional_diff_codex_v9.py`
  - `simulations/src/hmf_rho00_ratio_functional_log_codex_v9.md`
  - `simulations/src/hmf_rho00_ratio_functional_scan_codex_v9.csv`
  - `simulations/src/hmf_rho00_ratio_functional_summary_codex_v9.csv`

### v10
- Purpose: Chi vs Delta Mismatch Diagnostic
- Primary script: `simulations/src/hmf_chi_vs_delta_mismatch_codex_v10.py`
- Main log: `simulations/src/hmf_chi_vs_delta_log_codex_v10.md`
- Artifacts:
  - `simulations/src/hmf_chi_vs_delta_log_codex_v10.md`
  - `simulations/src/hmf_chi_vs_delta_mismatch_codex_v10.py`
  - `simulations/src/hmf_chi_vs_delta_scan_codex_v10.csv`
  - `simulations/src/hmf_chi_vs_delta_summary_codex_v10.csv`

### v11
- Purpose: Kernel Similarity Diagnostic
- Primary script: `simulations/src/hmf_kernel_factor_similarity_codex_v11.py`
- Main log: `simulations/src/hmf_kernel_factor_similarity_log_codex_v11.md`
- Artifacts:
  - `simulations/src/hmf_kernel_factor_similarity_codex_v11.png`
  - `simulations/src/hmf_kernel_factor_similarity_codex_v11.py`
  - `simulations/src/hmf_kernel_factor_similarity_curves_codex_v11.csv`
  - `simulations/src/hmf_kernel_factor_similarity_log_codex_v11.md`
  - `simulations/src/hmf_kernel_factor_similarity_summary_codex_v11.csv`

### v12
- Purpose: Best Model vs Ordered vs ED
- Primary script: `simulations/src/hmf_bestmodel_ordered_ed_sweeps_codex_v12.py`
- Main log: `simulations/src/hmf_bestmodel_ordered_ed_log_codex_v12.md`
- Key takeaway snippet: # Best Model vs Ordered vs ED (Codex v12)
- Artifacts:
  - `simulations/src/hmf_bestmodel_ordered_ed_log_codex_v12.md`
  - `simulations/src/hmf_bestmodel_ordered_ed_scan_codex_v12.csv`
  - `simulations/src/hmf_bestmodel_ordered_ed_summary_codex_v12.csv`
  - `simulations/src/hmf_bestmodel_ordered_ed_sweeps_codex_v12.png`
  - `simulations/src/hmf_bestmodel_ordered_ed_sweeps_codex_v12.py`

### v13
- Purpose: Population Turning Diagnostic
- Primary script: `simulations/src/hmf_population_turning_diagnostic_codex_v13.py`
- Main log: `simulations/src/hmf_population_turning_log_codex_v13.md`
- Key takeaway snippet: Detected turning betas (best): [1.505, 4.115]
- Artifacts:
  - `simulations/src/hmf_population_turning_diag_codex_v13.png`
  - `simulations/src/hmf_population_turning_diagnostic_codex_v13.py`
  - `simulations/src/hmf_population_turning_log_codex_v13.md`
  - `simulations/src/hmf_population_turning_scan_codex_v13.csv`

### v14
- Purpose: TeX Exact-Form Consistency
- Primary script: `simulations/src/hmf_tex_exact_consistency_codex_v14.py`
- Main log: `simulations/src/hmf_tex_exact_consistency_log_codex_v14.md`
- Artifacts:
  - `simulations/src/hmf_tex_exact_consistency_codex_v14.py`
  - `simulations/src/hmf_tex_exact_consistency_log_codex_v14.md`
  - `simulations/src/hmf_tex_exact_consistency_scan_codex_v14.csv`
  - `simulations/src/hmf_tex_exact_consistency_summary_codex_v14.csv`

### v15
- Purpose: Ordered Convergence Probe
- Primary script: `simulations/src/hmf_ordered_convergence_point_codex_v15.py`
- Main log: `simulations/src/hmf_ordered_convergence_point_log_codex_v15.md`
- Artifacts:
  - `simulations/src/hmf_ordered_convergence_point_codex_v15.py`
  - `simulations/src/hmf_ordered_convergence_point_log_codex_v15.md`
  - `simulations/src/hmf_ordered_convergence_point_scan_codex_v15.csv`

### v16
- Purpose: Stable Exact Evaluator Check
- Primary script: `simulations/src/hmf_stable_exact_eval_codex_v16.py`
- Main log: `simulations/src/hmf_stable_exact_eval_log_codex_v16.md`
- Artifacts:
  - `simulations/src/hmf_stable_exact_eval_codex_v16.py`
  - `simulations/src/hmf_stable_exact_eval_log_codex_v16.md`
  - `simulations/src/hmf_stable_exact_eval_scan_codex_v16.csv`
  - `simulations/src/hmf_stable_exact_eval_summary_codex_v16.csv`

### v17
- Purpose: Population Sweep Stable Comparison
- Primary script: `simulations/src/hmf_population_sweep_stable_compare_codex_v17.py`
- Main log: `simulations/src/hmf_population_sweep_stable_log_codex_v17.md`
- Artifacts:
  - `simulations/src/hmf_population_sweep_stable_compare_codex_v17.png`
  - `simulations/src/hmf_population_sweep_stable_compare_codex_v17.py`
  - `simulations/src/hmf_population_sweep_stable_log_codex_v17.md`
  - `simulations/src/hmf_population_sweep_stable_scan_codex_v17.csv`
  - `simulations/src/hmf_population_sweep_stable_summary_codex_v17.csv`

### v18
- Purpose: Diagonal Rapidity Mismatch
- Primary script: `simulations/src/hmf_diagonal_rapidity_mismatch_codex_v18.py`
- Main log: `simulations/src/hmf_diagonal_rapidity_mismatch_log_codex_v18.md`
- Artifacts:
  - `simulations/src/hmf_diagonal_rapidity_mismatch_codex_v18.png`
  - `simulations/src/hmf_diagonal_rapidity_mismatch_codex_v18.py`
  - `simulations/src/hmf_diagonal_rapidity_mismatch_log_codex_v18.md`
  - `simulations/src/hmf_diagonal_rapidity_mismatch_scan_codex_v18.csv`
  - `simulations/src/hmf_diagonal_rapidity_mismatch_summary_codex_v18.csv`

### v19
- Purpose: Diagonal Rapidity Renorm Comparison
- Primary script: `simulations/src/hmf_diagonal_rapidity_renorm_compare_codex_v19.py`
- Main log: `simulations/src/hmf_diagonal_rapidity_renorm_log_codex_v19.md`
- Artifacts:
  - `simulations/src/hmf_diagonal_rapidity_renorm_compare_codex_v19.png`
  - `simulations/src/hmf_diagonal_rapidity_renorm_compare_codex_v19.py`
  - `simulations/src/hmf_diagonal_rapidity_renorm_log_codex_v19.md`
  - `simulations/src/hmf_diagonal_rapidity_renorm_scan_codex_v19.csv`
  - `simulations/src/hmf_diagonal_rapidity_renorm_summary_codex_v19.csv`

### v20
- Purpose: Beta Rate-Bound Diagnostic
- Primary script: `simulations/src/hmf_beta_rate_bound_diagnostic_codex_v20.py`
- Main log: `simulations/src/hmf_beta_rate_bound_log_codex_v20.md`
- Key takeaway snippet: - rmse_p00_raw_vs_ordered: 0.669707
- Artifacts:
  - `simulations/src/hmf_beta_rate_bound_diagnostic_codex_v20.png`
  - `simulations/src/hmf_beta_rate_bound_diagnostic_codex_v20.py`
  - `simulations/src/hmf_beta_rate_bound_log_codex_v20.md`
  - `simulations/src/hmf_beta_rate_bound_scan_codex_v20.csv`
  - `simulations/src/hmf_beta_rate_bound_summary_codex_v20.csv`

### v21
- Purpose: ED Beta-Rate Convergence
- Primary script: `simulations/src/hmf_ed_beta_rate_convergence_codex_v21.py`
- Main log: `simulations/src/hmf_ed_beta_rate_convergence_log_codex_v21.md`
- Artifacts:
  - `simulations/src/hmf_ed_beta_rate_convergence_codex_v21.png`
  - `simulations/src/hmf_ed_beta_rate_convergence_codex_v21.py`
  - `simulations/src/hmf_ed_beta_rate_convergence_log_codex_v21.md`
  - `simulations/src/hmf_ed_beta_rate_convergence_scan_codex_v21.csv`
  - `simulations/src/hmf_ed_beta_rate_convergence_summary_codex_v21.csv`

### v22
- Purpose: Multimode ED vs Exact/Ordered
- Primary script: `simulations/src/hmf_multimode_ed_vs_exact_codex_v22.py`
- Main log: `simulations/src/hmf_multimode_ed_vs_exact_log_codex_v22.md`
- Artifacts:
  - `simulations/src/hmf_multimode_ed_vs_exact_codex_v22.png`
  - `simulations/src/hmf_multimode_ed_vs_exact_codex_v22.py`
  - `simulations/src/hmf_multimode_ed_vs_exact_log_codex_v22.md`
  - `simulations/src/hmf_multimode_ed_vs_exact_scan_codex_v22.csv`
  - `simulations/src/hmf_multimode_ed_vs_exact_summary_codex_v22.csv`

### v22b
- Purpose: Extended Multimode ED vs Exact/Ordered
- Primary script: `simulations/src/hmf_multimode_ed_vs_exact_extended_codex_v22b.py`
- Main log: `simulations/src/hmf_multimode_ed_vs_exact_extended_log_codex_v22b.md`
- Artifacts:
  - `simulations/src/hmf_multimode_ed_vs_exact_extended_codex_v22b.png`
  - `simulations/src/hmf_multimode_ed_vs_exact_extended_codex_v22b.py`
  - `simulations/src/hmf_multimode_ed_vs_exact_extended_log_codex_v22b.md`
  - `simulations/src/hmf_multimode_ed_vs_exact_extended_scan_codex_v22b.csv`
  - `simulations/src/hmf_multimode_ed_vs_exact_extended_summary_codex_v22b.csv`

### v23
- Purpose: Best Model (Beta-Flow, ED-Fit) Codex v23
- Primary script: `simulations/src/hmf_bestmodel_betaflow_edfit_codex_v23.py`
- Main log: `simulations/src/hmf_bestmodel_betaflow_edfit_log_codex_v23.md`
- Key takeaway snippet: # Best Model (Beta-Flow, ED-Fit) Codex v23
- Artifacts:
  - `simulations/src/hmf_bestmodel_betaflow_anchorfit_codex_v23.csv`
  - `simulations/src/hmf_bestmodel_betaflow_edfit_codex_v23.png`
  - `simulations/src/hmf_bestmodel_betaflow_edfit_codex_v23.py`
  - `simulations/src/hmf_bestmodel_betaflow_edfit_log_codex_v23.md`
  - `simulations/src/hmf_bestmodel_betaflow_edfit_scan_codex_v23.csv`
  - `simulations/src/hmf_bestmodel_betaflow_edfit_summary_codex_v23.csv`

### v24
- Purpose: Best Model (Beta-Flow Blended) Codex v24
- Primary script: `simulations/src/hmf_bestmodel_betaflow_blended_codex_v24.py`
- Main log: `simulations/src/hmf_bestmodel_betaflow_blended_log_codex_v24.md`
- Key takeaway snippet: # Best Model (Beta-Flow Blended) Codex v24
- Artifacts:
  - `simulations/src/hmf_bestmodel_betaflow_blended_codex_v24.png`
  - `simulations/src/hmf_bestmodel_betaflow_blended_codex_v24.py`
  - `simulations/src/hmf_bestmodel_betaflow_blended_log_codex_v24.md`
  - `simulations/src/hmf_bestmodel_betaflow_blended_scan_codex_v24.csv`
  - `simulations/src/hmf_bestmodel_betaflow_blended_summary_codex_v24.csv`

### v25
- Purpose: v12 ED Convergence Checkpoint Scan
- Primary script: `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25.py`
- Main log: `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_log.md`
- Artifacts:
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25.py`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_log.md`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_smoketest_log.md`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_smoketest_summary.csv`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_summary.csv`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25.png`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_smoketest.png`
  - `simulations/src/hmf_v12_ed_convergence_checkpoint_codex_v25_scan.csv`
  - ... plus 1 additional file(s) for this version

### v26
- Purpose: ED beta-rate convergence with analytic overlays (v12-style and raw compact).
- Primary script: `simulations/src/hmf_ed_beta_rate_with_analytic_codex_v26.py`
- Artifacts:
  - `simulations/src/hmf_ed_beta_rate_with_analytic_codex_v26.py`

### v27
- Purpose: ED Overlay From v21
- Primary script: `simulations/src/hmf_ed_beta_rate_overlay_from_v21_codex_v27.py`
- Main log: `simulations/src/hmf_ed_beta_rate_overlay_from_v21_log_codex_v27.md`
- Artifacts:
  - `simulations/src/hmf_ed_beta_rate_overlay_from_v21_codex_v27.png`
  - `simulations/src/hmf_ed_beta_rate_overlay_from_v21_codex_v27.py`
  - `simulations/src/hmf_ed_beta_rate_overlay_from_v21_log_codex_v27.md`
  - `simulations/src/hmf_ed_beta_rate_overlay_from_v21_scan_codex_v27.csv`
  - `simulations/src/hmf_ed_beta_rate_overlay_from_v21_summary_codex_v27.csv`

### v28
- Purpose: Mode-safe ED convergence extension with checkpoints.
- Primary script: `simulations/src/hmf_ed_modesafe_convergence_codex_v28.py`
- Artifacts:
  - `simulations/src/hmf_ed_modesafe_convergence_codex_v28.py`

### v29
- Purpose: High-Beta Low-Mode ED Convergence
- Primary script: `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29.py`
- Main log: `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29_log.md`
- Artifacts:
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29.png`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29.py`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29_log.md`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29_scan.csv`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v29_summary.csv`

### v30
- Purpose: High-Beta Low-Mode ED Convergence
- Primary script: `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30.py`
- Main log: `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30_log.md`
- Artifacts:
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30.png`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30.py`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30_log.md`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30_scan.csv`
  - `simulations/src/hmf_ed_convergence_highbeta_lomodes_codex_v30_summary.csv`

### v31
- Purpose: High-Beta Scale-Bound Analytic Overlay
- Primary script: `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_codex_v31.py`
- Main log: `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_log_codex_v31.md`
- Artifacts:
  - `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_codex_v31.png`
  - `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_codex_v31.py`
  - `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_log_codex_v31.md`
  - `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_scan_codex_v31.csv`
  - `simulations/src/hmf_ed_convergence_highbeta_scalebound_overlay_summary_codex_v31.csv`

### v32
- Purpose: Convergence dashboard with ultrastrong-limit reference overlays.
- Primary script: `simulations/src/hmf_plot_sweeps_ultrastrong_ref_codex_v32.py`
- Artifacts:
  - `simulations/src/hmf_plot_sweeps_ultrastrong_ref_codex_v32.py`

### v33
- Purpose: High-Beta ED vs Best Analytic + Ultrastrong
- Primary script: `simulations/src/hmf_ed_convergence_bestanalytic_usref_codex_v33.py`
- Main log: `simulations/src/hmf_ed_convergence_bestanalytic_usref_log_codex_v33.md`
- Key takeaway snippet: # High-Beta ED vs Best Analytic + Ultrastrong (Codex v33)
- Artifacts:
  - `simulations/src/hmf_ed_convergence_bestanalytic_usref_codex_v33.png`
  - `simulations/src/hmf_ed_convergence_bestanalytic_usref_codex_v33.py`
  - `simulations/src/hmf_ed_convergence_bestanalytic_usref_log_codex_v33.md`
  - `simulations/src/hmf_ed_convergence_bestanalytic_usref_scan_codex_v33.csv`
  - `simulations/src/hmf_ed_convergence_bestanalytic_usref_summary_codex_v33.csv`

### v34
- Purpose: Spectral Window Characterization
- Primary script: `simulations/src/hmf_spectral_window_characterization_codex_v34.py`
- Main log: `simulations/src/hmf_spectral_window_characterization_log_codex_v34.md`
- Artifacts:
  - `simulations/src/hmf_spectral_window_characterization_codex_v34.png`
  - `simulations/src/hmf_spectral_window_characterization_codex_v34.py`
  - `simulations/src/hmf_spectral_window_characterization_log_codex_v34.md`
  - `simulations/src/hmf_spectral_window_characterization_scan_codex_v34.csv`
  - `simulations/src/hmf_spectral_window_characterization_summary_codex_v34.csv`

### v35
- Purpose: Omega_q Scaled Window Scan
- Primary script: `simulations/src/hmf_omegaq_scaled_window_scan_codex_v35.py`
- Main log: `simulations/src/hmf_omegaq_scaled_window_log_codex_v35.md`
- Artifacts:
  - `simulations/src/hmf_omegaq_scaled_window_codex_v35.png`
  - `simulations/src/hmf_omegaq_scaled_window_log_codex_v35.md`
  - `simulations/src/hmf_omegaq_scaled_window_scan_codex_v35.csv`
  - `simulations/src/hmf_omegaq_scaled_window_scan_codex_v35.py`
  - `simulations/src/hmf_omegaq_scaled_window_summary_codex_v35.csv`

### v36
- Purpose: Narrowband Convergence Around omega_q
- Primary script: `simulations/src/hmf_narrowband_convergence_codex_v36.py`
- Main log: `simulations/src/hmf_narrowband_convergence_log_codex_v36.md`
- Artifacts:
  - `simulations/src/hmf_narrowband_convergence_codex_v36.png`
  - `simulations/src/hmf_narrowband_convergence_codex_v36.py`
  - `simulations/src/hmf_narrowband_convergence_cutoff_summary_codex_v36.csv`
  - `simulations/src/hmf_narrowband_convergence_log_codex_v36.md`
  - `simulations/src/hmf_narrowband_convergence_scan_codex_v36.csv`
  - `simulations/src/hmf_narrowband_convergence_summary_codex_v36.csv`

### v37
- Purpose: Complete omega_q=1 diagnostic suite for ED vs analytic agreement.
- Primary script: `simulations/src/hmf_omega1_diagnostic_suite_codex_v37.py`
- Artifacts:
  - `simulations/src/hmf_omega1_diagnostic_suite_codex_v37.py`

### v38
- Purpose: Omega_q=1 Fair Diagnostic Suite
- Primary script: `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_codex_v38.py`
- Main log: `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_log_codex_v38.md`
- Key takeaway snippet: Global best agreement uses the discrete-kernel analytic branch:
- Artifacts:
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_codex_v38.py`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_log_codex_v38.md`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_summary_codex_v38.csv`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_codex_v38.png`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_report_fallback_codex_v38.pdf`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_casebest_codex_v38.csv`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_cutoff_codex_v38.csv`
  - `simulations/src/hmf_omega1_diagnostic_suite_discretekernel_scan_codex_v38.csv`
  - ... plus 3 additional file(s) for this version

### v39
- Purpose: Omega_q=1 Two-Mode Cutoff<=12 Scan
- Primary script: `simulations/src/hmf_omega1_twomode_cutoff12_codex_v39.py`
- Main log: `simulations/src/hmf_omega1_twomode_cutoff12_log_codex_v39.md`
- Key takeaway snippet: Best cutoff by fair metric (ED vs analytic-discrete p00): n_cut=6, rmse=0.011046
- Artifacts:
  - `simulations/src/hmf_omega1_twomode_cutoff12_codex_v39.png`
  - `simulations/src/hmf_omega1_twomode_cutoff12_codex_v39.py`
  - `simulations/src/hmf_omega1_twomode_cutoff12_conv_codex_v39.csv`
  - `simulations/src/hmf_omega1_twomode_cutoff12_log_codex_v39.md`
  - `simulations/src/hmf_omega1_twomode_cutoff12_scan_codex_v39.csv`
  - `simulations/src/hmf_omega1_twomode_cutoff12_summary_codex_v39.csv`

### v40
- Purpose: Omega_q=1 Three-Mode Cutoff<=12 Scan
- Primary script: `simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.py`
- Main log: `simulations/src/hmf_omega1_threemode_cutoff12_log_codex_v40.md`
- Key takeaway snippet: Best cutoff by fair metric (ED vs analytic-discrete p00): n_cut=9, rmse=0.027387
- Artifacts:
  - `simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.png`
  - `simulations/src/hmf_omega1_threemode_cutoff12_codex_v40.py`
  - `simulations/src/hmf_omega1_threemode_cutoff12_conv_codex_v40.csv`
  - `simulations/src/hmf_omega1_threemode_cutoff12_log_codex_v40.md`
  - `simulations/src/hmf_omega1_threemode_cutoff12_scan_codex_v40.csv`
  - `simulations/src/hmf_omega1_threemode_cutoff12_summary_codex_v40.csv`

### v41
- Purpose: Three-Mode Resonant-Node Test
- Primary script: `simulations/src/hmf_omega1_resonant_nodes_threemode_codex_v41.py`
- Main log: `simulations/src/hmf_omega1_resonant_nodes_threemode_log_codex_v41.md`
- Key takeaway snippet: Setup: omega_q=1, theta=pi/2, g=0.5, n_modes=3, n_cut=3.
- Artifacts:
  - `simulations/src/hmf_omega1_resonant_nodes_threemode_codex_v41.png`
  - `simulations/src/hmf_omega1_resonant_nodes_threemode_codex_v41.py`
  - `simulations/src/hmf_omega1_resonant_nodes_threemode_log_codex_v41.md`
  - `simulations/src/hmf_omega1_resonant_nodes_threemode_scan_codex_v41.csv`
  - `simulations/src/hmf_omega1_resonant_nodes_threemode_summary_codex_v41.csv`

### v42
- Purpose: Omega_q=1 Best-Kernel Parameter Sweeps
- Primary script: `simulations/src/hmf_omega1_bestkernel_sweeps_codex_v42.py`
- Main log: `simulations/src/hmf_omega1_bestkernel_sweeps_log_codex_v42.md`
- Key takeaway snippet: # Omega_q=1 Best-Kernel Parameter Sweeps (Codex v42)
- Artifacts:
  - `simulations/src/hmf_omega1_bestkernel_sweeps_codex_v42.png`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_codex_v42.py`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_log_codex_v42.md`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_scan_codex_v42.csv`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_summary_codex_v42.csv`

### v43
- Purpose: Omega_q=1 Best-Kernel Sweeps: Discrete vs Continuous Analytic
- Primary script: `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.py`
- Main log: `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_log.md`
- Key takeaway snippet: # Omega_q=1 Best-Kernel Sweeps: Discrete vs Continuous Analytic (Codex v43)
- Artifacts:
  - `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.png`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43.py`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_log.md`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_scan.csv`
  - `simulations/src/hmf_omega1_bestkernel_sweeps_disc_cont_codex_v43_summary.csv`

### v44
- Purpose: Crossover Effects Explainer
- Primary script: `simulations/src/hmf_crossover_effects_explainer_codex_v44.py`
- Main log: `simulations/src/hmf_crossover_effects_explainer_log_codex_v44.md`
- Key takeaway snippet: # Crossover Effects Explainer (Codex v44)
- Artifacts:
  - `simulations/src/hmf_crossover_effects_explainer_codex_v44.png`
  - `simulations/src/hmf_crossover_effects_explainer_codex_v44.py`
  - `simulations/src/hmf_crossover_effects_explainer_log_codex_v44.md`
  - `simulations/src/hmf_crossover_effects_explainer_summary_codex_v44.csv`

### v45
- Purpose: Turnover Magic-Line Formulation
- Primary script: `simulations/src/hmf_turnover_magicline_codex_v45.py`
- Main log: `simulations/src/hmf_turnover_magicline_codex_v45_log.md`
- Key takeaway snippet: # Turnover Magic-Line Formulation (Codex v45)
- Artifacts:
  - `simulations/src/hmf_turnover_magicline_codex_v45.png`
  - `simulations/src/hmf_turnover_magicline_codex_v45.py`
  - `simulations/src/hmf_turnover_magicline_codex_v45_log.md`
  - `simulations/src/hmf_turnover_magicline_codex_v45_summary.csv`

### v46
- Purpose: Execution and safety conventions
- Primary script: none (artifact/log-only).
- Main log: `simulations/src/hmf_agent_handoff_summary_codex_v46.md`
- Key takeaway snippet: - Existing CSV outputs are now sufficient for crossover and turnover analysis; no new ED data is required for those conclusions.
- Artifacts:
  - `simulations/src/hmf_agent_handoff_summary_codex_v46.md`

## Cross-version synthesis
- v8-v11: Isolated mismatch structure in diagonal terms, tested sign/factor variants, and compared inferred factors to kernel-like forms.
- v12-v20: Built and stress-tested best-model/turning-point narratives; established stable evaluation and rapidity/renorm diagnostics.
- v21-v35: Focused on ED truncation/mode convergence and spectral-window dependence; added overlays to compare ordered, scale-bound, best-analytic, and ultrastrong references.
- v36-v45: Established fair-kernel omega_q=1 workflow, demonstrated crossover regimes, and formulated turnover magic-line condition.

## Current baseline for paper figures
- Fair baseline case: `n_modes=2`, `n_cut=6`, window `[0.2,1.8]` at `omega_q=1` (from v38-v43).
- Regime evidence: use v44 crossover figure together with v45 magic-line figure.
- Keep two-mode and three-mode cutoff sweeps (v39/v40) as convergence context panels.