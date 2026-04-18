# Equation Registry (Initial)

Purpose: maintain a stable mapping from legacy equations to theory-first section placement.

| Registry ID | Equation / identity | Legacy source | New destination | Status |
|---|---|---|---|---|
| EQ-001 | \(\chi = s\sqrt{c^2\Sigma_\perp^2+s^2\Sigma_z^2}\) | `05_results_v2.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-002 | \(\gamma(\chi)=\tanh(\chi)/\chi\) | `05_results_v2.tex` | `02_general_duality_framework.tex`, `03_strong_weak_involution.tex` | mapped |
| EQ-003 | \(r=\lvert\tanh\chi\rvert\) | `05_results_v2.tex` | `02_general_duality_framework.tex`, `06_spin_boson_first_example.tex` | mapped |
| EQ-004 | \(\Theta=\operatorname{arctanh}(\gamma s^2\Sigma_z)-\beta\omega_q/2\) | `05_results_v2.tex` | `04_orientation_gauge_and_affine_reflection.tex`, `06_spin_boson_first_example.tex` | mapped |
| EQ-005 | \(m_z=\tanh\Theta\) | `05_results_v2.tex` | `04_orientation_gauge_and_affine_reflection.tex`, `06_spin_boson_first_example.tex` | mapped |
| EQ-006 | \(\tan\eta=-\cot\theta\,(\Sigma_\perp/\Sigma_z)\) | `05_results_v2.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-007 | \(K(u)=g^2K_0(u)\) and \(\Sigma_{z,\perp}\propto g^2\) | `05_results_v2.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-008 | \(\chi(\beta,g,\theta)=g^2\chi_0(\beta,\theta)\) | `06_physical_regimes_v2.tex` | `03_strong_weak_involution.tex` | mapped |
| EQ-009 | \(g_\star(\beta,\theta)=\chi_0^{-1/2}\) | `06_physical_regimes_v2.tex` | `03_strong_weak_involution.tex` | mapped |
| EQ-010 | \(y=\log\chi\) and \(y\mapsto-y\iff \chi\mapsto\chi^{-1}\) | `13_branch_bifurcation_involutive.tex` | `03_strong_weak_involution.tex` | mapped |
| EQ-011 | \(g^\vee=g_\star^2/g=1/(\chi_0g)\) | `mean_force_duality_extended.pdf` | `03_strong_weak_involution.tex` | mapped |
| EQ-012 | \((g^\vee)^\vee=g\) | `mean_force_duality_extended.pdf` | `03_strong_weak_involution.tex` | mapped |
| EQ-013 | \(\chi(\beta,g^\vee,\theta)=1/\chi(\beta,g,\theta)\) | `mean_force_duality_extended.pdf` | `03_strong_weak_involution.tex` | mapped |
| EQ-014 | \(\beff:=-(2/\omega_q)\Theta=\beta-\delta\beta\) | `mean_force_duality_extended.pdf` | `04_orientation_gauge_and_affine_reflection.tex` | mapped |
| EQ-015 | \(\beff'=2\beta-\beff\) | `mean_force_duality_extended.pdf` | `04_orientation_gauge_and_affine_reflection.tex` | mapped |
| EQ-016 | \(R(\beta,\theta)=r(2\beta-\beff)/r(\beff)\) | `mean_force_duality_extended.pdf` | `05_composed_duality_and_constraints.tex` | mapped |
| EQ-017 | \(S=\Delta_0\I+\Delta_z\sigma_z+\Delta_\perp(\sigma_x\cos\phi_f+\sigma_y\sin\phi_f)\) | `10b_appendix_qubit_derivation.tex`, `12_branch_alignment_derivation_codex.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-018 | \(e^S=e^{\Delta_0}\cosh\chi(\I+\gamma M)\) | `12_branch_alignment_derivation_codex.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-019 | \(\rho_\Delta=e^S/\Tr e^S\) | `12_branch_alignment_derivation_codex.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-020 | \(\chi_0\to 0\) as \(\beta\to 0\), \(\chi_0\to\infty\) as \(\beta\to\infty\) implies \(\chi=g^2\chi_0\) endpoint branch exchange and \(g^\vee=1/(\chi_0 g)\to(\infty,0)\) | `mean_force_duality_extended.pdf` | `05_composed_duality_and_constraints.tex` | mapped |
| EQ-021 | \(H_{\mathrm{tot}}=H_Q+H_X+H_{\mathrm{int}},\ H_{\mathrm{int}}=g\,f\otimes B\) | `05_results_v2.tex`, `10b_appendix_qubit_derivation.tex` | `02_general_duality_framework.tex`, `06_spin_boson_first_example.tex` | mapped |
| EQ-022 | \(e^{-\beta H_{\mathrm{MF}}} = Z_X^{-1}\Tr_X e^{-\beta H_{\mathrm{tot}}}\) | `mean_force_duality_extended.pdf` | `02_general_duality_framework.tex` | mapped |
| EQ-023 | \(e^{S_\beta}=e^{+\beta H_Q/2}(\bar{\rho}_Q/Z_X)e^{+\beta H_Q/2}\) | `12_branch_alignment_derivation_codex.tex` | `02_general_duality_framework.tex` | mapped |
| EQ-024 | \((\bar{\rho}_Q/Z_X)=e^{-\beta H_Q/2}e^{S_\beta}e^{-\beta H_Q/2}\) (thermal-plus-influence partition) | `12_branch_alignment_derivation_codex.tex` | `02_general_duality_framework.tex`, `06_spin_boson_first_example.tex` | mapped |
| EQ-025 | \(\nabla y=(\partial_\beta\log\chi_0,2/g)\) with \(y=\log\chi\) | `06_physical_regimes_v2.tex` | `05b_normal_duality_geometrization.tex` | mapped |
| EQ-026 | Normal dual constraint \(y(p^\perp)=-y(p)\) on the same normal ray to \(\mathcal C_\theta\) | new_theory_extension | `05b_normal_duality_geometrization.tex` | mapped |
| EQ-027 | Spectral family \(J_s(\omega)=2g^2\omega_c^{1-s}\omega^s e^{-\omega/\omega_c}\) (sub/Ohmic/super-Ohmic classes) | `06_physical_regimes_v2.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-028 | Exact canonical-channel order-parameter law \(m_z(\beta,g;s)=\tanh[g^2\Sigma_z^{(0)}(\beta;s)-\beta\omega_q/2]\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-029 | Exact low-\(T\) slope and critical relation \(\lambda_s=\lim_{\beta\to\infty}\Sigma_z^{(0)}/\beta,\ g_c^2=\omega_q/(2\lambda_s)\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-030 | Exact spectral-family critical coupling \(g_c^2(s)=\omega_q/(4\omega_c\Gamma(s))\) for \(J_s(\omega)=2g^2\omega_c^{1-s}\omega^s e^{-\omega/\omega_c}\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-031 | Qubit entropy from Bloch radius \(S_Q=-\sum_\pm \lambda_\pm\log\lambda_\pm,\ \lambda_\pm=(1\pm r)/2\) with \(r=|\tanh\chi|\) | `05_results_v2.tex` | `06_spin_boson_first_example.tex` | mapped |
| EQ-032 | Closed entropy functional \(S_Q(\beta,g,\theta)=h_2((1+|\tanh\chi|)/2)\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-033 | Dual entropy representative \(S_Q(\beta,g^\vee,\theta)=h_2((1+|\tanh(\chi^{-1})|)/2)\) with \(\chi^\vee=\chi^{-1}\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-034 | Weak-branch entropy asymptotic \(S_Q(\chi)=\log 2-\chi^2/2+O(\chi^4)\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-035 | Strong-branch entropy asymptotic \(S_Q(\chi)=(1+2\chi)e^{-2\chi}+O(e^{-4\chi})\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |
| EQ-036 | Dual-weak coordinate form \(S_Q=(1+2/\chi^\vee)e^{-2/\chi^\vee}+O(e^{-4/\chi^\vee})\) | new_theory_extension | `06_spin_boson_first_example.tex` | mapped |

## Registry Policy

1. Any equation promoted to the main text gets a stable `EQ-XXX` ID.
2. If notation changes, update equation text but keep the same `EQ-XXX` ID.
3. If a legacy equation is split into multiple statements, create suffix IDs (`EQ-XXXa`, `EQ-XXXb`).
