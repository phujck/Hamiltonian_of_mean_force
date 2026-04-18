# Impact Upgrade Plan (Nature-Level Push)

## Objective
Increase persuasive strength, readability, and submission impact by adding high-information visuals, clarifying where results are exact versus approximate, and tightening theory-to-example flow.

## Priority Stack

1. **Visual explanatory power (highest ROI)**
   - Add a spectral-class phase diagram in \((s,g)\) showing localized/delocalized regions.
   - Add an orientation-gauge Bloch visualization showing \(m\mapsto m'\) reflection and unchanged radius.
   - Acceptance:
     - Figures are legible at one-column width.
     - Captions are self-contained and tied to named results in Sections 04 and 06.
     - Reproducible scripts live in `validation/`.

2. **Theory/example bridge refinement**
   - Add short transition text linking formal gauge involution to geometric action on Bloch vectors.
   - Add short text linking phase-boundary equations to plotted regions and approximation scope.
   - Acceptance:
     - Each new figure is explicitly interpreted in one paragraph directly below insertion.

3. **Claim strength hardening**
   - Mark the sub-Ohmic boundary as an analytic adiabatic estimator.
   - Keep Ohmic KT and super-Ohmic no-transition statements visibly separated by method class.
   - Acceptance:
     - No caption or body text implies stronger rigor than the cited source class supports.

4. **Next-pass upgrades (after current turn)**
   - Add a compact table of symbols/representatives (\(\chi,y,g^\vee,\beff\)).
   - Add a small appendix derivation for normal-duality local construction algorithm.
   - Add one additional validation panel comparing direct numerics and dual asymptotics for an oriented observable.

## This Turn Implementation

- Deliverables:
  - `validation/make_spin_boson_phase_diagram_figure.py`
  - `validation/make_orientation_gauge_bloch_figure.py`
  - new figure assets in `manuscript/figures/`
  - manuscript insertions in Sections `04` and `06`
- Verification:
  - `py paper_dual_hmf/validation/check_duality_identities.py`
  - `pdflatex -> bibtex -> pdflatex -> pdflatex`
