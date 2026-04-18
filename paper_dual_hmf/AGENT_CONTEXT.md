# AGENT CONTEXT: `paper_dual_hmf`

Last updated (UTC): `2026-02-27T00:49:06Z`

## Mission

Produce a **submission-ready, Nature-level manuscript** on dual representatives of mean-force equilibrium, with a **theory-first narrative** and spin-boson used only as Example 1.

## Definition Of Done (Target State)

1. Main text is conceptually sharp, mathematically rigorous, and editorially polished to high-impact journal standards.
2. Claims are clearly typed (`Theorem`, `Lemma`, `Proposition`, `Corollary`) or explicitly marked as interpretation.
3. Citation coverage is complete for all nontrivial claims, with no weakly supported assertions.
4. Figure set is minimal but publication quality, with captions that stand alone.
5. Reproducibility artifacts (scripts/checks) are stable and side-effect free.
6. Manuscript compiles cleanly and consistently (`pdflatex`, `bibtex`, reruns) with no undefined refs/citations.

## Current Project State

1. Theory-first structure implemented in `manuscript/tex/main.tex` with sections `01`-`09`.
2. Core formal duality statements are present in Sections `02`-`05`.
3. Spin-boson is framed as a post-theory worked example in Section `06`.
4. Core validation figure exists:
   - `manuscript/figures/hmf_dual_map_core.pdf`
   - `manuscript/figures/hmf_dual_map_core.png`
5. Validation scripts exist and pass:
   - `validation/check_duality_identities.py`
6. Coordination artifacts are active:
   - `WORKLOG.md`
   - `TASK_GRAPH.yaml`
   - `HANDOFF.md`
   - `notes/equation_registry.md`
   - `notes/source_map.md`

## Remaining Gaps To Nature-Level Quality

1. Citation depth is still thin in theory sections (`02`-`05`).
2. Narrative polish is mid-draft: precision is strong, but flow/positioning language needs higher editorial finish.
3. Some TeX underfull warnings remain from dense inline expressions and long path strings.
4. Figure/story coupling should be tightened so each figure directly supports one main claim.
5. Discussion/outlook should sharpen scope boundaries and high-impact framing.

## Autonomous Execution Rules (For Future Agents)

1. Preserve theory-first ordering; do not drift into model-first exposition.
2. Keep spin-boson explicitly as Example 1, not the conceptual center.
3. Every turn should update `WORKLOG.md`; update `TASK_GRAPH.yaml`/`HANDOFF.md` when task state changes.
4. Maintain equation traceability in `notes/equation_registry.md` for new promoted statements.
5. Re-run validation + compile after substantial edits:
   - `py paper_dual_hmf/validation/check_duality_identities.py`
   - `pdflatex -interaction=nonstopmode -halt-on-error main.tex` (repeat as needed)
   - `bibtex main` when bibliography-affecting edits occur.

## Prioritized To-Do List

### P0: Scientific + Structural Integrity

1. Audit Sections `02`-`05` line-by-line to ensure each statement has one of:
   - proof/derivation in main text or appendix
   - explicit interpretation label
   - direct citation support.
2. Expand theorem-level citation support in `01`, `02`, `03`, `04`, `05`.
3. Add a short "Scope and non-claims" paragraph in `08` that prevents over-interpretation.

### P1: Editorial Polish (Nature-level Readability)

1. Rewrite abstract and opening two introduction paragraphs for stronger problem-solution-impact arc.
2. Tighten transitions between Sections `03` -> `04` -> `05` so composition logic reads as one continuous argument.
3. Reduce symbol density where possible (especially long inline formula stretches).
4. Standardize notation and terminology globally (`representative`, `control-regime transfer`, `orientation gauge`, `asymptotic exchange`).

### P2: Figures + Validation Presentation

1. Refine Figure 1 caption in Section `07` so it is fully self-contained and claim-linked.
2. Ensure axis labels, legend entries, and panel annotations use publication-ready wording.
3. Add one short reproducibility note (inputs/assumptions/command) in Section `07` or appendix.

### P3: Final Submission Readiness

1. Final compile hygiene pass: remove avoidable underfull/overfull warnings through line breaks and wording.
2. Run full compile chain and record exact command transcript summary in `WORKLOG.md`.
3. Final internal checklist:
   - all references resolved
   - all claims mapped
   - all figures present and cited
   - bibliography style coherent
   - no placeholder text.

## Suggested Next Atomic Task

"Complete a citation-and-claim audit for Sections `02`-`05`, then patch those sections and append supporting references in one integrated pass."
