# paper_dual_hmf

Theory-first manuscript workspace for:

- dual representatives of mean-force equilibrium,
- strong-weak involution in the crossover coordinate,
- orientation-gauge affine temperature reflection,
- spin-boson as Example 1 (not the conceptual centre).

## Phase Scope (Current)

This phase scaffolds the project and defines the writing/coordination interfaces:

1. Manuscript tree and section stubs.
2. Coordination artifacts (`WORKLOG.md`, `TASK_GRAPH.yaml`, `HANDOFF.md`).
3. Source and equation mapping from the current clean draft.
4. Minimal validation stub for core algebraic identities.

No full prose drafting is included beyond section-level claim/theorem headers.

## Directory Layout

```text
paper_dual_hmf/
  HANDOFF.md
  README.md
  TASK_GRAPH.yaml
  WORKLOG.md
  manuscript/
    figures/
    tex/
      main.tex
      sections/
  notes/
    equation_registry.md
    source_map.md
  validation/
    check_duality_identities.py
```

## Build (RevTeX Draft)

From repository root:

```powershell
cd paper_dual_hmf/manuscript/tex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
