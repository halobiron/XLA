# Bootstrap Prompt for Coding Agent

You are implementing TACS-X from this repository.

Before coding:

1. Read `AGENTS.md`.
2. Read `PROJECT_SPEC.md`.
3. Read `docs/PAPER_NOTES.md`.
4. Read `docs/IMPLEMENTATION_PLAN.md`.
5. Read `docs/EXPERIMENT_MATRIX.md`.
6. Inspect `paper/TACS_CVPR_2026.pdf`.
7. Inspect the official DINOv3 repository after bootstrap.

Then:

- audit the skeleton;
- produce a short implementation checklist;
- implement phase-by-phase;
- run unit tests after each phase;
- keep all experiment modes behind the same config/CLI;
- preserve TODOs until complete;
- log paper ambiguities in `docs/IMPLEMENTATION_DECISIONS.md`.

First milestone:
- No-Context CIFAR-100 end-to-end;
- then static DINO retrieval;
- only then learned selection.
