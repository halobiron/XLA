# Implementation Plan

## Phase 0 — Environment

1. Clone official DINOv3 into `third_party/dinov3`.
2. Obtain ViT-S/16 weights via the official access flow.
3. Install this package and dependencies.
4. Run dummy-encoder unit tests.

Exit: `pytest -q` passes.

## Phase 1 — No-Context baseline

Implement:
- CIFAR-100 loader;
- DINOv3 adapter;
- query-only classifier;
- train/eval loop;
- metrics/checkpoints.

Exit:
- trains without NaNs;
- validation accuracy rises above chance;
- outputs are saved.

## Phase 2 — Candidate pool + static retrieval

Implement:
- fixed train-only candidate pool;
- random context;
- frozen DINO cosine retrieval.

Exit:
- `random_context` and `dino_similarity` run;
- selected-pair metadata includes IDs, labels, similarity.

## Phase 3 — Dual-input model

Implement query/context patch-token concatenation before transformer processing.

Exit:
- `context=None` and real context both work;
- positional embedding handling is documented.

## Phase 4 — Gumbel path

Implement:
- selector;
- projection;
- scores;
- ST Gumbel-Softmax;
- context extraction.

Exit:
- `gumbel_only` trains;
- selector gets gradients.

## Phase 5 — Policy path

Implement:
- Categorical sampling;
- query-only baseline loss;
- selected-context loss;
- detached reward;
- standardized advantage;
- REINFORCE loss.

Exit:
- `policy_only` trains;
- policy loss finite;
- reward sign verified.

## Phase 6 — Full TACS

Combine both paths.

If exact two-path sampling semantics require an implementation choice, record it in `docs/IMPLEMENTATION_DECISIONS.md`.

Exit:
- full hybrid training completes;
- evaluation is comparable across baselines.

## Phase 7 — Analysis

Implement:
- ablation table;
- cross-class selection rate;
- selected-pair visualization;
- selector entropy;
- reward histograms.

## Phase 8 — TACS-X

Top-K:
- take top K scores;
- normalize weights;
- aggregate context using a documented mechanism.

Optional adaptive K:
- choose smallest K whose cumulative selector probability exceeds a threshold.

## Phase 9 — Demo

Minimal Gradio:
- query image;
- DINO nearest neighbor;
- TACS context;
- prediction without/with context;
- utility/selection score;
- optional attention rollout.
