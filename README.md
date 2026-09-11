# TACS-X Coding Agent Pack

Starter pack for reproducing and extending **Task-Aligned Context Selection (TACS)** from:

> Jingyu Guo et al., *Learning What Helps: Task-Aligned Context Selection for Vision Tasks*, CVPR 2026.

## Goal

Build a compact, defensible reproduction for image classification, then add one small extension:

- DINOv3 ViT-S/16 backbone.
- Fixed candidate pool.
- Learned selector.
- Straight-through Gumbel-Softmax path.
- Policy-gradient path with downstream task reward.
- Dual-image downstream ViT.
- Baselines: No Context, Random, Frozen DINO similarity.
- Ablations: Gumbel-only, Policy-only, Full TACS.
- Extension: Top-K / optional adaptive-K context.
- Interpretability: retrieval pairs, cross-class selection, attention rollout if feasible.

## Recommended first dataset

Use **CIFAR-100** first for fast iteration. Add Oxford-IIIT Pets or CUB-200 only after the core system is stable.

## Start here

Read in order:

1. `AGENTS.md`
2. `PROJECT_SPEC.md`
3. `docs/PAPER_NOTES.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/EXPERIMENT_MATRIX.md`
6. `prompts/AGENT_BOOTSTRAP.md`

Then:

```bash
bash scripts/bootstrap.sh
pytest -q
```

The source tree already defines reusable interfaces and tests. The Coding Agent should complete DINOv3 integration and the training/evaluation pipeline rather than replacing the project with a different framework.
