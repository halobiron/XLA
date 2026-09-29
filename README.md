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

## Kaggle: use mounted data, do not download it at runtime

The `169M` progress bar is the CIFAR-100 archive. It is deliberately excluded
from Git because it exceeds GitHub's normal single-file limit. Upload the
existing `data/cifar-100-python/` directory as a private Kaggle Dataset, attach
it to the notebook, and set its mount directory with `--data-root`. The default
configuration now refuses a runtime download.

```powershell
python -m tacsx.cli --config configs/cifar100.yaml --data-root /kaggle/input/<your-cifar100-dataset> --mode full_tacs --name cifar100_kaggle
```

The directory supplied to `--data-root` must directly contain
`cifar-100-python/`. Commit the source files, configuration, and the local
DINOv3 checkpoint already tracked in `checkpoints/`; do not commit `data/`,
`src/tacsx.egg-info/`, or generated outputs. Attach the official DINOv3 source
and weights as Kaggle Datasets too if they are not already present in the
repository checkout.

## Run experiments

Obtain DINOv3 ViT-S/16 weights through Meta's official access flow, set
`backbone.weights` in `configs/cifar100.yaml`, then run one shared CLI for all
baselines and ablations:

```powershell
python -m tacsx.cli --config configs/cifar100.yaml --mode no_context --name cifar100_e0
python -m tacsx.cli --config configs/cifar100.yaml --mode dino_similarity --name cifar100_e2
python -m tacsx.cli --config configs/cifar100.yaml --mode gumbel_only --name cifar100_e3
python -m tacsx.cli --config configs/cifar100.yaml --mode policy_only --name cifar100_e4
python -m tacsx.cli --config configs/cifar100.yaml --mode full_tacs --name cifar100_e5
```

`random_context`, `topk_tacs`, and `adaptive_topk_tacs` are supported by the
same command. Outputs contain config, metrics, history, checkpoint, and
selected-pair metadata. Use `--allow-random-backbone` only for a smoke test;
it is explicitly marked as a fidelity deviation.

For E6/E7, set `topk.k: 2` or `topk.k: 4` in the config before using
`--mode topk_tacs`.

After runs finish, aggregate them with `python -m tacsx.summarize --outputs outputs`.

The source tree already defines reusable interfaces and tests. The Coding Agent should complete DINOv3 integration and the training/evaluation pipeline rather than replacing the project with a different framework.
