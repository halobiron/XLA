# Instructions for Coding Agents

## Mission

Implement **TACS-X** faithfully enough to reproduce the qualitative behavior and ablation logic of TACS, while keeping the codebase small, testable, and practical on limited compute.

## Non-negotiable constraints

- Use PyTorch.
- Prefer official **DINOv3 ViT-S/16** when weights are available.
- Do not silently substitute CLIP/DINOv2.
- Keep TACS logic separated into selector, differentiable selection, policy optimization, and downstream dual-image model.
- Keep baseline architectures/training settings comparable.
- Prevent candidate/query leakage and test leakage.
- Make experiment modes selectable from config/CLI.
- Save config, metrics, checkpoints, and selected-pair metadata for each run.
- Use deterministic seeds where practical.

## Scientific fidelity

Reference behavior:

- Selector and downstream network use ViT-S/16 initialized from DINOv3.
- For classification, concatenate query and selected-context patch embeddings before transformer blocks.
- Selector scores define a categorical distribution over candidates.
- Differentiable path uses straight-through Gumbel-Softmax.
- Policy reward is:
  `CE(query-only) - CE(query + selected-context)`.
- Policy loss is REINFORCE-style:
  `-log_prob(action) * advantage`.
- Overall objective combines task loss and policy loss.
- Preserve paper defaults unless intentionally changed:
  - `tau = 0.1`
  - `lambda_policy = 1.0`
  - fixed candidate pool derived from training data.

## Practical deviations allowed

For a mini reproduction:
- fewer epochs,
- smaller batch size,
- smaller sampled candidate set,
- frozen/partially frozen backbone,
- 224x224 if necessary.

Log every deviation and disclose it in the report.

## Adapter rule

Do not copy DINOv3 internals throughout the repo. Implement a single `DINOv3Adapter` exposing:

- `encode_global(images) -> [B,D]`
- `prepare_tokens(images)`
- `forward_blocks(tokens)`
- `embedding_dim`
- `patch_size`

Keep version-specific logic in that adapter.

## Minimum tests

- selector score shape;
- hard Gumbel one-hot behavior;
- policy loss finite gradients;
- reward sign;
- candidate pool excludes query;
- Top-K weights sum to 1.

Tests must use dummy encoders and must not require model downloads.

## Completion criteria

The project is complete only when it can:

1. train/evaluate No-Context;
2. train/evaluate DINO-similarity baseline;
3. run differentiable-only selector;
4. run policy-only selector;
5. run full hybrid TACS;
6. emit comparison CSV/JSON;
7. render retrieval examples;
8. optionally run Top-K TACS-X.

## Do not

- Do not rebuild around a huge external framework.
- Do not claim exact reproduction if the setup differs.
- Do not use test labels/data to build the candidate pool.
- Do not allow self-retrieval except explicit ablations.
- Do not backpropagate through detached policy reward.
