# Implementation Decisions Log

Update this whenever code makes a choice not explicitly fixed by the paper.

### 2026-11-14 — Correct pair positions and preserve legacy checkpoints

**Evidence**
Image-major patch concatenation was assigned a row-major horizontal RoPE grid.
On 64 validation images with the same frozen official weights and query-only
head, duplicate-query context accuracy was 65.63% with the old mapping and
78.13% after correcting the mapping (query-only: 79.69%). This is a diagnostic,
not a retrained benchmark. The official query encoder and adapter match exactly.

**Decision**
Keep concatenation before transformer blocks. Put all position handling in
`DINOv3Adapter`; expose `legacy`, corrected `horizontal`, and native-grid
`per_image` layouts. New repository configs use horizontal. Configs without
the key keep legacy semantics so existing checkpoint predictions do not change.
Regenerate RoPE per block as official DINOv3 does, rather than once per forward.

**Impact on fidelity**
The paper does not specify the multi-image RoPE scheme; neither corrected
horizontal coordinates nor per-image coordinates are claimed to be author code.
Classification backbones must adapt to paired inputs; frozen/partial models
remain explicit compute-limited deviations from full fine-tuning.

**Validation and next step**
The historical CIFAR diagnostic script was retired during migration to CUB.
Use the adapter unit tests to verify token ordering without downloading a model.
Retrain matched baselines and TACS under trainer v3 with new run names. The old
unf4 checkpoint changed all 12 blocks; it does not test the corrected partial
fine-tuning configuration. Test-time argmax and corrected adaptive coverage also
change protocol; do not relabel historical metrics as new results.

### 2026-11-14 — Align reward metadata and freeze baseline retrieval separately

**Decision**
Use the Gumbel-selected action for the policy term in Full TACS and detach the
already-computed per-item task loss for its reward. This makes the stored action,
context and reward refer to the same pair and saves a downstream forward. Sharing
the action is an implementation choice, not a requirement explicitly stated by
the paper. Both estimators still update the selector, with no task-model gradient
from the reward.

When the downstream DINO-similarity baseline fine-tunes, keep an independent,
frozen official encoder for both query and cached candidate retrieval embeddings.
Only fully frozen adapters are forced into eval mode during training, so trainable
backbones retain official position augmentation. Record trainer_version=3,
layout, selection rule and trainable parameter counts in new metrics.

## Template

### YYYY-MM-DD — Decision title

**Question**
What was ambiguous or resource-constrained?

**Decision**
What was implemented?

**Reason**
Why?

**Impact on fidelity**
How could this differ from the paper?

**Experiment needed**
How will it be validated?

### 2026-09-11 — DINOv3 pair-token RoPE layout

**Question**
The paper specifies concatenating query/context patch embeddings before transformer blocks, but not how DINOv3's RoPE coordinates should be assigned to the longer sequence.

**Decision**
TACS-X retains query CLS/storage tokens once and concatenates query then context patch tokens. It presents them to DINOv3 RoPE as a horizontal `H x 2W` patch grid.

**Reason**
This gives every patch a unique, deterministic position while preserving native spatial adjacency inside each image. DINOv3 has RoPE rather than an interpolated absolute position table.

**Impact on fidelity**
The paper does not disclose an equivalent coordinate layout, so cross-image relative positions may differ from the authors' implementation.

**Experiment needed**
Compare this layout against a two-grid/block-specific RoPE implementation if original code becomes available.

### 2026-09-11 — Missing official weights fail closed

**Question**
Official DINOv3 ViT-S/16 weights require the official access/download flow and are not bundled in this repository.

**Decision**
Normal training refuses to start without `backbone.weights`. `--allow-random-backbone` is an explicit smoke-test escape hatch and is saved in metrics.

**Reason**
This avoids silently replacing official DINOv3 initialization or making a misleading reproduction claim.

**Impact on fidelity**
Random-backbone smoke results are not comparable with TACS/DINOv3 results.

**Experiment needed**
Run the full experiment matrix after supplying official weights.

### 2026-11-18 — Anchored TACS-X: trained query-only path and abstain gate

> **Current state:** the null option described below (`selector.null_candidate`, `null_logit`, `lr_null`) has been **removed** and replaced by `selector.abstain_gate` (`--abstain-gate`, `training.lr_gate`, default 1e-3). See the last subsection. The null text is kept as history.

**Question**
On CUB (quick warm-start, epoch 9) the TACS model's own query-only accuracy fell to 81.82% versus 84.34% for No-Context, about 2.5pp of the 3.0pp loss. The paper reward baseline `CE(f(q, none))` is computed under `no_grad` and that input format is never trained, so it drifts out of distribution. Train `mean_reward` rose from -0.19 to +0.0145 while validation stayed below No-Context.

**Decision**
Two opt-in flags, both off by default (paper path unchanged):
- `policy.aux_query_weight` (`--aux-query-weight`): loss `CE(q,c) + beta*CE(q,none) + lambda*L_policy`; the reward remains detached.
- `selector.null_candidate` (`--null-candidate`): an extra "no context" action. Its score is the detached best real-candidate score plus a learnable `null_logit` (selector learning rate). Its reward is exactly zero. The null decision is learned only by REINFORCE and applies only at inference, where choosing null predicts query-only. The context path is always trained on every query, using the best Gumbel-perturbed real candidate. (The first version mixed `m0*f(q,none) + (1-m0)*f(q,c)`; it was replaced after the collapse described below.)

**Reason**
Keep the query-only capability that defines the reward, and let the model abstain when no candidate helps.

**Impact on fidelity**
Explicit TACS-X extension, not in the paper. Both flags are logged as deviations by the CLI. The query-only forward was already computed for the reward, so the extra cost is one backward through it.

**Experiment needed**
Matched multi-seed comparison against No-Context with validation-selected checkpoints, then official test once.

GPU smoke tests (`scripts/smoke_cub_anchored.py`; logs `outputs/cub200_anchored_smoke_s42*.log`), full 1080-image pool, batch 16, starting from the query-only warm start at absolute epoch 5:
- Losses were finite and the reward was detached. Gradients were finite except during the usual first GradScaler steps (4 paper, 5 anchored). Step time was 0.33 s vs 0.29 s for paper TACS (+15%); peak VRAM 3.25 vs 2.65 GiB. Zero test-image opens; the warm-start hash was unchanged.
- At the selector learning rate (1e-4), `null_logit` moved only 0.0025 in 40 steps. It now has its own group: `training.lr_null`, default 1e-2, no weight decay.
- With `lr_null` 1e-2, after 80 steps `null_logit` = 0.51 and validation argmax chose null for 128/128 queries. Early context rewards are negative, and null-selected rows give no gradient to the context path, so the null option can lock in. This is a collapse risk to resolve before a long run.

Smoke v2 (`outputs/cub200_anchored_smoke_s42_v2`) used the decoupled null, 300 steps per variant, and 256 validation queries:
- The context path now trains as in paper TACS: task loss 1.98 → 1.57, versus 1.90 → 1.58 for paper TACS.
- The mean context reward stayed negative (−0.50 → −0.09). REINFORCE therefore raised `null_logit` steadily, about +0.33 per 50 steps, to 1.98. The training null rate rose from 15% to 43%.
- Structural limitation: the null score is the best candidate score plus a global offset. Once the offset is > 0, argmax chooses null for every query (256/256 here), so it is a global on/off switch, not per-query abstention. If null sampling saturates, every action has reward 0, the advantage vanishes, and the policy stops exploring context. A query-dependent gate, trained without stopping context sampling, is needed before a long run.

**Replacement: per-query abstain gate.** Selection and training follow the paper path exactly. Context is always sampled and trained, the policy sees every reward, and `aux_query_weight` keeps the query-only path trained. A small gate (BatchNorm without affine parameters, then MLP 11→16→1, last layer zero-initialized) decides per query, at inference, whether to use the context prediction (logit ≥ 0) or fall back to query-only.
- Gate inputs are 11 label-free, detached features: query-only and context-path confidence (max probability, margin, entropy), whether the two predictions agree, each path's probability for the other path's class, and the retrieval top score and top1−top2 gap.
- It is trained only by a BCE on `sign(reward)`, weighted by `|reward|`, using the reward already computed for every training query. No other parameter receives the gate gradient.
- At initialization the logit is 0, which selects context, so behavior equals paper TACS.
- Known risk: the gate learns from training queries the task model has fit. Their reward distribution may differ from validation, which is the failure mode of the earlier selector heads.
