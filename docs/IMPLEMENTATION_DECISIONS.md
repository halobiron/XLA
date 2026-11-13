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
Run `scripts/diagnose_fidelity.py` for validation-only, no-training probes.
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
