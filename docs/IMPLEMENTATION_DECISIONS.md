# Implementation Decisions Log

Update this whenever code makes a choice not explicitly fixed by the paper.

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
