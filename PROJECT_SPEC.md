# Project Specification — TACS-X

## 1. Problem statement

Nearest-neighbor retrieval asks **"what looks similar?"**. TACS learns **"what context actually helps the downstream model make a better decision?"**

## 2. Core components

### Candidate Pool

Build a fixed pool from a configurable percentage of training data. Default target: **20%**.

For each query batch, sample `N_c` candidates (e.g. 16 or 32) from that pool to keep compute tractable.

Prevent:
- exact query duplication;
- test leakage;
- identity/patient leakage where applicable.

### Selector

Input:
- query `[B,C,H,W]`
- candidates `[B,N,C,H,W]`

Output:
- utility scores `[B,N]`

Reference implementation:
- shared encoder;
- optional small projection head;
- L2 normalize;
- dot-product/cosine utility.

### Differentiable path

Use:

```python
F.gumbel_softmax(scores, tau=tau, hard=True, dim=-1)
```

Forward is discrete; backward uses straight-through relaxation.

### Policy path

Sample from `Categorical(logits=scores)` and compute:

```text
reward = CE(f_d(query, None), y) - CE(f_d(query, selected_context), y)
```

Detach task losses when forming reward.

Standardize in-batch:

```text
A = (r - mean(r)) / (std(r) + eps)
```

Policy loss:

```text
L_policy = -mean(log_prob(action) * A)
```

### Downstream model

Classification target:
- ViT-S/16 initialized from DINOv3;
- concatenate query and context patch embeddings before transformer blocks;
- support `context=None`.

### Joint objective

```text
L_total = L_grad + lambda_policy * L_policy
```

Defaults:
- `tau = 0.1`
- `lambda_policy = 1.0`

## 3. Required modes

- `no_context`
- `random_context`
- `dino_similarity`
- `gumbel_only`
- `policy_only`
- `full_tacs`

Optional:
- `topk_tacs`
- `adaptive_topk_tacs`

## 4. Metrics

Classification:
- top-1 accuracy
- average task loss

Retrieval analysis:
- cross-class selection rate
- mean query/context cosine similarity
- reward distribution
- selection entropy

Optional:
- LPIPS
- attention rollout

## 5. Run outputs

```text
outputs/<run_name>/
  config.yaml
  metrics.json
  history.csv
  checkpoint_best.pt
  selected_pairs.csv
  figures/
```

Aggregate:
```text
outputs/summary.csv
outputs/ablation_table.md
```

## 6. Scientific success criteria

Do not define success as matching CVPR numbers. Seek these trends:

- learned TACS > random context;
- learned retrieval competitive with or better than frozen DINO similarity;
- hybrid objective at least as stable/useful as either single path;
- qualitative selections differ meaningfully from nearest-neighbor retrieval.

If results differ, analyze and report causes transparently.
