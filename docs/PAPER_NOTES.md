# Paper Notes — TACS (CVPR 2026)

The PDF in `paper/` is authoritative.

## Core idea

TACS learns which contextual image is **useful** for the downstream task rather than retrieving solely by visual similarity.

It has:
- a Selector;
- a downstream Task Network.

## Selector

The selector encodes query and candidate images, assigns utility scores, converts them to selection probabilities, and uses the highest-probability candidate at inference.

## Hybrid optimization

### Differentiable path
Uses straight-through Gumbel-Softmax for hard categorical selection with gradient flow.

Paper default: `tau = 0.1`.

### Policy path
The selector is also a policy over candidate indices.

Reward:
```text
task_loss_without_context - task_loss_with_selected_context
```

Positive reward means the context helps.

The paper standardizes reward in-batch and applies a REINFORCE-style policy loss.

### Joint objective

```text
L_TACS = L_grad + lambda * L_policy
```

Paper default: `lambda = 1.0`.

## Classification architecture

The paper concatenates patch embeddings of query and paired image before transformer blocks, then fine-tunes the downstream model.

## Candidate pool

The paper uses a fixed candidate pool from 20% of training images in most experiments and excludes query-associated images to prevent leakage.

## Important ablation

- Frozen DINO similarity retrieval.
- Differentiable selection, no policy.
- Policy-based selection, no Gumbel-Softmax.
- Full dual optimization.

## Interpretability finding

TACS often retrieves more cross-class and perceptually diverse examples than frozen similarity retrieval, supporting the idea of complementary/contrastive context.

## Explicit paper limitations

- fixed candidate pool;
- one retrieved example per query.

These motivate the Top-K extension.

## Project scope

Implement **classification first**. Segmentation is intentionally out of scope for the initial version because it adds gated cross-attention and a DPT head.
