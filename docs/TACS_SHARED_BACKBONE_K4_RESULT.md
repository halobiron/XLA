# TACS-X shared-backbone K=4 result

## Result that can be reported

On the preselected Gaussian-noise follow-up, Full TACS was more accurate than
the independently trained No-Context model. Mean top-1 accuracy across the two
fixed noise levels (`sigma=0.08` and `0.16`) was higher by **0.736 percentage
points** on 9,988 identity-filtered CIFAR-100 test images. The two-sided 95%
paired query-cluster bootstrap interval was **[+0.290, +1.182] points**.

| Test condition | No-Context | Full TACS | Difference |
|---|---:|---:|---:|
| Gaussian noise, sigma 0.08 | 48.04% | 48.25% | +0.21 pp |
| Gaussian noise, sigma 0.16 | 16.76% | 18.02% | +1.26 pp |
| Mean of the two conditions, primary endpoint | — | — | **+0.736 pp (95% CI +0.290 to +1.182)** |

This supports a **narrow robustness result for the tested Gaussian-noise
conditions**. It does not show a clean-image accuracy gain or an overall gain
across corruptions.

## Validation results and endpoint choice

Before looking at test results, the full 4,996-image validation set was scored
under the registered seven-comparison family. No metric passed that familywise
gate. Clean accuracy was 89.79% for Full TACS and 90.07% for No-Context
(`difference=-0.280 pp`, 99.286% CI `[-1.021, +0.466]`). Mean accuracy over the
eight project corruptions was 68.98% versus 70.05% (`difference=-1.071 pp`, CI
`[-1.515, -0.648]`). Clean NLL reduction was `-0.00893` (CI
`[-0.02250, +0.00515]`).

Gaussian-noise mean accuracy was the **only positive point estimate** among
those seven validation comparisons: `+0.641 pp`, 99.286% CI `[-0.220,
+1.511]`. Because the interval crossed zero, it did not pass the original
validation gate. A separate follow-up protocol then fixed that single endpoint
before test-label deserialization. The test follow-up made no model or selector
updates and reported no test-selected checkpoint.

The full validation result, including every negative metric and corruption,
is in [`validation_result.json`](../outputs/tacs_shared_backbone_k4_v1/evaluation/validation_result.json).

## Experiment setup

- Official DINOv3 ViT-S/16 initialized both task arms from the same five-epoch
  No-Context checkpoint. The selector and dual-image task network referenced
  the **same trainable ViT adapter**.
- The task network concatenated query and selected-context patch tokens before
  the transformer blocks. Full TACS used hard straight-through Gumbel with
  `tau=0.1`, detached CE-difference reward, REINFORCE, and `lambda_policy=1`.
- DINO proposed the nearest four images from a fixed 9,000-image training-only
  pool. Candidate labels were not used. Exact pixel identities were excluded.
- Both task arms had the same seed (42), 10 continuation epochs, batch size
  32, optimizer schedule, and 224-pixel input size. All 12 task ViT blocks were
  trainable. The No-Context checkpoint was reused from an already completed
  arm after matching its seed, data, initialization, optimizer, and schedule;
  see [`baseline_provenance.json`](../outputs/tacs_shared_backbone_k4_v1/no_context/baseline_provenance.json).
- To fit the compute budget, the shortlist used four proposals and the
  candidate-image encoder path was stop-gradient, with embeddings refreshed
  from the current shared ViT each training step. The shared query-selector
  path and task path both updated the shared backbone. The selector projector
  was a trainable identity-initialized linear layer.
- The best Full TACS checkpoint was selected on validation at epoch 9. The
  test evaluation excluded 12 test images pixel-identical to training images
  or repeated within the held-out set, leaving 9,988 queries.

An audit found 188 state-dict tensors in the shared backbone aliases, with
**zero mismatches** between `selector.encoder` and `task_model.backbone`; all
12 transformer blocks changed from their common initialization.

## Scope and limitations

This is a one-seed **TACS-X mini-reproduction**, not an exact reproduction of
the paper. The paper reports 100 training epochs and three seeds. This run used
10 continuation epochs, one seed, a nearest-four proposal shortlist, a
trainable projection, and a stop-gradient candidate-image encoding path.
Gaussian noise here is the project's custom transform on native 32-pixel
images; it is **not CIFAR-100-C and not a robustness benchmark reported by the
paper**.

The official CIFAR-100 test split was used earlier in this workspace for
unrelated residual experiments and a clean-image comparison. The present
checkpoint and single follow-up endpoint were selected without using its
labels, but the split is not historically pristine. Treat the result as
exploratory evidence for this one noise endpoint and seek a fresh dataset or
an independently collected test set, plus multiple seeds, before making a
broad claim that TACS is generally superior to No-Context.

## Reproducibility artifacts

- Run plan: [`plan.json`](../outputs/tacs_shared_backbone_k4_v1/plan.json)
- Training histories: [`full_tacs/history.json`](../outputs/tacs_shared_backbone_k4_v1/full_tacs/history.json)
- Shared-backbone checkpoint: [`full_tacs/checkpoint_best.pt`](../outputs/tacs_shared_backbone_k4_v1/full_tacs/checkpoint_best.pt)
- Follow-up protocol and checkpoint/data lock:
  [`prospective_protocol.json`](../outputs/tacs_shared_backbone_k4_v1/evaluation/gaussian_noise_followup/prospective_protocol.json),
  [`checkpoint_lock.json`](../outputs/tacs_shared_backbone_k4_v1/evaluation/gaussian_noise_followup/checkpoint_lock.json)
- Test predictions and selected contexts:
  [`test_predictions.pt`](../outputs/tacs_shared_backbone_k4_v1/evaluation/gaussian_noise_followup/test_predictions.pt)
- Test result:
  [`test_result.json`](../outputs/tacs_shared_backbone_k4_v1/evaluation/gaussian_noise_followup/test_result.json)
- Evaluator: [`evaluate_gaussian_noise_followup.py`](../scripts/evaluate_gaussian_noise_followup.py)
