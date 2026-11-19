# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS opened by double-click (no server, no network). A Python builder (`scripts/build_cub_demo.py`) bakes data from `outputs/` and the saved checkpoints into a local JS data file next to the page. An optional local server (`scripts/serve_demo.py`, localhost only, CUDA) serves the same page and adds live inference on uploaded images.

## Users

Students presenting a university computer-vision course project (XLA) to an instructor and a grading committee. Secondary: teammates inspecting runs. The interface is in Vietnamese. ML technical terms stay in English (reward, selector, context, query, No-Context, seed, checkpoint, validation, test, self-training, ...).

## Product Purpose

TACS-X is a compact PyTorch implementation of Task-Aligned Context Selection (Guo et al., CVPR 2026) on CUB-200-2011 with an official DINOv3 ViT-S/16 backbone. The demo shows the committee three things:
- what was measured;
- which three changes produced the result;
- how one prediction is made, step by step, for No-Context and TACS-X on the same test image.

## Positioning

An honest, pre-registered result with a deliberately narrow claim. On CUB with 2 labeled images per class, after one round of class-balanced self-training applied identically to both models:
- TACS-X reaches 63.48% on the official test split;
- No-Context reaches 60.61%;
- the difference is +2.87pp, query-bootstrap 95% CI [+2.40, +3.34];
- all 4 seeds are positive (+1.73 to +3.68), and the seed-level t-interval [+1.43, +4.31] also excludes 0.

The result is built from three changes, shown in this order (official test, seeds 10–13):

1. **Anchored training.** CE(query-only) trains the reward baseline. No-Context 53.41% → TACS-X 56.08%, +2.67pp.
2. **Context pool of every training-split image, labels never read.** 56.08% → 56.75%, +0.67pp, CI [+0.23, +1.10].
3. **Class-balanced self-training.** The model pseudo-labels the unlabeled training images and keeps the 5 most confident per predicted class (≈1,000 images, ≈80% correct). 56.75% → 63.48%, +6.73pp. No-Context gains +7.20pp from the same procedure.

**Mechanism.** Most of the final advantage comes from training: TACS-X without its context image is still +2.33pp above No-Context. The context image at prediction time adds +0.54pp, CI [+0.17, +0.91].

**What it never claims:**
- a general or full-data win (with full data TACS-X loses, −1.35pp on validation);
- an exact paper reproduction;
- that paper TACS (unanchored) wins.

Paper figures (CUB No-Context 82.1, TACS 85.2; full data, 100 epochs) may appear only labelled as a different setup, never compared directly.

**Disclosed history.** The official test has been used three times, each for a new protocol registered before training:
- `cub200_lowdata2_v1`: labeled pool, +2.66pp;
- `cub200_unlabeled_v1`: unlabeled pool;
- `cub200_selftrain_v1`: self-training, the headline.

Validation-only follow-ups (a second self-training round, 10 images per class) are not part of the claim.

## Operating Context

Presented live on a projector or laptop during a defense, and reopened later. The page must work without the server. Live upload is an optional extra, decided with the user on 2026-11-19, and is never required for the result.

Data sources:
- **Headline:** `outputs/cub200_selftrain_v1/`, including `test/test_result.json`, the predictions, the checkpoints, `pseudo_labels.json` per seed and arm, and `amendments.json`.
- **Teachers and the earlier steps:** `outputs/cub200_unlabeled_v1/`.
- **Data-size and full-data context:** the validation-only runs `outputs/cub200_anchored_dev_s42` and `cub200_lowdata{5,3,2}_dev_s42`.

## Capabilities and Constraints

- Page order, decided with the user on 2026-11-19:
  1. **Thiết lập & giới hạn.** Short. The other data sizes sit in a collapsed panel.
  2. **Kết quả.** One sentence plus a per-seed dumbbell chart (No-Context dot to TACS-X dot), mean row last.
  3. **Ba cải tiến.** One row per change, in three columns: Vấn đề (with a visual) → Cách làm (with a visual) → Hiệu quả (a before→after arrow on one shared scale).
     - Step 1 shows trained and untrained branches.
     - Step 2 shows a unit chart of the pool.
     - Step 3 shows a seeded-random species with its 2 labeled and 5 pseudo-labeled images, each marked right or wrong.
  4. **Từng bước.** The step-by-step inference walkthrough. When served by `scripts/serve_demo.py`, it also accepts an uploaded image: no true label, so no right/wrong marks and no reward; near-duplicate pool images are masked.
- Write in plain language. A section title must say what the reader learns, not name a technique.
- Walkthrough examples come from the official CUB test split. They are a seeded random sample per outcome group, never hand-picked: TACS-X fixes, TACS-X breaks, both right, both wrong. The size of each group is shown. The checkpoints are the self-trained seed-10 models.
- Protocol facts:
  - 224px, batch 16;
  - 2 labeled training images per class (400), selected by data seed 42;
  - context pool: the 5,400 training-split images, identity-masked;
  - warm-up of 5 epochs; teachers trained for 75 epochs; self-trained students for 21 epochs;
  - checkpoint selected on validation (594 images);
  - tau 0.1, lambda_policy 1.0, aux_query_weight 1.0.
- Primary readout: forced context (always use the selected context image). The abstain gate is reported, but it is not presented as a working component.
- Disclose all deviations; never claim exact reproduction.

## Brand Commitments

Name: TACS-X. No logo exists. Self-hosted fonts already in `demo/fonts/` (Archivo, Source Serif 4, Courier Prime with Vietnamese subsets).

## Evidence on Hand

- Three confirmatory runs, each with its registration, pre-test lock and saved predictions. The builder re-derives every headline accuracy from the saved predictions.
- Validation-only development runs (`outputs/cub200_k2_variants/summary.json`).
- CUB images under `data/CUB_200_2011/`, local only, used to bake example thumbnails.
- No paper figures are bundled. No user studies or external endorsements exist; do not invent any.

## Product Principles

1. Lead with the confirmed result and its exact scope; the narrow claim is the strength.
2. Make the story causal and linear: every number on the page belongs to one of the three changes or to the scope.
3. Show mistakes alongside successes: examples where TACS-X breaks a correct prediction stay visible.
4. Separate what the model computed from interpretation; label validation versus test numbers.
5. Works offline from a double-click.

## Accessibility & Inclusion

Readable from the back of a room: large type, high contrast, no color-only encoding of right and wrong. Vietnamese diacritics must render correctly.
