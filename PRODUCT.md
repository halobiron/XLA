# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS opened by double-click (no server, no network). A Python builder bakes data from `outputs/` and the saved checkpoints into a local JS data file next to the page.

## Users

Students presenting a university computer-vision course project (XLA) to an instructor and a grading committee. Secondary: teammates inspecting runs. The interface is in Vietnamese; ML technical terms stay in English (reward, selector, context, query, No-Context, seed, checkpoint, validation, test, ...).

## Product Purpose

TACS-X is a compact PyTorch implementation of Task-Aligned Context Selection (Guo et al., CVPR 2026) on CUB-200-2011 with an official DINOv3 ViT-S/16 backbone. It includes an explicit extension, "anchored TACS-X", which trains the query-only reward baseline and adds an abstain gate. The demo shows the committee what was measured and how one prediction is made, step by step, for No-Context and TACS on the same image.

## Positioning

An honest, pre-registered result with a deliberately narrow claim. On CUB with 2 labeled images per class, anchored TACS-X beats a matched No-Context model on the official test split by +2.66 percentage points: 4 fresh seeds, the test evaluated once, query-bootstrap 95% CI [+2.16, +3.17]. Every seed is positive. About +1.69pp comes from the anchored training (its own query-only path) and +0.97pp (CI [+0.51, +1.43]) from context. The regime was chosen on validation after five explored regimes. The demo always shows the others:
- full data: −1.35pp (validation);
- corrupted query: −4.46pp (validation);
- K=5: −0.17pp (validation);
- K=3: −0.34pp (validation).

It never claims a general or full-data win, an exact paper reproduction, or that paper TACS (unanchored) wins. Paper figures (CUB No-Context 82.1, TACS 85.2; full data, 100 epochs) may appear only labelled as a different setup, never compared directly.

## Operating Context

Presented live on a projector or laptop during a defense, and reopened later. No live model inference during the defense. The data comes from `outputs/cub200_lowdata2_v1/`:
- `test/test_result.json`
- `verification.json`
- `comparison.csv`
- the predictions and checkpoints
- the explored-regime runs `outputs/cub200_anchored_dev_s42`, `cub200_lowdata{5,3,2}_dev_s42` and the corruption probe.

## Capabilities and Constraints

- Two parts, decided with the user on 2026-11-18:
  - a results section (the confirmed test result, per-seed numbers, mechanism decomposition, and every explored regime);
  - a step-by-step inference walkthrough on real images: query, the selector's top candidates, the chosen context, and No-Context vs TACS predictions.
- Examples come from the official CUB test split, which has already been evaluated once. They are a seeded random sample per outcome group, never hand-picked: TACS fixes, TACS breaks, both right, both wrong. The size of each group is shown.
- Protocol facts:
  - 224px, batch 16;
  - 2 training images per class (400) selected by data seed 42; the context pool is that same labeled subset, with identical pixels masked;
  - query-only warm-up of 5 epochs, then 75 epochs per arm; checkpoint selected on validation (594 images);
  - tau 0.1, lambda_policy 1.0, aux_query_weight 1.0.
- Primary readout: "luôn dùng context" (forced context). The abstain gate always chose query-only. Report it, and do not present it as a working component.
- Disclose all deviations; never claim exact reproduction.

## Brand Commitments

Name: TACS-X. No logo exists. Self-hosted fonts already in `demo/fonts/` (Archivo, Source Serif 4, Courier Prime with Vietnamese subsets).

## Evidence on Hand

- Confirmatory run with its pre-test lock, predictions and independent verification (12/12 checks pass).
- Validation-only development runs and pre-registered criteria files.
- CUB images under `data/CUB_200_2011/` (local only, used to bake example thumbnails).
- No paper figures are bundled. No user studies or external endorsements exist; do not invent any.

## Product Principles

1. Lead with the confirmed result and its exact scope; the narrow claim is the strength.
2. Never hide the negative regimes or the selection history; they sit on the same page as the win.
3. Show mistakes alongside successes: examples where TACS breaks a correct prediction stay visible.
4. Separate what the model computed from interpretation; label exploratory versus confirmatory numbers.
5. Works offline from a double-click.

## Accessibility & Inclusion

Readable from the back of a room: large type, high contrast, no color-only encoding of right and wrong. Vietnamese diacritics must render correctly.
