# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS opened by double-click (no server). `scripts/build_stepwise_demo.py` bakes clean_v3 step-by-step traces into `demo/steps_data.js`; `scripts/build_demo.py` bakes the older v3 explorer (`demo/results_v3.html`). Optional live upload via `python -m tacsx.serve` (same `tacsx.stepwise` engine).

## Users

Students presenting a university computer-vision course project (XLA) to an instructor and a grading committee. Secondary: teammates inspecting runs. Interface language: Vietnamese; technical terms stay English.

## Product Purpose

TACS-X is a compact reproduction of Task-Aligned Context Selection (Guo et al., CVPR 2026) on CIFAR-100 with a DINOv3 ViT-S/16 backbone, plus a Top-K / adaptive Top-K extension. The main page (`demo/index.html`) walks a committee through each inference step of No-Context vs Full TACS (clean_v3) on the same validation image, one step per click; the older 24-run v3 explorer remains at `demo/results_v3.html`.

## Positioning

An honest mini reproduction. At the user's request, the main demo opens with project Full TACS 90.01% versus paper No-Context 87.8% (+2.21 percentage points), explicitly a comparison of different training setups. The project's own No-Context result (90.22%) stays beside that comparison so the gap is not attributed to TACS alone. A compact table shows GPU, resolution/batch, epochs/warm start, augmentation and seed differences. DINOv3 ViT-S/16, AdamW and cosine scheduling are explicitly shared with the paper; fixing old project code is not a paper difference. The page then shows inference on individual images. It does not claim exact reproduction or an accuracy advantage over the matched project baseline.

## Operating Context

Presented live on a projector or laptop, and reopened later. CPU-only machine; no live model inference during the defense. Data comes from `outputs/` (summary.csv, metrics.json, history.csv, selected_pairs.csv, retrieval_examples/*.png).

## Capabilities and Constraints

- Step-by-step walkthrough is the main page (decided with the user, 2026-11-14): click/arrow keys reveal one step at a time; same visual world as the v3 explorer.
- Examples: seeded random sample per outcome group, never hand-picked; all four outcome groups remain available. The demo builder reads validation only. A separate locked evaluation has evaluated all 10,000 standard test images.
- Main demo protocol: 224px, batch 32, seed 42; 5-epoch query-only warm start with 4 task blocks unfrozen, followed by 10 epochs with all 12 task blocks trainable; nearest-16 proposals from 9,000 train images, frozen selector encoder plus learned projector; tau=0.1, lambda_policy=1.0.
- Group C runs (c16, unf4, seed 1) are not directly comparable with group A; two seeds only, no statistical claim.
- Must disclose deviations; never claim exact reproduction.

## Brand Commitments

Name: TACS-X. No logo exists.

## Evidence on Hand

24 runs in `outputs/`, `docs/EXPERIMENT_STATUS.md`, retrieval example PNGs for most context runs. No paper figures bundled.

## Product Principles

1. Show the user's requested paper No-Context versus project Full TACS comparison with a short table of actual setup differences.
2. Retain correct and incorrect examples for both models; do not imply that TACS improves accuracy.
3. Comparable runs are grouped; incomparable ones are labelled as such.
4. Works offline from a double-click.

## Accessibility & Inclusion

Readable from the back of a room: large type, high contrast, no color-only encoding. Vietnamese diacritics must render correctly.
