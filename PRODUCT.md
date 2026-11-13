# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS, one self-contained page opened by double-click (no server). A small Python script bakes `outputs/` into a JSON data file the page reads. Confirmed with the user.

## Users

Students presenting a university computer-vision course project (XLA) to an instructor and a grading committee. Secondary: teammates inspecting runs. Interface language: Vietnamese; technical terms stay English.

## Product Purpose

TACS-X is a compact reproduction of Task-Aligned Context Selection (Guo et al., CVPR 2026) on CIFAR-100 with a DINOv3 ViT-S/16 backbone, plus a Top-K / adaptive Top-K extension. The demo lets a committee understand the mechanism and see the 24 completed runs, honestly, in a short defense.

## Positioning

An honest mini reproduction: the demo shows that learned top-1 selectors did not beat No-Context or DINO-similarity here, and that the Top-K extension recovers about 20+ points over full TACS top-1. It does not claim exact reproduction of the paper.

## Operating Context

Presented live on a projector or laptop, and reopened later. CPU-only machine; no live model inference during the defense. Data comes from `outputs/` (summary.csv, metrics.json, history.csv, selected_pairs.csv, retrieval_examples/*.png).

## Capabilities and Constraints

- Static results explorer only; no inference tab (decided with the user).
- Protocol: 45k/5k/10k split, frozen backbone, 224px, 5 epochs, batch 32, C=4 candidates, seed 42 unless stated; tau=0.1, lambda_policy=1.0.
- Group C runs (c16, unf4, seed 1) are not directly comparable with group A; two seeds only, no statistical claim.
- Must disclose deviations; never claim exact reproduction.

## Brand Commitments

Name: TACS-X. No logo exists.

## Evidence on Hand

24 runs in `outputs/`, `docs/EXPERIMENT_STATUS.md`, retrieval example PNGs for most context runs. No paper figures bundled.

## Product Principles

1. Honest numbers first: negative results are shown as plainly as positive ones.
2. Show the mechanism (query, candidates, selected context) before the metrics.
3. Comparable runs are grouped; incomparable ones are labelled as such.
4. Works offline from a double-click.

## Accessibility & Inclusion

Readable from the back of a room: large type, high contrast, no color-only encoding. Vietnamese diacritics must render correctly.
