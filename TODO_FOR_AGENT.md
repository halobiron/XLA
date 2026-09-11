# Coding Agent TODO

## P0
- [x] Bootstrap official DINOv3.
- [x] Complete `DINOv3Adapter`.
- [x] Implement `DualInputViT`.
- [x] Implement CIFAR-100 + fixed candidate-pool wrapper.
- [x] Implement No-Context train/eval.
- [x] Implement DINO similarity baseline.
- [x] Implement config/CLI.
- [x] Implement checkpoints + metrics logging.

## P1
- [x] Gumbel-only mode.
- [x] Policy-only mode.
- [x] Full TACS mode.
- [x] Verify reward detach and gradient flow.
- [x] Selected-pair CSV.
- [x] Ablation runner.

## P2
- [x] Cross-class selection analysis.
- [x] Retrieval-pair visualization.
- [x] Top-K extension.
- [x] Adaptive-K extension.
- [x] Gradio demo.

## Before finalizing
- [x] All tests pass.
- [ ] Smoke train every required mode.
- [x] Update implementation decisions.
- [x] Add exact commands to README.
- [ ] Generate result summary from real runs (requires official DINOv3 weights).
- [ ] Never fabricate paper-matching numbers.
