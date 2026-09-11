#!/usr/bin/env bash
set -euo pipefail

mkdir -p third_party

if [ ! -d third_party/dinov3 ]; then
  git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
else
  echo "third_party/dinov3 already exists; skipping clone"
fi

python -m pip install -U pip
python -m pip install -e .
python -m pip install -r requirements.txt

echo
echo "Bootstrap complete."
echo "Next:"
echo "  1) Follow official DINOv3 instructions to obtain ViT-S/16 weights."
echo "  2) Set configs/cifar100.yaml -> backbone.weights."
echo "  3) Run: pytest -q"
