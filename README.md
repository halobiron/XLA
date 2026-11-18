# TACS-X ? CUB-200-2011

Compact PyTorch implementation of Task-Aligned Context Selection with the official
DINOv3 ViT-S/16 backbone. The active dataset is **CUB-200-2011** (200 bird classes).
Old CIFAR training scripts, datasets, experiment checkpoints and packaged runs have
been retired. The original DINOv3 pretrained weights remain in `checkpoints/`.

## Download the dataset

Official record: https://data.caltech.edu/records/65de6-vp158

Download **CUB_200_2011.tgz** (about 1.2 GB):
https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1

Official MD5: `97eceeb196236b17998738112f37df78`.

Save the archive in `data/`, then run in PowerShell from this repository:

```powershell
Get-FileHash .\data\CUB_200_2011.tgz -Algorithm MD5
tar -xzf .\data\CUB_200_2011.tgz -C .\data
```

Expected layout:

```text
data/CUB_200_2011/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
  classes.txt
```

The archive includes other annotations. Image classification needs the files above;
no separate segmentation download is needed. Use the official train/test split.
Validation is stratified from official training images, and the fixed candidate
pool uses only the remaining training images. Query IDs are excluded from retrieval.

## Code layout

- `configs/cub200.yaml`: local starter settings and explicit deviations.
- `src/tacsx/data/`: official CUB metadata reader and generic context wrapper.
- `src/tacsx/models/`: DINO adapter, selector, dual-image classifier, extensions.
- `src/tacsx/losses/policy.py`: detached reward and REINFORCE.
- `src/tacsx/engine.py`: common training/evaluation for all arms.
- `src/tacsx/cli.py`: configurable runner, 200 output classes for CUB.
- `tests/`: CPU tests with dummy encoders and synthetic image metadata.
- `paper/`, `docs/`: methodology and implementation notes.

Use `.venv-gpu/Scripts/python.exe` for real model diagnostics and training on this
workstation. The original `.venv` is CPU-only. No training starts automatically.
The package is installed in `.venv-gpu` in editable mode. To set it up again
without changing the existing Torch/CUDA dependencies:

```powershell
.venv-gpu/Scripts/python.exe -m pip install -e . --no-deps --no-build-isolation
```

## Commands

Inspect the CLI without loading weights or images:

```powershell
.venv-gpu/Scripts/python.exe -m tacsx.cli --help
```

After the dataset is extracted, the same runner supports `no_context`,
`random_context`, `dino_similarity`, `gumbel_only`, `policy_only`, `full_tacs`,
`topk_tacs`, and `adaptive_topk_tacs` via `--mode`.

```powershell
.venv-gpu/Scripts/python.exe -m tacsx.cli --config configs/cub200.yaml --mode no_context --name cub200_no_context
.venv-gpu/Scripts/python.exe -m tacsx.cli --config configs/cub200.yaml --mode full_tacs --name cub200_full_tacs
```

These are training commands, not an already validated winning recipe. The starter
config uses 100 epochs; budget changes must be chosen before running both arms.
By default runs evaluate **validation only**. Official test evaluation requires
explicit `--evaluate-test`, after checkpoint selection on validation.

Runs save config, history, metrics, checkpoint and selected-context metadata under
`outputs/<name>/`. Candidate labels are diagnostic metadata, never selector inputs.
Aggregate runs with `python -m tacsx.summarize`; render saved pairs with
`python scripts/render_retrieval_examples.py outputs/<name>`.

Run CPU tests without model downloads:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

## Fidelity

Classification concatenates query/context patch embeddings before transformer
blocks. Full TACS combines straight-through Gumbel-Softmax and detached CE-difference
REINFORCE. The CUB config shares the trainable selector/task backbone.

This is a limited-compute implementation, not an exact reproduction. The starter
uses batch size 4 and 8 uniformly sampled candidates, rather than claiming full-pool
retrieval. Multi-image RoPE layout is a documented project choice.
`AGENTS.md` specifies `lambda_policy=1.0`; paper section 4.2 specifies 0.5. The config
keeps the repository value and logs the discrepancy. No CUB accuracy has been measured.
