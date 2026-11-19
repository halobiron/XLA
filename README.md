# TACS-X ? CUB-200-2011

Compact PyTorch implementation of Task-Aligned Context Selection with the official
DINOv3 ViT-S/16 backbone. The active dataset is **CUB-200-2011** (200 bird classes).
Old CIFAR training scripts, datasets, experiment checkpoints and packaged runs have
been retired. The original DINOv3 pretrained weights remain in `checkpoints/`.

## Result (narrow, pre-registered)

With **2 labeled training images per class**, anchored TACS-X beat a matched
No-Context model on the official CUB test split.

| Official test, 5,794 images | Accuracy (mean of 4 seeds) |
|---|---:|
| No-Context | 53.33% |
| Anchored TACS-X, always uses the selected context (primary readout) | **55.99%** |

- **Difference:** +2.66pp. 95% CI [+2.16, +3.17] from a query bootstrap; [+1.80, +3.52] from a t-interval over seeds. Every seed is positive.
- **Run:** fresh seeds 0-3. The test split was evaluated once, against a hash lock written before any test image was read.
- **Where the gain comes from:** +1.69pp from anchored training (TACS-X's own query-only path) and +0.97pp from context (CI [+0.51, +1.43]).
- **How the regime was chosen:** on validation, after five explored regimes. Full data: -1.35pp. Corrupted query: -4.46pp. K=5: -0.17pp. K=3: -0.34pp. K=2 dev: +2.19pp.

This is not a full-data win and not an exact paper reproduction. "Anchored" is a TACS-X extension: a trained query-only reward baseline plus an abstain gate. The gate adds nothing; it falls back to query-only on every disagreement.

- Artifacts: `outputs/cub200_lowdata2_v1/`, including `test/test_result.json`, `test/test_lock.json`, `verification.json` and `comparison.csv`.
- Independent re-check: `python scripts/verify_cub_lowdata2.py` (12/12 checks pass).
- Training/evaluation: `scripts/run_cub_anchored.py`.
- Kaggle bundle: `scripts/package_kaggle_cub_anchored.py` and `scripts/kaggle_cub_anchored_cell.py`.

## Offline demo

Open `demo/index.html` by double-click; no server is needed. The page has four sections:

1. Setup and limits.
2. The official-test result.
3. The three changes that produced it: anchored training, the unlabeled context pool, and class-balanced self-training.
4. A seven-step No-Context vs TACS-X walkthrough on official-test images, using a seeded random sample per outcome group.

Rebuild its data with:

```powershell
$env:PYTHONPATH="src"; .venv-gpu/Scripts/python.exe scripts/build_cub_demo.py
```

The builder re-runs the seed-10 self-trained checkpoints (`outputs/cub200_selftrain_v1`) and refuses to write if any rebuilt prediction differs from the saved predictions.

### Try your own images

Double-click `Mo-demo.bat` in the project root. Equivalent from a terminal:

```powershell
.venv-gpu/Scripts/python.exe scripts/serve_demo.py --open
```

After about a minute it opens `http://127.0.0.1:8765/demo/index.html`, where the walkthrough's "Ảnh của bạn" row gets an upload button. It runs both seed-10 models on your image against the 5,400-image training pool. Close the console window to stop the server.

Uploads have no true label, so they get no right/wrong marks and no reward. The server needs a CUDA GPU, listens on localhost only, writes nothing to disk, and never reads the official test split.

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
