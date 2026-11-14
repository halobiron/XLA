# TACS-X Coding Agent Pack

## Active matched TACS experiment

The current goal requires a defensible TACS advantage over independently
trained No-Context. The residual-head result below does not establish that.
The active run is `outputs/tacs_shortlist_fullft_clean_v3/`.

This explicit TACS-X variant retains query/context patch concatenation,
hard straight-through Gumbel selection (`tau=.1`), the detached CE-difference
reward, and REINFORCE (`lambda_policy=1`). Both task networks continue the
same v3 No-Context checkpoint for 10 epochs with all 12 task blocks trainable,
batch 32, AdamW, cosine scheduling, and identical query data/order/augmentation.
The selector uses frozen official DINOv3 and a trainable identity-initialized
linear projection with cosine logits scaled by `1/.07`. Its candidates are the
16 nearest official-DINO proposals from the fixed 9,000-image training pool.
These changes, the common query warm start, and 224px resolution are logged
deviations from the paper; no exact reproduction is claimed.

Pixel identities are checked in addition to row IDs. Four validation images
identical to training images are excluded (4,996 remain), and all pixel-identical
query/candidate aliases are masked. Test evaluation excludes images identical
to the original CIFAR training file and repeated heldout images. Candidate
labels are absent from scoring and optimization inputs.

The prospective evaluation covers clean accuracy, mean accuracy over eight
fixed corruptions, and clean NLL; all are reported with paired query bootstrap
intervals and Bonferroni correction over three comparisons. Corruption results
are an additional project benchmark, not CIFAR-100-C or a paper benchmark.
`scripts/finish_shortlist_benchmarks.ps1` monitors the verified live training
process, evaluates validation when both budgets finish, and permits a locked
test evaluation only after a positive predeclared validation interval.
`pipeline_status.json` records its stage; it is not a substitute for checking
the live process when recovering an interrupted job.

Resuming the same experiment (the immutable execution snapshot is preferred
while editing source files):

```powershell
$env:PYTHONPATH = "scripts"
.venv-gpu\Scripts\python.exe outputs/tacs_shortlist_fullft_clean_v3/execution/train_tacs_shortlist_cpu_state_snapshot.py --stage train --arms no_context full_tacs --cached-pixels --pixel-cache outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt
```

The scalar AdamW step counters are loaded on CPU; mapping the entire optimizer
checkpoint to CUDA adds a synchronization per parameter. Optimizer and model
state are saved before validation and again after validation. The imported
No-Context first epoch is unchanged optimization; its prediction filtering and
source hash are recorded in `no_context/prefix_provenance.json`. Earlier v2
TACS checkpoints are exploratory artifacts and are not reused in the clean run.
Real-model gradient audits verify both selector learning paths and detached
reward, plus exact checked-batch forward/backward equivalence of the pixel cache.

## Local GPU environment

This Windows machine has an NVIDIA RTX 5060 Ti with 16 GB VRAM. The original
`.venv` contains a CPU-only PyTorch build; a false CUDA check in that environment
does not mean the hardware has no GPU. Use `.venv-gpu\Scripts\python.exe` for
GPU diagnostics/training. It contains PyTorch 2.14.0+cu130 and torchvision
0.29.0+cu130; CUDA forward/backward has been verified on this GPU.

```powershell
.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv-gpu\Scripts\python.exe scripts/selector_credit_probe.py --device cuda
```

The selector probe caches fixed features/losses, learns cloned MLP heads only
on training query IDs, and scores held-out validation images after the fixed
training schedule. It is a diagnostic, not Full TACS or a paper reproduction.
Original zipped checkpoints are never overwritten. Results and resumable
feature caches are in `outputs/diagnostics/selector_credit_gpu/`.

## Verified local GPU result (2026-11-14)

Fixed-checkpoint evaluation on all 10,000 CIFAR-100 test images:

| Model | Test accuracy |
| --- | ---: |
| Original v3 No-Context | 86.17% |
| Frozen native encoder + query-only residual MLP | 88.01% |
| Same-size residual MLP with retrieved image context | 87.98% |

The query-only extension improves the original baseline by 1.84 percentage
points (paired bootstrap 95% CI: +1.42 to +2.27). It adds 786,180 trainable
parameters and 20 epochs of head training on cached features; the original
backbone and linear head stay frozen. This is an architecture/training
extension, not a matched-budget improvement or an original TACS reproduction.
Context does not demonstrate an accuracy benefit over the same-size query-only
control: -0.03 points (95% CI: -0.39 to +0.33). Results cover one training seed.

The existing `outputs.zip` supplies the unchanged v3 No-Context checkpoint.
Training and pool features use only the train file; no candidate labels are
inputs to retrieval or fusion. Epoch selection uses the first 2,500 validation
images. Several context families were explored and the second validation half
was reused across families. Final checkpoint hashes and the protocol were
locked before opening test data; no configuration was selected using test.

```powershell
# Native features: validation/pool, then 45k training queries (resumable).
.venv-gpu\Scripts\python.exe scripts/search_native_context.py --stage cache
.venv-gpu\Scripts\python.exe scripts/search_native_context.py --stage train_cache
# Both matched heads; completed runs are skipped, interrupted runs resume.
.venv-gpu\Scripts\python.exe scripts/train_native_residual.py
.venv-gpu\Scripts\python.exe scripts/train_native_residual.py --stage confirm
# Fixed final evaluation; an existing result is returned without rescoring.
.venv-gpu\Scripts\python.exe scripts/evaluate_native_residual_test.py
```

Winning head: `outputs/native_residual_v1/query_only_control/checkpoint_best.pt`
(epoch 9, validation-selected). It is a residual head checkpoint and requires
the native encoder from the original No-Context checkpoint. The context head
uses epoch 20. Final metrics, predictions, context pairs, locked hashes, and
comparison CSV are under `outputs/native_residual_final_test/`. Final feature
extraction/evaluation took 30.5 seconds on the RTX 5060 Ti; this excludes prior
feature caching and head training. The full dummy-encoder suite passes 42 tests.

### Original TACS robustness probe

`scripts/probe_tacs_robustness.py` compares the unchanged original Full TACS
and independently trained No-Context checkpoints. Its locked project benchmark
applies two severities each of noise, blur, contrast, and brightness to native
32px query images before the normal preprocessing; context remains clean.
This is not CIFAR-100-C and is not a robustness benchmark from the paper.
The first 512 validation images yielded mean corrupted-image accuracy 64.43%
for Full TACS versus 69.65% for No-Context (paired query-cluster bootstrap
difference -5.22 points, 95% CI -6.59 to -3.91). All eight conditions are
retained. This probe provides no evidence of a robustness advantage; no
corrupted test images have been evaluated. Protocol and predictions are saved
in `outputs/tacs_robustness_v1/` and the probe took 35 seconds on the local GPU.

Starter pack for reproducing and extending **Task-Aligned Context Selection (TACS)** from:

> Jingyu Guo et al., *Learning What Helps: Task-Aligned Context Selection for Vision Tasks*, CVPR 2026.

The saved 23 runs use the legacy trainer. They do not validate the corrected
partial fine-tuning, token positions or argmax evaluation. See
[`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) for the validation-only
diagnosis and measured limitations. New runs record `trainer_version: 3`.

For a matched, compute-limited rerun, start with:

```powershell
$env:PYTHONPATH = "src"
.venv-gpu\Scripts\python.exe -m tacsx.cli --device cuda --config configs/cifar100_unf4.yaml --mode no_context --name v3_unf4_no_context_s42
.venv-gpu\Scripts\python.exe -m tacsx.cli --device cuda --config configs/cifar100_unf4.yaml --mode random_context --name v3_unf4_random_s42
.venv-gpu\Scripts\python.exe -m tacsx.cli --device cuda --config configs/cifar100_unf4.yaml --mode dino_similarity --name v3_unf4_dino_s42
.venv-gpu\Scripts\python.exe -m tacsx.cli --device cuda --config configs/cifar100_unf4.yaml --mode full_tacs --name v3_unf4_full_s42
```

Use a GPU for training. These are 5-epoch partial-fine-tuning diagnostics, not
paper-equivalent runs or a promise to exceed the frozen No-Context baseline.
Old checkpoints without `backbone.pair_rope_layout` keep the legacy layout.

## Goal

Build a compact, defensible reproduction for image classification, then add one small extension:

- DINOv3 ViT-S/16 backbone.
- Fixed candidate pool.
- Learned selector.
- Straight-through Gumbel-Softmax path.
- Policy-gradient path with downstream task reward.
- Dual-image downstream ViT.
- Baselines: No Context, Random, Frozen DINO similarity.
- Ablations: Gumbel-only, Policy-only, Full TACS.
- Extension: Top-K / optional adaptive-K context.
- Interpretability: retrieval pairs, cross-class selection, attention rollout if feasible.

## Recommended first dataset

Use **CIFAR-100** first for fast iteration. Add Oxford-IIIT Pets or CUB-200 only after the core system is stable.

## Start here

Read in order:

1. `AGENTS.md`
2. `PROJECT_SPEC.md`
3. `docs/PAPER_NOTES.md`
4. `docs/IMPLEMENTATION_PLAN.md`
5. `docs/EXPERIMENT_MATRIX.md`
6. `prompts/AGENT_BOOTSTRAP.md`

Then:

```bash
bash scripts/bootstrap.sh
pytest -q
```

## Kaggle: use mounted data, do not download it at runtime

The `169M` progress bar is the CIFAR-100 archive. It is deliberately excluded
from Git because it exceeds GitHub's normal single-file limit. Upload the
existing `data/cifar-100-python/` directory as a private Kaggle Dataset, attach
it to the notebook, and set its mount directory with `--data-root`. The default
configuration now refuses a runtime download.

```powershell
python -m tacsx.cli --config configs/cifar100.yaml --data-root /kaggle/input/<your-cifar100-dataset> --mode full_tacs --name cifar100_kaggle
```

The directory supplied to `--data-root` must directly contain
`cifar-100-python/`. Commit the source files, configuration, and the local
DINOv3 checkpoint already tracked in `checkpoints/`; do not commit `data/`,
`src/tacsx.egg-info/`, or generated outputs. Attach the official DINOv3 source
and weights as Kaggle Datasets too if they are not already present in the
repository checkout.

## Run experiments

Obtain DINOv3 ViT-S/16 weights through Meta's official access flow, set
`backbone.weights` in `configs/cifar100.yaml`, then run one shared CLI for all
baselines and ablations:

```powershell
python -m tacsx.cli --config configs/cifar100.yaml --mode no_context --name cifar100_e0
python -m tacsx.cli --config configs/cifar100.yaml --mode dino_similarity --name cifar100_e2
python -m tacsx.cli --config configs/cifar100.yaml --mode gumbel_only --name cifar100_e3
python -m tacsx.cli --config configs/cifar100.yaml --mode policy_only --name cifar100_e4
python -m tacsx.cli --config configs/cifar100.yaml --mode full_tacs --name cifar100_e5
```

`random_context`, `topk_tacs`, and `adaptive_topk_tacs` are supported by the
same command. Outputs contain config, metrics, history, checkpoint, and
selected-pair metadata. Use `--allow-random-backbone` only for a smoke test;
it is explicitly marked as a fidelity deviation.

For E6/E7, set `topk.k: 2` or `topk.k: 4` in the config before using
`--mode topk_tacs`.

After runs finish, aggregate them with `python -m tacsx.summarize --outputs outputs`.

The source tree already defines reusable interfaces and tests. The Coding Agent should complete DINOv3 integration and the training/evaluation pipeline rather than replacing the project with a different framework.

## Demo trực quan (offline)

```powershell
.venv-gpu/Scripts/python.exe scripts/build_demo.py  # đọc 8 run v3 từ outputs.zip
start demo/index.html             # mở bằng double-click, không cần server
```

Để dùng thêm tab **Thử nhận diện** (tải ảnh lên, chạy 4 checkpoint v3 trên GPU):

```powershell
powershell -NoProfile -File scripts/start_demo.ps1   # http://127.0.0.1:8765
```

Trang bằng tiếng Việt: kết luận và cơ chế, so sánh các mode, truy hồi context
trên 12 query test, ablation (K=2/4 và đường học selector) và giao thức. Bảng số
liệu, đường validation và cặp truy hồi đều đọc trực tiếp từ `outputs.zip`;
không trộn các run cũ trong `outputs/`. Tab nhận diện dùng cùng archive,
với encoder DINO pretrained đóng băng riêng cho truy hồi DINO-similarity.

Demo trong Tailscale: **https://rgb.tail27fd0e.ts.net:10000/**. Tuyến Serve
hiện có proxy về `127.0.0.1:8765`; các thiết bị cần kết nối cùng tailnet và
được chính sách mạng cho phép truy cập. Tác vụ Windows `TACS-X Demo` khởi
động server khi đăng nhập. Sau khi cập nhật code/checkpoint, chạy
`Stop-ScheduledTask -TaskName 'TACS-X Demo'` rồi
`Start-ScheduledTask -TaskName 'TACS-X Demo'` để nạp lại mô hình.
Kiểm tra sức khỏe tại `/api/status`. Biên bản kiểm tra triển khai nằm ở
`outputs/deployment/v3-deployment.json`.
