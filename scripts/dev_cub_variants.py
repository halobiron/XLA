"""Validation-only development of anchored TACS-X variants at K=2 (seed 42).

Reuses the seed-42 warm-up of outputs/cub200_lowdata2_dev_s42 and trains ONE
anchored_tacs arm per call with the confirmatory recipe (75 epochs, batch 16),
changing only the flags below. Reference: that run's No-Context = 55.39 %,
anchored TACS-X = 57.58 % on validation. Official test pixels are blocked.

  --pool labeled     the 400 labeled training images (confirmatory setting)
  --pool unlabeled   every training-split image, labels never read (identity-masked)
  --aux-weight       beta on CE(query-only)
  --no-gate          disable the abstain gate (it never changes the primary readout)
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import torch
from PIL import Image
import run_cub_anchored as R

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--pool", choices=("labeled", "unlabeled"), default="labeled")
p.add_argument("--aux-weight", type=float, default=1.0)
p.add_argument("--no-gate", action="store_true")
p.add_argument("--epochs", type=int, default=75)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--warm", default="outputs/cub200_lowdata2_dev_s42")
a = p.parse_args()
root = HERE.parents[1]
data_root = root / "data/CUB_200_2011"
if (data_root / "CUB_200_2011").is_dir(): data_root = data_root / "CUB_200_2011"
weights = root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"; repo = root / "third_party/dinov3"
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
out = Path("outputs/cub200_k2_variants") / a.name
out.mkdir(parents=True, exist_ok=True)

orig, audit = R.protect_test_images(data_root)
train, val, _, pixels, pool_ids, ident, split = R.prepare(data_root, 16, 0, train_per_class=2)
if a.pool == "unlabeled":
    full, *_ = R.make_cub200_loaders(data_root, 224, .2, 1, 16, 0, R.DATA_SEED, include_context=True)
    clean = full.dataset.candidate_base
    cache = Path("outputs/cub200_k2_variants/identity_cache.pt")
    ident = ident.cpu().clone()
    if cache.exists():
        ident = torch.load(cache)
    else:
        groups = {}
        for qid in list(full.dataset.query_ids) + list(val.dataset.query_ids):
            with Image.open(clean.root / "images" / clean.paths[qid]) as im:
                im = im.convert("RGB"); key = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            ident[qid] = groups.setdefault(key, qid)
        torch.save(ident, cache)
    val_groups = {int(ident[v]) for v in val.dataset.query_ids}
    ids = [i for i in full.dataset.query_ids if int(ident[i]) not in val_groups]
    assert not set(ids) & set(val.dataset.query_ids)
    pool_ids = torch.tensor(ids).cuda()
    pixels = torch.stack([clean[int(i)][0] for i in ids]).half().cuda()
    ident = ident.cuda()
    # train queries must map to the same identity groups as before (self-retrieval masked)
    assert all(int(ident[q]) >= 0 for q in train.dataset.query_ids)
split["pool_count"] = len(pool_ids)

args = argparse.Namespace(seeds=[a.seed], arms=["no_context", "anchored_tacs"], warmup_epochs=5, epochs=a.epochs,
                          batch_size=16, train_per_class=2, max_steps=0, eval_limit=0, output_path=out)
R.ARM_FLAGS["anchored_tacs"] = {"aux_query_weight": a.aux_weight, "abstain_gate": not a.no_gate}
R.STATUS_NAME = "status.json"
R.dump(out / "config.json", {**vars(a), "split": {k: v for k, v in split.items() if not isinstance(v, list)},
                             "scope": "validation only"})
train.dataset.seed = a.seed
warm = R.build_warm_model(weights, repo, a.seed)
warm.load_state_dict(torch.load(Path(a.warm) / f"seed_{a.seed}/warmup/checkpoint_best.pt", map_location="cpu",
                                weights_only=False)["model"])
model = R.make_arm("anchored_tacs", warm, a.seed)
t0 = time.perf_counter()
best = R.run_phase("anchored_tacs", model, a.epochs, train, val, pixels, pool_ids, ident, a.seed, 5, args, out)
Image.open = orig
R.dump(out / "image_access_audit.json", audit)
print("RESULT", a.name, json.dumps(best), f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
