"""Validation-only inference probe on the locked K=2 checkpoints (no retraining, no test access).

Variants per seed (anchored_tacs checkpoint):
  top1_labeled   : the deployed readout (pool = 400 labeled images)
  topK_labeled   : mean softmax over the K best-scored contexts
  top1_unlabeled : pool = every training-split image (labels unused), identity-masked
  topK_unlabeled
"""
import argparse, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import torch, torch.nn.functional as F
from PIL import Image
import run_cub_anchored as R

p = argparse.ArgumentParser()
p.add_argument("--run", default="outputs/cub200_lowdata2_v1")
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
p.add_argument("--out", default="outputs/cub200_lowdata2_v1/probe_inference_validation.json")
a = p.parse_args()
root = HERE.parents[1]
data_root = root / "data/CUB_200_2011"
if (data_root / "CUB_200_2011").is_dir(): data_root = data_root / "CUB_200_2011"
weights = root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"; repo = root / "third_party/dinov3"
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

orig, _ = R.protect_test_images(data_root)
train, val, _, pix_lab, pool_lab, ident, split = R.prepare(data_root, 16, 0, train_per_class=2)
# full training split (labels never read) for the unlabeled pool
full_train, *_ = R.make_cub200_loaders(data_root, 224, .2, 1, 16, 0, R.DATA_SEED, include_context=True)
clean = full_train.dataset.candidate_base
all_train = list(full_train.dataset.query_ids)
ident = ident.clone().cpu()
cache = Path("outputs/cub200_k2_variants/identity_cache.pt"); cache.parent.mkdir(parents=True, exist_ok=True)
if cache.exists():
    ident = torch.load(cache)
else:
    groups = {}
    for qid in all_train + list(val.dataset.query_ids):
        with Image.open(clean.root / "images" / clean.paths[qid]) as im:
            im = im.convert("RGB"); key = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        ident[qid] = groups.setdefault(key, qid)
    torch.save(ident, cache)
Image.open = orig
val_ids = set(val.dataset.query_ids)
unl_ids = [i for i in all_train if int(ident[i]) not in {int(ident[v]) for v in val_ids}]
assert not set(unl_ids) & val_ids
pool_unl = torch.tensor(unl_ids).cuda()
pix_unl = torch.stack([clean[int(i)][0] for i in unl_ids]).half().cuda()
ident = ident.cuda()
print(f"val={len(val_ids)} labeled_pool={len(pool_lab)} unlabeled_pool={len(unl_ids)}", flush=True)

@torch.no_grad()
def run(model, pixels, pool_ids):
    model.eval(); feats = R.candidate_features(model, pixels)
    probs = {k: [] for k in a.ks}; labels = []
    for b in val:
        q, y = b["query"].cuda(), b["label"]
        with torch.autocast("cuda", dtype=torch.float16):
            s = R.utility(model, q, feats, b["query_id"], pool_ids, ident)
            top = s.topk(max(a.ks), -1).indices
            per = torch.stack([model.task_model(q, pixels[top[:, j]]).float().softmax(-1) for j in range(max(a.ks))], 1)
        for k in a.ks: probs[k].append(per[:, :k].mean(1).cpu())
        labels.append(y)
    y = torch.cat(labels)
    return {k: float(torch.cat(v).argmax(-1).eq(y).float().mean()) * 100 for k, v in probs.items()}

res = {}
for seed in a.seeds:
    warm = R.build_warm_model(weights, repo, seed)
    m = R.make_arm("anchored_tacs", warm, seed)
    m.load_state_dict(torch.load(f"{a.run}/seed_{seed}/anchored_tacs/checkpoint_best.pt", map_location="cpu", weights_only=False)["model"])
    res[seed] = {"labeled": run(m, pix_lab, pool_lab), "unlabeled": run(m, pix_unl, pool_unl)}
    print(seed, json.dumps(res[seed]), flush=True)
    del m, warm; torch.cuda.empty_cache()
mean = {pool: {k: sum(res[s][pool][k] for s in a.seeds) / len(a.seeds) for k in a.ks} for pool in ("labeled", "unlabeled")}
print("MEAN", json.dumps(mean))
Path(a.out).write_text(json.dumps({"scope": "validation only", "per_seed": res, "mean": mean}, indent=2))
