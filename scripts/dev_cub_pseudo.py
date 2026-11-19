"""Validation-only development: one round of self-training (pseudo-labels) at K=2, seed 42.

A teacher checkpoint (an arm trained on the 400 labeled images) labels the unlabeled training
images; the most confident --frac of them join the 400 labeled images, and a student of the
SAME arm is retrained from the shared warm-up. Each arm uses only its own teacher, so No-Context
and TACS-X get the same treatment. True labels of unlabeled images are read only to report
pseudo-label precision (diagnostic, never used for training or selection).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import torch
from PIL import Image

import run_cub_anchored as R


@torch.no_grad()
def teacher_predictions(model, pixels, pool_ids, ident, ids):
    """Primary readout of the teacher on the given unlabeled pool positions."""
    model.eval()
    is_nc = model.selector is None
    feats = None if is_nc else R.candidate_features(model, pixels)
    probs = []
    for chunk in ids.split(64):
        q = pixels[chunk]
        with torch.autocast("cuda", dtype=torch.float16):
            if is_nc:
                logits = model.task_model(q)
            else:
                s = R.utility(model, q, feats, pool_ids[chunk].cpu(), pool_ids, ident)
                logits = model.task_model(q, pixels[s.argmax(-1)])
        probs.append(logits.float().softmax(-1))
    return torch.cat(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--arm", choices=("no_context", "anchored_tacs"), required=True)
    p.add_argument("--teacher", required=True, help="folder holding the teacher checkpoint_best.pt (same arm)")
    p.add_argument("--frac", type=float, default=0.3, help="global top fraction by confidence (if --per-class 0)")
    p.add_argument("--per-class", type=int, default=0,
                   help="class-balanced: keep the K most confident images per predicted class")
    p.add_argument("--epochs", type=int, default=16, help="16 x 1,900 images ~ the 75 x 400 step budget")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warm", default="outputs/cub200_lowdata2_dev_s42")
    a = p.parse_args()
    root = HERE.parents[1]
    data = root / "data/CUB_200_2011"
    data = data / "CUB_200_2011" if (data / "CUB_200_2011").is_dir() else data
    weights = root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    repo = root / "third_party/dinov3"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    out = Path("outputs/cub200_k2_variants") / a.name
    out.mkdir(parents=True, exist_ok=True)

    orig, audit = R.protect_test_images(data)
    train, val, _, pools, ident, split = R.prepare(data, 16, 0, train_per_class=2, unlabeled=True)
    pixels, pool_ids = pools["unlabeled"]
    labeled = set(train.dataset.query_ids)
    unl_pos = torch.tensor([j for j, i in enumerate(pool_ids.tolist()) if i not in labeled]).cuda()

    R.ARM_FLAGS["anchored_tacs"] = {"aux_query_weight": 1.0, "abstain_gate": True}
    warm_path = Path(a.warm) / f"seed_{a.seed}/warmup/checkpoint_best.pt"

    def fresh(arm):
        warm = R.build_warm_model(weights, repo, a.seed)
        warm.load_state_dict(torch.load(warm_path, map_location="cpu", weights_only=False)["model"])
        return R.make_arm(arm, warm, a.seed)

    teacher = fresh(a.arm)
    teacher.load_state_dict(torch.load(Path(a.teacher) / "checkpoint_best.pt", map_location="cpu",
                                       weights_only=False)["model"])
    probs = teacher_predictions(teacher, pixels, pool_ids, ident, unl_pos)
    del teacher
    torch.cuda.empty_cache()
    conf, pred = probs.max(-1)
    if a.per_class:
        take = torch.zeros_like(conf, dtype=torch.bool)
        for c in pred.unique():
            idx = (pred == c).nonzero().squeeze(1)
            take[idx[conf[idx].argsort(descending=True)[:a.per_class]]] = True
    else:
        take = conf >= conf.quantile(1 - a.frac)
    pseudo_ids = pool_ids[unl_pos[take]].tolist()
    pseudo_labels = pred[take].tolist()

    clean = train.dataset.candidate_base
    truth = [clean.targets[i] for i in pseudo_ids]  # diagnostic only
    precision = sum(int(x == y) for x, y in zip(pseudo_labels, truth)) / len(truth)

    subset = train.dataset.base
    student_images = copy.copy(subset.dataset)
    student_images.targets = list(subset.dataset.targets)
    for i, y in zip(pseudo_ids, pseudo_labels):
        student_images.targets[i] = int(y)
    subset.dataset = student_images
    train.dataset.query_ids = list(train.dataset.query_ids) + pseudo_ids
    subset.indices = train.dataset.query_ids
    assert not set(pseudo_ids) & set(val.dataset.query_ids)

    R.dump(out / "config.json", {**vars(a), "pseudo_count": len(pseudo_ids), "train_count": len(train.dataset),
                                 "pseudo_precision_diagnostic": precision,
                                 "pseudo_classes": len(set(pseudo_labels)), "scope": "validation only"})
    print(f"pseudo {len(pseudo_ids)} images, precision {precision * 100:.1f}% (diagnostic); "
          f"train {len(train.dataset)}", flush=True)

    args = argparse.Namespace(seeds=[a.seed], arms=["no_context", "anchored_tacs"], warmup_epochs=5,
                              epochs=a.epochs, batch_size=16, train_per_class=2, max_steps=0, eval_limit=0,
                              output_path=out)
    R.STATUS_NAME = "status.json"
    train.dataset.seed = a.seed
    student = fresh(a.arm)
    t0 = time.perf_counter()
    best = R.run_phase(a.arm, student, a.epochs, train, val, pixels, pool_ids, ident, a.seed, 5, args, out)
    Image.open = orig
    R.dump(out / "image_access_audit.json", audit)
    print("RESULT", a.name, json.dumps(best), f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
