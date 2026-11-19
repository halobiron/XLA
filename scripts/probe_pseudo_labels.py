"""Diagnostic only: how accurate are pseudo-labels on the unlabeled training images at K=2?

Uses the dev seed-42 No-Context checkpoint (outputs/cub200_lowdata2_dev_s42). The true labels of the
unlabeled images are read ONLY to score pseudo-label precision here; nothing is trained.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import torch
import torch.nn.functional as F
from PIL import Image

import run_cub_anchored as R


def main():
    root = HERE.parents[1]
    data = root / "data/CUB_200_2011"
    data = data / "CUB_200_2011" if (data / "CUB_200_2011").is_dir() else data
    weights = root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    repo = root / "third_party/dinov3"
    orig, _ = R.protect_test_images(data)
    try:
        train, val, _, pools, ident, _ = R.prepare(data, 16, 0, train_per_class=2, unlabeled=True)
    finally:
        Image.open = orig
    clean = train.dataset.candidate_base
    pixels, pool_ids = pools["unlabeled"]
    labeled = set(train.dataset.query_ids)
    keep = torch.tensor([int(i) not in labeled for i in pool_ids.tolist()])
    ids = pool_ids.cpu()[keep]
    truth = torch.tensor([clean.targets[int(i)] for i in ids])
    warm = R.build_warm_model(weights, repo, 42)
    model = R.make_arm("no_context", warm, 42)
    model.load_state_dict(torch.load("outputs/cub200_lowdata2_dev_s42/seed_42/no_context/checkpoint_best.pt",
                                     map_location="cpu", weights_only=False)["model"])
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for chunk in pixels[keep.cuda()].split(64):
            probs.append(model.task_model(chunk).float().softmax(-1).cpu())
    p = torch.cat(probs)
    conf, pred = p.max(-1)
    ok = pred.eq(truth)
    rows = []
    for q in (0.0, 0.5, 0.7, 0.8, 0.9):
        thr = float(conf.quantile(q))
        m = conf >= thr
        rows.append({"top_fraction": round(1 - q, 2), "threshold": round(thr, 3), "count": int(m.sum()),
                     "precision": round(float(ok[m].float().mean()) * 100, 2),
                     "classes_covered": int(pred[m].unique().numel())})
    out = {"scope": "diagnostic on unlabeled training images, dev NC checkpoint seed 42", "unlabeled": int(len(ids)),
           "overall_accuracy": round(float(ok.float().mean()) * 100, 2), "by_confidence": rows}
    Path("outputs/cub200_k2_variants/pseudo_label_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
