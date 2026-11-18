"""Validation-only robustness probe: corrupted query, clean context pool.

Evaluates existing validation-selected checkpoints of a run_cub_anchored.py
output (no training). The protocol and go/no-go rule are written to
probe_registration.json BEFORE any corrupted prediction is computed.

PRIMARY endpoint (fixed a priori from the only positive earlier result, the
CIFAR shared-K4 Gaussian-noise follow-up): mean top-1 over Gaussian noise
sigma 0.08 and 0.16 (pixel scale [0,1]), anchored_tacs forced-context readout
minus No-Context, paired bootstrap over validation queries.
Secondary: gated readout, paper TACS, every other condition, mean over all
eight corruptions. Official-test images are blocked and audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import run_cub_anchored as rca

MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
CONDITIONS = {"clean": None,
              "gaussian_noise_0.08": ("noise", .08), "gaussian_noise_0.16": ("noise", .16),
              "gaussian_blur_1.5": ("blur", 1.5), "gaussian_blur_3.0": ("blur", 3.0),
              "low_contrast_0.5": ("contrast", .5), "low_contrast_0.3": ("contrast", .3),
              "dark_0.5": ("brightness", .5), "dark_0.3": ("brightness", .3)}
PRIMARY = ["gaussian_noise_0.08", "gaussian_noise_0.16"]
GO_RULE = "GO for a confirmatory Kaggle run iff primary anchored_tacs:forced_context minus no_context >= +0.5pp"


def corrupt(images, condition, query_ids):
    spec = CONDITIONS[condition]
    if spec is None:
        return images
    mean, std = MEAN.to(images.device), STD.to(images.device)
    x = (images * std + mean).clamp(0, 1)
    kind, level = spec
    if kind == "noise":
        index = list(CONDITIONS).index(condition)
        noise = torch.stack([torch.randn(x.shape[1:], generator=torch.Generator().manual_seed(index * 10**6 + int(q)))
                             for q in query_ids]).to(x.device)
        x = x + level * noise
    elif kind == "blur":
        size = 2 * math.ceil(3 * level) + 1
        x = TF.gaussian_blur(x, [size, size], [level, level])
    elif kind == "contrast":
        gray = x.mean(dim=(1, 2, 3), keepdim=True)
        x = gray + level * (x - gray)
    elif kind == "brightness":
        x = x * level
    return (x.clamp(0, 1) - mean) / std


@torch.no_grad()
def evaluate(model, loader, condition, pixels, pool_ids, identities):
    model.eval()
    is_nc = model.selector is None
    features = None if is_nc else rca.candidate_features(model, pixels)
    out = {"query_only": [], "forced_context": [], "gated": [], "query_id": []}
    for batch in loader:
        q = corrupt(batch["query"].cuda(), condition, batch["query_id"])
        y = batch["label"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            base = model.task_model(q).float()
            out["query_only"].append(base.argmax(-1).eq(y).cpu())
            if not is_nc:
                scores = rca.utility(model, q, features, batch["query_id"], pool_ids, identities)
                ctx = model.task_model(q, pixels[scores.argmax(-1)]).float()
                out["forced_context"].append(ctx.argmax(-1).eq(y).cpu())
                if model.abstain_gate:
                    _, query_only = model._deploy(base, ctx, scores)
                    out["gated"].append(torch.where(query_only[:, None], base, ctx).argmax(-1).eq(y).cpu())
        out["query_id"].append(batch["query_id"])
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="outputs/cub200_anchored_dev_s42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", default=str(root / "data/CUB_200_2011"))
    p.add_argument("--weights", default=str(root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    p.add_argument("--dinov3-repo", default=str(root / "third_party/dinov3"))
    args = p.parse_args()
    run, seed = Path(args.run), args.seed
    out_dir = run / "corruption_probe"
    if (out_dir / "probe_result.json").exists():
        raise RuntimeError("Probe already evaluated for this run; registration is one-shot")
    plan = json.loads((run / "registration.json").read_text())["plan"]
    arms = [a for a in ("no_context", "paper_tacs", "anchored_tacs") if (run / f"seed_{seed}" / a / "final.json").exists()]
    checkpoints = {}
    for arm in arms:
        info = json.loads((run / f"seed_{seed}" / arm / "final.json").read_text())
        path = run / f"seed_{seed}" / arm / "checkpoint_best.pt"
        assert rca.digest(path) == info["checkpoint_sha256"], f"checkpoint changed: {path}"
        checkpoints[arm] = {"sha256": info["checkpoint_sha256"], "validation_epoch": info["best"]["epoch"]}
    rca.dump(out_dir / "probe_registration.json", {
        "registered_utc": datetime.now(timezone.utc).isoformat(), "scope": "validation only, no training",
        "conditions": {k: v for k, v in CONDITIONS.items()}, "corruption_target": "query only; context pool clean",
        "noise_seed": "condition index * 1e6 + query_id", "primary_conditions": PRIMARY,
        "primary_readout": "anchored_tacs:forced_context minus no_context", "go_rule": GO_RULE,
        "checkpoints": checkpoints, "run_plan_protocol": plan["protocol"]})

    data_root = Path(args.data_root)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    original_open, audit = rca.protect_test_images(data_root)
    try:
        _, val, _, pixels, pool_ids, identities, _ = rca.prepare(data_root, 32, 0)
        results = {}
        warm = rca.build_warm_model(Path(args.weights), Path(args.dinov3_repo), seed)
        for arm in arms:
            model = rca.make_arm(arm, warm, seed)
            model.load_state_dict(torch.load(run / f"seed_{seed}" / arm / "checkpoint_best.pt",
                                             map_location="cpu", weights_only=False)["model"])
            for condition in CONDITIONS:
                r = evaluate(model, val, condition, pixels, pool_ids, identities)
                for readout in ("query_only", "forced_context", "gated"):
                    if readout in r and (arm != "no_context" or readout == "query_only"):
                        results[(arm, readout, condition)] = r[readout]
                rca.log(f"{arm} {condition} " + json.dumps({k: round(float(v.mean()), 4) for k, v in r.items()
                                                             if k != "query_id"}))
            del model
            torch.cuda.empty_cache()
    finally:
        Image.open = original_open

    nc = lambda c: results[("no_context", "query_only", c)]
    corrupted = [c for c in CONDITIONS if c != "clean"]
    table, comparisons = {}, {}
    for (arm, readout, condition), correct in results.items():
        key = f"{arm}:{readout}"
        table.setdefault(key, {})[condition] = float(correct.mean())
        if arm != "no_context":
            comparisons.setdefault(key, {})[condition] = rca.paired_bootstrap(nc(condition), correct)
    for key in list(table):
        arm, readout = key.split(":")
        group = lambda conds: np.mean([results[(arm, readout, c)] for c in conds], axis=0)
        table[key]["primary_noise_mean"] = float(group(PRIMARY).mean())
        table[key]["all_corruptions_mean"] = float(group(corrupted).mean())
        if arm != "no_context":
            comparisons[key]["primary_noise_mean"] = rca.paired_bootstrap(
                np.mean([nc(c) for c in PRIMARY], 0), group(PRIMARY))
            comparisons[key]["all_corruptions_mean"] = rca.paired_bootstrap(
                np.mean([nc(c) for c in corrupted], 0), group(corrupted))
    primary = comparisons["anchored_tacs:forced_context"]["primary_noise_mean"]
    result = {"scope": "validation-only robustness probe, one seed, existing checkpoints",
              "accuracy": table, "difference_vs_no_context": comparisons, "primary": primary,
              "go": primary["difference_pp"] >= 0.5, "go_rule": GO_RULE, "image_access": audit}
    rca.dump(out_dir / "probe_result.json", result)
    rca.log("PRIMARY " + json.dumps(primary) + f" GO={result['go']}")


if __name__ == "__main__":
    main()
