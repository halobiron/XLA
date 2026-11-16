"""Fixed Gaussian-noise endpoint for the shared-backbone K4 TACS-X recipe.

Default is validation. Test requires explicit --split test; this is a replication
of a previously selected one-seed endpoint, not a pristine test or clean win.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from probe_v3_context import TrainingImages, load_model
from probe_tacs_robustness import corrupt, paired_ci
from search_native_context import atomic_save, json_save
from train_tacs_shortlist import build_task, encode_candidate_globals
from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap
from tacsx.models.dinov3_adapter import DINOv3Adapter
from tacsx.models.retrieval_fusion import retrieve


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("Enable a CUDA GPU")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = args.run.resolve()
    output = run / "evaluation" / f"fixed_noise_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / "plan.json").read_text())
    if not plan.get("shared_selector_backbone") or plan["shortlist_k"] != 4 or plan["epochs"] != 10:
        raise ValueError("Expected the exact shared-K4 ten-epoch recipe")
    lock = {"checkpoint_sha256": {a: fingerprint(run / a / "checkpoint_best.pt") for a in ("no_context", "full_tacs")},
            "plan_sha256": fingerprint(run / "plan.json"), "split": args.split,
            "conditions": [["gaussian_noise", .08], ["gaussian_noise", .16]],
            "scope": "one-seed exploratory replication; clean-trained models, noisy query, clean train-only context",
            "historical_test_reuse": "official CIFAR100 test has previously been evaluated in this workspace; not pristine",
            "training_or_test_selection": False}
    path = output / "protocol.json"
    if path.exists() and json.loads(path.read_text())["lock"] != lock:
        raise ValueError("Evaluation inputs changed")
    if not path.exists():
        json_save({"registered_utc": datetime.now(timezone.utc).isoformat(), "lock": lock}, path)
    if (output / "result.json").exists():
        print((output / "result.json").read_text(), flush=True)
        return
    train_source = TrainingImages(ROOT / "data")
    order = list(range(50000))
    random.Random(42).shuffle(order)
    if args.split == "test":
        from evaluate_native_residual_test import TestImages
        source = TestImages(ROOT / "data")
        identity = pixel_identity_ids(chain(train_source.images, source.images))
        ids, excluded = heldout_without_identity_overlap(list(range(50000)), list(range(50000, 60000)), identity)
        locals_ = [i - 50000 for i in ids]
    else:
        source = train_source
        identity = pixel_identity_ids(source.images)
        ids, excluded = heldout_without_identity_overlap(order[5000:], order[:5000], identity)
        locals_ = ids
    cached = torch.load(run / "train_shortlist.pt", map_location="cpu", weights_only=False)
    pixels = torch.load(run / "pool_pixels_fp16.pt", map_location="cpu", weights_only=False)
    if not torch.equal(cached["pool_ids"], torch.tensor(order[5000:14000])) or not torch.equal(cached["pool_ids"], pixels["pool_ids"]):
        raise ValueError("Wrong fixed training candidate pool")
    pool_ids, pool_features = cached["pool_ids"].cuda(), cached["pool_features"].cuda()
    pool_pixels = pixels["pixels"].cuda()
    del pixels
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    models = {}
    for arm in ("no_context", "full_tacs"):
        model = build_task(archive, arm, "cuda", shared_selector_backbone=arm == "full_tacs")
        saved = torch.load(run / arm / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        if saved["plan"] != plan:
            raise ValueError("Checkpoint/plan mismatch")
        model.load_state_dict(saved["model"])
        models[arm] = model.eval().requires_grad_(False)
    full = models["full_tacs"]
    _, _, cfg = load_model(archive, "no_context", "cpu")
    proposal = DINOv3Adapter(cfg["backbone"]["repo_dir"], weights=cfg["backbone"]["weights"]).cuda().eval().requires_grad_(False)
    live_pool = encode_candidate_globals(full.selector.encoder, pool_pixels.unsqueeze(0))[0]
    packed, selected, targets = {a: [] for a in models}, [], []
    progress = output / "progress.pt"
    done = 0
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=False)
        if state["lock"] != lock:
            raise ValueError("Resume mismatch")
        packed, selected, targets, done = (state[k] for k in ("packed", "selected", "targets", "done"))
    started = time.perf_counter()
    for start in range(done, len(ids), args.batch_size):
        batch_ids, batch_local = ids[start:start + args.batch_size], locals_[start:start + args.batch_size]
        qids = torch.tensor(batch_ids, device="cuda")
        qidentity = identity[qids.cpu()].cuda()
        condition_logits = {a: [] for a in models}
        contexts = []
        for sigma in (.08, .16):
            q = torch.stack([source.transform(corrupt(source.images[local], "gaussian_noise", sigma, qid))
                             for local, qid in zip(batch_local, batch_ids)]).cuda()
            _, positions = retrieve(proposal.encode_global(q), pool_features, pool_ids, 4,
                                    qids, qidentity, identity[pool_ids.cpu()].cuda())
            qproj = F.normalize(full.selector.projector(full.selector.encoder.encode_global(q)), dim=-1)
            cproj = F.normalize(full.selector.projector(live_pool[positions]), dim=-1)
            scores = torch.einsum("bd,bnd->bn", qproj, cproj) * plan["selector_logit_scale"]
            chosen = positions[torch.arange(len(q), device="cuda"), scores.argmax(-1)]
            context_ids = pool_ids[chosen]
            if identity[context_ids.cpu()].eq(qidentity.cpu()).any():
                raise ValueError("Self retrieval")
            condition_logits["full_tacs"].append(full.task_model(q, pool_pixels[chosen].float()).cpu())
            condition_logits["no_context"].append(models["no_context"].task_model(q).cpu())
            contexts.append(context_ids.cpu())
        for arm in models:
            packed[arm].append(torch.stack(condition_logits[arm], 1))
        selected.append(torch.stack(contexts, 1))
        targets.extend(source.targets[i] for i in batch_local)
        done = start + len(batch_ids)
        if done % (args.batch_size * 10) == 0 or done == len(ids):
            atomic_save({"lock": lock, "packed": packed, "selected": selected, "targets": targets, "done": done}, progress)
            print(f"fixed noise {args.split}: {done}/{len(ids)} x2, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    y = torch.tensor(targets)
    logits = {a: torch.cat(values) for a, values in packed.items()}
    correct = {a: value.argmax(-1).eq(y[:, None]).double() for a, value in logits.items()}
    result = {"lock": lock, "count": len(y), "excluded_ids": excluded,
              "accuracy": {a: {"sigma008": float(v[:, 0].mean()), "sigma016": float(v[:, 1].mean()),
                                "mean": float(v.mean())} for a, v in correct.items()},
              "full_minus_no_context_pp": paired_ci((correct["full_tacs"] - correct["no_context"]).mean(1).numpy() * 100),
              "elapsed_seconds": time.perf_counter() - started}
    atomic_save({"logits": logits, "labels": y, "query_ids": torch.tensor(ids), "selected_context_ids": torch.cat(selected), "lock": lock}, output / "predictions.pt")
    json_save(result, output / "result.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
