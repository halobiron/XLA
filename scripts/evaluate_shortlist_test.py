"""Evaluate locked clean_v3 checkpoints on all 10,000 standard CIFAR-100 test images.

This user-requested report does not require a validation win. It preserves the
earlier gated evaluation and reports both the full standard split and a secondary
pixel-identity-filtered split. No training, checkpoint selection or test tuning.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100

from evaluate_native_residual_test import TestImages
from probe_v3_context import ROOT, TrainingImages
from probe_tacs_robustness import paired_ci
from search_native_context import atomic_save, json_save
from train_tacs_shortlist import build_task, fingerprints
from tacsx.data.identity import heldout_without_identity_overlap, pixel_identity_ids
from tacsx.models.retrieval_fusion import retrieve

ARMS = ("no_context", "full_tacs")


def summarize_clean(logits, labels):
    """Paired comparisons on a fixed split; intervals do not estimate seed variation."""
    correct = {arm: values.argmax(-1).eq(labels) for arm, values in logits.items()}
    ce = {arm: F.cross_entropy(values, labels, reduction="none") for arm, values in logits.items()}
    fixed = ~correct["no_context"] & correct["full_tacs"]
    broken = correct["no_context"] & ~correct["full_tacs"]
    return {
        "count": len(labels),
        "metrics": {arm: {"correct": int(correct[arm].sum()),
                          "clean_accuracy": float(correct[arm].float().mean()),
                          "clean_nll": float(ce[arm].mean())} for arm in ARMS},
        "paired_counts": {"fixed": int(fixed.sum()), "broken": int(broken.sum()),
                          "both_correct": int((correct["no_context"] & correct["full_tacs"]).sum()),
                          "both_wrong": int((~correct["no_context"] & ~correct["full_tacs"]).sum())},
        "tacs_improvement": {
            "clean_accuracy_pp": paired_ci((correct["full_tacs"].float() - correct["no_context"].float()).numpy() * 100, .975),
            "clean_nll_reduction": paired_ci((ce["no_context"] - ce["full_tacs"]).numpy(), .975)},
    }


def verify_checkpoint_lock(actual, previous):
    if actual != previous["checkpoint_sha256"]:
        raise ValueError("Checkpoints differ from the original validation lock")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/tacs_shortlist_fullft_clean_v3")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--base-archive", default="outputs.zip")
    parser.add_argument("--batch-size", type=int, default=16, help="inference batch only; training settings stay unchanged")
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        parser.error("Positive batch size and CUDA required; use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    run, data_root = ROOT / args.run, ROOT / args.data_root
    output = run / "evaluation/test_standard"
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("shared_selector_backbone", False):
        raise ValueError("This evaluation is for clean_v3's frozen-selector-encoder variant")
    histories = {arm: json.loads((run / arm / "history.json").read_text()) for arm in ARMS}
    if any(len(history) != plan["epochs"] for history in histories.values()):
        raise ValueError("Both original training budgets must be complete")
    checkpoints = {arm: run / arm / "checkpoint_best.pt" for arm in ARMS}
    hashes = {arm: fingerprints(path) for arm, path in checkpoints.items()}
    previous_lock = json.loads((run / "evaluation/checkpoint_lock.json").read_text())
    verify_checkpoint_lock(hashes, previous_lock)
    if plan != previous_lock["training_plan"]:
        raise ValueError("Training plan changed after validation")
    lock = {
        "checkpoint_sha256": hashes, "training_plan": plan,
        "training_data_sha256": fingerprints(data_root / "cifar-100-python/train"),
        "pool_cache_sha256": fingerprints(run / "train_shortlist.pt"),
        "evaluator_sha256": fingerprints(Path(__file__)),
        "original_validation_lock_sha256": fingerprints(run / "evaluation/checkpoint_lock.json"),
        "protocol": {"primary_split": "all 10000 standard CIFAR-100 test images, original order",
                     "secondary_split": "exclude pixel-identical overlaps with original train file and repeated test images",
                     "authorization": "user explicitly requested test evaluation of existing clean_v3, regardless of validation outcome",
                     "validation_outcomes_already_known": True,
                     "checkpoint_selection": "existing validation-selected checkpoints; no reselection",
                     "inference": "float32, TF32 disabled, eval mode, nearest-16 official DINO shortlist, learned projector argmax",
                     "inference_batch_size": args.batch_size,
                     "context": "unchanged 9000-image train-only candidate pool; exclude pixel-identical query aliases",
                     "metrics": ["clean top1 accuracy", "clean NLL"],
                     "uncertainty": "10000 paired-query bootstrap draws, Bonferroni over 2 metrics (97.5% each); one training seed",
                     "secondary_intervals": "descriptive; not a separate family-controlled success claim",
                     "training_deviations": plan["deviations"]}}
    if lock["training_data_sha256"] != previous_lock["training_data_sha256"]:
        raise ValueError("Training data changed")
    lock_path = output / "checkpoint_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Evaluation lock changed; refusing to overwrite")
    else:
        json_save(lock, lock_path)
        json_save({"registered_utc": datetime.now(timezone.utc).isoformat(), "lock": lock}, output / "registration.json")
    if (output / "test_result.json").exists():
        print((output / "test_result.json").read_text(), flush=True)
        return

    # Models and protocol are locked before reading the test file.
    started = time.perf_counter()
    test_file = data_root / "cifar-100-python/test"
    test_md5 = hashlib.md5(test_file.read_bytes()).hexdigest()
    if test_md5 != dict(CIFAR100.test_list)["test"]:
        raise ValueError("Test file does not match torchvision's official CIFAR-100 checksum")
    source, test = TrainingImages(data_root), TestImages(data_root)
    if len(source) != 50000 or len(test) != 10000:
        raise ValueError("Expected the standard CIFAR-100 train/test counts")
    identities = pixel_identity_ids(chain(source.images, test.images))
    test_ids = list(range(50000, 60000))
    filtered_ids, excluded = heldout_without_identity_overlap(list(range(50000)), test_ids, identities)
    cache = torch.load(run / "train_shortlist.pt", map_location="cpu", weights_only=False)
    ids = list(range(50000))
    random.Random(42).shuffle(ids)
    if cache["pool_ids"].tolist() != ids[5000:14000] or not torch.equal(cache["identity_ids"], identities[:50000]):
        raise ValueError("Candidate pool or source identities changed")
    pool_ids, pool = cache["pool_ids"].to(device), cache["pool_features"].to(device)
    pool_identity = identities[pool_ids.cpu()].to(device)
    models, best_epochs = {}, {}
    with zipfile.ZipFile(ROOT / args.base_archive) as archive:
        for arm in ARMS:
            saved = torch.load(checkpoints[arm], map_location="cpu", weights_only=False)
            if saved["plan"] != plan or saved["arm"] != arm:
                raise ValueError("Checkpoint arm/plan mismatch")
            best_epochs[arm] = saved["best"]["epoch"]
            models[arm] = build_task(archive, arm, device, plan["task_unfreeze_last_blocks"]).eval().requires_grad_(False)
            models[arm].load_state_dict(saved["model"], strict=True)
    full = models["full_tacs"]
    logits, candidates, done = {arm: [] for arm in ARMS}, [], 0
    progress = output / "test_progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        if saved["lock"] != lock:
            raise ValueError("Progress lock mismatch")
        logits, candidates, done = saved["logits"], saved["candidate_ids"], saved["done"]
    print(f"Locked checkpoints {best_epochs}; test MD5 verified; evaluating all 10000 images", flush=True)
    for start in range(done, len(test), args.batch_size):
        local = list(range(start, min(start + args.batch_size, len(test))))
        query_ids = torch.tensor([50000 + i for i in local], device=device)
        q = torch.stack([test[i][0] for i in local]).to(device)
        query_identity = identities[query_ids.cpu()].to(device)
        qfeatures = full.selector.encoder.encode_global(q)
        _, positions = retrieve(qfeatures, pool, pool_ids, plan["shortlist_k"], query_ids, query_identity, pool_identity)
        qproj = F.normalize(full.selector.projector(qfeatures), dim=-1)
        cproj = F.normalize(full.selector.projector(pool[positions]), dim=-1)
        scores = torch.einsum("bd,bnd->bn", qproj, cproj) * plan["selector_logit_scale"]
        chosen = pool_ids[positions[torch.arange(len(q), device=device), scores.argmax(-1)]]
        if identities[chosen.cpu()].eq(query_identity.cpu()).any():
            raise ValueError("Self-retrieval by pixel identity")
        c = torch.stack([source[int(i)][0] for i in chosen.cpu()]).to(device)
        logits["no_context"].append(models["no_context"].task_model(q, None).cpu())
        logits["full_tacs"].append(full.task_model(q, c).cpu())
        candidates.append(chosen.cpu())
        done = start + len(local)
        if done % (args.batch_size * 20) == 0 or done == len(test):
            atomic_save({"logits": logits, "candidate_ids": candidates, "done": done, "lock": lock}, progress)
            print(f"test: {done}/10000, {time.perf_counter() - started:.1f}s", flush=True)
    packed = {arm: torch.cat(logits[arm]) for arm in ARMS}
    if any(not torch.isfinite(values).all() for values in packed.values()):
        raise FloatingPointError("Nonfinite logits")
    verify_checkpoint_lock({arm: fingerprints(path) for arm, path in checkpoints.items()}, previous_lock)
    labels = torch.tensor(test.targets)
    keep = torch.tensor([i - 50000 for i in filtered_ids])
    result = {"split": "standard_test", **summarize_clean(packed, labels),
              "identity_filtered_test": summarize_clean({arm: values[keep] for arm, values in packed.items()}, labels[keep]),
              "excluded_duplicate_query_ids_secondary_only": excluded,
              "best_epochs": best_epochs, "official_test_md5": test_md5,
              "lock": lock, "elapsed_seconds": time.perf_counter() - started}
    atomic_save({"logits": packed, "labels": labels, "query_ids": torch.tensor(test_ids),
                 "candidate_ids": torch.cat(candidates), "filtered_query_ids": torch.tensor(filtered_ids), "lock": lock},
                output / "test_predictions.pt")
    rows = [{"split": split, "model": arm, "count": report["count"], **values}
            for split, report in (("standard_test", result), ("identity_filtered_test", result["identity_filtered_test"]))
            for arm, values in report["metrics"].items()]
    with (output / "test_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "selected_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_id", "candidate_id"))
        writer.writerows(zip(test_ids, torch.cat(candidates).tolist()))
    json_save(result, output / "test_result.json")
    print(json.dumps({key: result[key] for key in ("split", "count", "metrics", "paired_counts", "tacs_improvement")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
