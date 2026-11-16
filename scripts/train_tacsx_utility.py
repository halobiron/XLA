"""Portable TACS-X second-stage utility learning on frozen classifier caches.

Select checkpoints/models only on a disjoint TRAIN-split holdout. Evaluate the
chosen selector once on validation. No oracle labels enter inference features.
This trains a TACS-X extension, not original joint Gumbel/REINFORCE TACS.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from tacsx.models.utility_selector import UtilitySelector, expected_context_loss

VARIANTS = {"ranker16": (False, True), "ranker17": (True, True), "statistics17": (True, False)}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    temp.replace(path)


def pt_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def model_inputs(data, pool, indices):
    return (data["query_features"][indices].float(), pool[data["positions"][indices]].float(),
            data["pair_logits"][indices].float(), data["query_logits"][indices].float(), data["similarities"][indices].float())


@torch.no_grad()
def evaluate(model, data, pool, batch_size=256):
    model.eval()
    logits, actions, confidence = [], [], []
    for indices in torch.arange(len(data["labels"]), device=pool.device).split(batch_size):
        scores, candidates = model(*model_inputs(data, pool, indices))
        action = scores.argmax(-1)
        logits.append(candidates[torch.arange(len(indices), device=pool.device), action].cpu())
        actions.append(action.cpu())
        confidence.append(scores.softmax(-1).amax(-1).cpu())
    logits, actions = torch.cat(logits), torch.cat(actions)
    labels = data["labels"].cpu()
    return {"accuracy": float(logits.argmax(-1).eq(labels).double().mean()),
            "ce": float(F.cross_entropy(logits, labels)), "no_context_fraction": float(actions.eq(0).double().mean()) if model.allow_null else 0.}, logits, actions, torch.cat(confidence)


def to_device(data, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def clustered_interval(differences, ids):
    unique, inverse, counts = torch.unique(ids.cpu(), return_inverse=True, return_counts=True)
    sums = torch.zeros(len(unique), dtype=torch.float64)
    sums.scatter_add_(0, inverse, differences.cpu().double())
    values = (sums / counts).numpy()
    rng = np.random.default_rng(20261116)
    draws = np.concatenate([values[rng.integers(0, len(values), (100, len(values)))].mean(1) for _ in range(100)])
    return {"difference_pp": float(values.mean()), "ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
            "query_clusters": len(unique), "scope": "descriptive one-seed query interval on reused validation"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/tacsx_utility_s42_v1")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/tacsx_utility_s42_v1/fit")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--models", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("Positive epochs and batch size required")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"registered_utc": None, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                "variants": args.models, "optimizer": "AdamW lr3e-4 wd.05 cosine", "hidden": 128, "dropout": .2,
                "objective": "exact expected within-query standardized detached candidate CE, std floor.05",
                "selection": "highest internal train-holdout accuracy then lowest CE, never validation",
                "method": "TACS-X frozen-task outcome-aware utility reranking, optional NC action; not original TACS",
                "inference_cost": "score16 frozen dual-image classifier forwards plus frozen No-Context and frozen-DINO features",
                "test_access": False,
                "cache_sha256": {name: sha256(cache / name) for name in
                                  ["train.pt", "holdout.pt", "validation.pt", "pool.pt", "cache_protocol.json"] +
                                  (["extra_no_context.pt"] if (cache / "extra_no_context.pt").exists() else [])},
                "source_sha256": {"trainer": sha256(__file__), "selector": sha256(ROOT / "src/tacsx/models/utility_selector.py")}}
    path = output / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        protocol["registered_utc"] = previous["registered_utc"]
        if previous != protocol:
            raise ValueError("Training protocol changed; preserve existing run")
    else:
        protocol["registered_utc"] = datetime.now(timezone.utc).isoformat()
        json_save(protocol, path)
    lock = sha256(path)
    pool_data = torch.load(cache / "pool.pt", map_location="cpu", weights_only=False)
    train_cpu = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
    holdout_cpu = torch.load(cache / "holdout.pt", map_location="cpu", weights_only=False)
    if train_cpu["split"] != "train" or holdout_cpu["split"] != "holdout" or train_cpu["lock"] != holdout_cpu["lock"] or train_cpu["lock"] != pool_data["lock"]:
        raise ValueError("Cache provenance mismatch")
    if set(train_cpu["query_ids"].tolist()) & set(holdout_cpu["query_ids"].tolist()):
        raise ValueError("Training query leaked into internal holdout")
    train, holdout = to_device(train_cpu, args.device), to_device(holdout_cpu, args.device)
    pool = pool_data["pool_features"].to(args.device).float()
    dim = pool.shape[-1]
    choices = []
    started = time.perf_counter()
    for name in args.models:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        allow_null, use_embeddings = VARIANTS[name]
        cfg = {"embedding_dim": dim, "hidden": 128, "allow_null": allow_null, "use_embeddings": use_embeddings}
        model = UtilitySelector(**cfg).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        folder = output / name
        folder.mkdir(exist_ok=True)
        history, best = [], None
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(train["labels"]), generator=torch.Generator().manual_seed(args.seed + epoch * 101)).to(args.device)
            loss_sum = 0.
            for indices in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                scores, logits = model(*model_inputs(train, pool, indices))
                loss = expected_context_loss(scores, logits, train["labels"][indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite utility loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
            scheduler.step()
            metrics, _, _, _ = evaluate(model, holdout, pool)
            row = {"epoch": epoch + 1, "train_loss": loss_sum / len(train["labels"]), "internal_holdout": metrics}
            history.append(row)
            if best is None or (metrics["accuracy"], -metrics["ce"]) > (best["internal_holdout"]["accuracy"], -best["internal_holdout"]["ce"]):
                best = row
                pt_save({"model": model.state_dict(), "config": cfg, "best": best, "variant": name,
                         "protocol_sha256": lock, "training_cache_lock": train_cpu["lock"]}, folder / "checkpoint_best.pt")
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                print(f"{name}: {json.dumps(row)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            json_save({"state": "running", "stage": "fit", "model": name, "epoch": epoch + 1}, output / "status.json")
        json_save(history, folder / "history.json")
        choices.append({"variant": name, "best": best})
        del model, optimizer
    winner = max(choices, key=lambda d: (d["best"]["internal_holdout"]["accuracy"], -d["best"]["internal_holdout"]["ce"]))
    # Lock the head and epoch BEFORE deserializing any final validation labels.
    json_save({"selected": winner, "all_internal_choices": choices, "selection_uses_validation": False}, output / "selection.json")
    saved = torch.load(output / winner["variant"] / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    model = UtilitySelector(**saved["config"]).to(args.device)
    model.load_state_dict(saved["model"])
    validation_cpu = torch.load(cache / "validation.pt", map_location="cpu", weights_only=False)
    if validation_cpu["split"] != "validation" or validation_cpu["lock"] != train_cpu["lock"]:
        raise ValueError("Validation provenance mismatch")
    if set(validation_cpu["query_ids"].tolist()) & (set(train_cpu["query_ids"].tolist()) | set(holdout_cpu["query_ids"].tolist())):
        raise ValueError("Validation query leakage")
    validation = to_device(validation_cpu, args.device)
    metrics, logits, actions, confidence = evaluate(model, validation, pool)
    labels = validation_cpu["labels"]
    nc_logits = validation_cpu["query_logits"].float()
    correct, nc_correct = logits.argmax(-1).eq(labels), nc_logits.argmax(-1).eq(labels)
    interval = clustered_interval((correct.double() - nc_correct.double()) * 100, validation_cpu["query_ids"])
    conditions = []
    for si, sigma in enumerate((.08, .16)):
        mask = validation_cpu["sigma_index"] == si
        conditions.append({"sigma": sigma, "no_context": float(nc_correct[mask].double().mean()),
                           "tacsx_utility": float(correct[mask].double().mean())})
    result = {"scope": "one-seed exploratory TACS-X utility variant, reused validation; no label at inference",
              "selected_variant": winner, "validation": metrics, "conditions": conditions,
              "no_context_accuracy": float(nc_correct.double().mean()), "difference_vs_no_context": interval,
              "positive_point_estimate": interval["difference_pp"] > 0, "test_access": False,
              "elapsed_seconds": time.perf_counter() - started, "protocol_sha256": lock}
    if (cache / "extra_no_context.pt").exists():
        extra = torch.load(cache / "extra_no_context.pt", map_location="cpu", weights_only=False)
        if not torch.equal(extra["query_ids"], validation_cpu["query_ids"]):
            raise ValueError("Extra baseline order mismatch")
        ensemble = (nc_logits.softmax(-1) + extra["probabilities"].float()) / 2
        ensemble_correct = ensemble.argmax(-1).eq(labels)
        result["two_no_context_ensemble_accuracy"] = float(ensemble_correct.double().mean())
        result["difference_vs_two_no_context"] = clustered_interval((correct.double() - ensemble_correct.double()) * 100, validation_cpu["query_ids"])
    json_save(result, output / "result.json")
    rows = [{"model": "no_context", "mean_accuracy": result["no_context_accuracy"]},
            {"model": f"tacsx_utility_{winner['variant']}", "mean_accuracy": metrics["accuracy"]}]
    if "two_no_context_ensemble_accuracy" in result:
        rows.append({"model": "two_no_context_ensemble", "mean_accuracy": result["two_no_context_ensemble_accuracy"]})
    with (output / "comparison.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_position = actions - 1 if model.allow_null else actions
    candidate_ids = pool_data["pool_ids"][validation_cpu["positions"][torch.arange(len(actions)), selected_position.clamp_min(0)]]
    candidate_ids = torch.where(selected_position < 0, -torch.ones_like(candidate_ids), candidate_ids)
    pt_save({"logits": logits, "query_ids": validation_cpu["query_ids"], "labels": labels, "actions": actions,
             "candidate_ids": candidate_ids, "selection_probability": confidence, "sigma_index": validation_cpu["sigma_index"],
             "protocol_sha256": lock}, output / "validation_predictions.pt")
    json_save({"state": "complete", "stage": "comparison", "test_access": False}, output / "status.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
