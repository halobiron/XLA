"""Matched frozen-feature query-only/context residual experiments on train data.

This is a logged TACS-X late-fusion extension. Original backbone and head remain
fixed; candidate ground-truth labels are never inputs. Only validation[0:2500]
selects epochs. The other half is evaluated by a separate confirm command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import zipfile

import torch
import torch.nn.functional as F

from probe_v3_context import ROOT, load_model
from diagnose_fidelity import TrainingImages
from search_native_context import atomic_save, json_save, metrics, paired_difference
from tacsx.models.retrieval_fusion import NativeResidualClassifier, retrieve


@torch.no_grad()
def contexts(query, query_ids, pool, pool_ids, path, k, temperature):
    meta = {"query_ids": query_ids.cpu().tolist(), "pool_ids": pool_ids.cpu().tolist(),
            "k": k, "temperature": temperature}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["metadata"] != meta:
            raise ValueError("Context cache mismatch")
        return saved["features"].to(query.device)
    features, selected, weights = [], [], []
    for start in range(0, len(query), 512):
        similarities, indices = retrieve(query[start:start + 512], pool, pool_ids, k,
                                        query_ids[start:start + 512])
        w = (similarities / temperature).softmax(-1)
        features.append((pool[indices] * w[..., None]).sum(1).cpu())
        selected.append(pool_ids[indices].cpu())
        weights.append(w.cpu())
    result = {"metadata": meta, "features": torch.cat(features),
              "candidate_ids": torch.cat(selected), "weights": torch.cat(weights)}
    atomic_save(result, path)
    return result["features"].to(query.device)


@torch.no_grad()
def evaluate(model, query, context, labels):
    logits = model(query, context)
    probs = logits.softmax(-1)
    return metrics(probs, labels), probs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("train", "confirm"), default="train")
    p.add_argument("--cache", default="outputs/native_context_search")
    p.add_argument("--output", default="outputs/native_residual_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=.0003)
    p.add_argument("--dropout", type=float, default=.2)
    p.add_argument("--label-smoothing", type=float, default=.05)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--temperature", type=float, default=.07)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required; use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    output, cache_dir = ROOT / args.output, ROOT / args.cache
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache = {key: torch.load(cache_dir / f"{key}.pt", map_location="cpu", weights_only=False)
             for key in ("train", "validation", "pool")}
    if len(cache["train"]["ids"]) != 45000 or len(cache["validation"]["ids"]) != 5000:
        raise ValueError("Expected 45k training and 5k validation features")
    if set(cache["train"]["ids"].tolist()) & set(cache["validation"]["ids"].tolist()):
        raise ValueError("Train/validation overlap")
    source = TrainingImages(ROOT / "data")
    labels = torch.tensor(source.targets, device=device)
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    base, entry, cfg = load_model(archive, "no_context", device)
    crc = archive.getinfo(entry).CRC
    if any(value["metadata"]["crc"] != crc for value in cache.values()):
        raise ValueError("Feature encoder does not match the baseline checkpoint")
    weight, bias = base.task_model.head.weight.detach().clone(), base.task_model.head.bias.detach().clone()
    del base
    pool, pool_ids = cache["pool"]["features"].to(device), cache["pool"]["ids"].to(device)
    data = {}
    for split in ("train", "validation"):
        q, ids = cache[split]["features"].to(device), cache[split]["ids"].to(device)
        context = contexts(q, ids, pool, pool_ids, output / f"{split}_context.pt", args.k, args.temperature)
        data[split] = {"q": q, "c": context, "ids": ids, "labels": labels[ids]}
    plan = {"extension": "native_feature_residual", "backbone_checkpoint": entry, "backbone_crc": crc,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": .05, "dropout": args.dropout, "hidden": 384,
            "label_smoothing": args.label_smoothing, "k": args.k, "temperature": args.temperature,
            "seed": 20261114, "candidate_labels_used_at_inference": False,
            "train_query_count": 45000, "tuning_count": 2500, "test_used": False,
            "arms": ["query_only_control", "image_context"]}
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Different experiment settings; use a new output directory")
    json_save(plan, plan_path)
    validation = data["validation"]
    if args.stage == "confirm":
        comparison = {}
        probabilities = {}
        for name, use_context in (("query_only_control", False), ("image_context", True)):
            saved = torch.load(output / name / "checkpoint_best.pt", map_location=device, weights_only=False)
            model = NativeResidualClassifier(weight, bias, use_context=use_context, dropout=args.dropout).to(device)
            model.load_state_dict(saved["model"])
            model.eval()
            result, probs = evaluate(model, validation["q"][2500:], validation["c"][2500:], validation["labels"][2500:])
            comparison[name] = {"best_epoch": saved["best_epoch"], **result}
            probabilities[name] = probs
        native = F.linear(validation["q"][2500:], weight, bias).softmax(-1)
        comparison["original_no_context"] = metrics(native, validation["labels"][2500:])
        comparison["context_vs_query_control"] = paired_difference(probabilities["image_context"], probabilities["query_only_control"], validation["labels"][2500:])
        comparison["context_vs_original"] = paired_difference(probabilities["image_context"], native, validation["labels"][2500:])
        json_save(comparison, output / "confirmation.json")
        print(json.dumps(comparison, indent=2), flush=True)
        return
    for name, use_context in (("query_only_control", False), ("image_context", True)):
        run = output / name
        run.mkdir(exist_ok=True)
        if (run / "metrics.json").is_file():
            print(f"skip completed {name}", flush=True)
            continue
        torch.manual_seed(plan["seed"])
        model = NativeResidualClassifier(weight, bias, use_context=use_context, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        generator = torch.Generator().manual_seed(plan["seed"])
        history, best, first_epoch = [], None, 0
        last = run / "checkpoint_last.pt"
        if last.is_file():
            saved = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            generator.set_state(saved["generator"].cpu())
            torch.set_rng_state(saved["rng"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
            history, best, first_epoch = saved["history"], saved["best"], len(saved["history"])
        for epoch in range(first_epoch, args.epochs):
            model.train()
            order = torch.randperm(45000, generator=generator).to(device)
            train_loss = 0.
            for batch in order.split(args.batch_size):
                optimizer.zero_grad()
                logits = model(data["train"]["q"][batch], data["train"]["c"][batch])
                loss = F.cross_entropy(logits, data["train"]["labels"][batch], label_smoothing=args.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite residual loss")
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach()) * len(batch)
            scheduler.step()
            model.eval()
            result, _ = evaluate(model, validation["q"][:2500], validation["c"][:2500], validation["labels"][:2500])
            row = {"epoch": epoch + 1, "train_loss": train_loss / 45000, **result}
            history.append(row)
            if best is None or (result["accuracy"], -result["ce"]) > (best["accuracy"], -best["ce"]):
                best = row
                atomic_save({"model": model.state_dict(), "plan": plan, "best_epoch": epoch + 1}, run / "checkpoint_best.pt")
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                         "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                         "history": history, "best": best}, last)
            json_save(history, run / "history.json")
            print(name, row, flush=True)
        json_save({"plan": plan, "best_tuning": best, "trainable_parameters": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": time.perf_counter() - started}, run / "metrics.json")
    print("Matched residual experiments complete; confirmation has not been evaluated.", flush=True)


if __name__ == "__main__":
    main()
