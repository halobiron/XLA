"""Cache native DINOv3 features and validate an image-only late-fusion extension.

Tune uses the first 2500 validation IDs; confirmation uses the other 2500 only
after the configuration is locked. Test is a separate explicit final stage.
No candidate ground-truth labels are input to retrieval or fusion.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from probe_v3_context import ROOT, load_model
from diagnose_fidelity import TrainingImages
from tacsx.models.retrieval_fusion import FusionConfig, NativeContextFusion, NativeResidualClassifier, combine_predictions, fuse_context_logits, fuse_retrieved, retrieve


class ImageSubset(Dataset):
    def __init__(self, source, ids):
        self.source, self.ids = source, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, _ = self.source[self.ids[index]]
        return image, self.ids[index]


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def json_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


@torch.no_grad()
def extract(model, source, ids, path, metadata, batch_size, workers, device):
    if path.is_file():
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["metadata"] != metadata or data["ids"].tolist() != ids:
            raise ValueError(f"Cache protocol mismatch: {path}")
        return data
    chunks = []
    folder = path.parent / (path.stem + "_chunks")
    folder.mkdir(exist_ok=True)
    manifest = folder / "metadata.json"
    manifest_data = {"metadata": metadata, "ids": ids}
    if manifest.is_file() and json.loads(manifest.read_text()) != manifest_data:
        raise ValueError(f"Chunk protocol mismatch: {folder}")
    json_save(manifest_data, manifest)
    done = 0
    for chunk in sorted(folder.glob("chunk_*.pt")):
        data = torch.load(chunk, map_location="cpu", weights_only=True)
        if data["ids"].tolist() != ids[done:done + len(data["ids"])]:
            raise ValueError("Feature chunks are out of order")
        chunks.append(data["features"])
        done += len(data["ids"])
    loader = DataLoader(ImageSubset(source, ids[done:]), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    pending, pending_ids = [], []
    first = True
    for images, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        features = model.task_model.backbone.encode_global(images)
        if first:
            native = model.task_model(images, None)
            cached = model.task_model.head(features)
            if not torch.allclose(native, cached, atol=1e-5, rtol=1e-5):
                raise ValueError("Cached native features differ from the original No-Context forward")
            first = False
        pending.append(features.cpu())
        pending_ids.extend(batch_ids.tolist())
        if len(pending_ids) >= 1024 or done + len(pending_ids) == len(ids):
            combined = torch.cat(pending)
            atomic_save({"features": combined, "ids": torch.tensor(pending_ids)},
                        folder / f"chunk_{done:06d}.pt")
            chunks.append(combined)
            done += len(pending_ids)
            pending, pending_ids = [], []
            print(f"{path.stem}: {done}/{len(ids)}", flush=True)
    data = {"features": torch.cat(chunks), "ids": torch.tensor(ids), "metadata": metadata}
    atomic_save(data, path)
    return data


def metrics(probs, labels):
    correct = probs.argmax(-1) == labels
    ce = -probs[torch.arange(len(labels), device=labels.device), labels].clamp_min(1e-12).log()
    return {"accuracy": float(correct.float().mean()), "ce": float(ce.mean())}


def paired_difference(probs, baseline_probs, labels):
    diff = ((probs.argmax(-1) == labels).float() - (baseline_probs.argmax(-1) == labels).float()).cpu().numpy()
    indices = np.random.default_rng(20261114).integers(0, len(diff), (10000, len(diff)))
    return {"difference_pp": float(diff.mean() * 100),
            "ci95_pp": (np.quantile(diff[indices].mean(1), [.025, .975]) * 100).tolist()}


@torch.no_grad()
def tune(cache, labels, weight, bias, output, device, classifier=None, classifier_metadata=None):
    q = cache["validation"]["features"][:2500].to(device)
    query_ids = cache["validation"]["ids"][:2500].to(device)
    pool, pool_ids = cache["pool"]["features"].to(device), cache["pool"]["ids"].to(device)
    labels = labels[:2500].to(device)
    q_logits = F.linear(q, weight, bias) if classifier is None else classifier(q)
    baseline_probs = q_logits.softmax(-1)
    baseline = metrics(baseline_probs, labels)
    similarities, indices = retrieve(q, pool, pool_ids, 128, query_ids)
    rows, best, best_probs = [], None, None
    pool_logits = F.linear(pool, weight, bias) if classifier is None else classifier(pool)
    selected_logits = pool_logits[indices]
    for method, k, temperature, context_temperature in itertools.product(
        ("probability_pool", "feature_pool", "logit_pool", "log_probability_pool", "pseudo_label_pool", "pseudo_log_pool"),
        (4, 16, 64, 128), (.01, .03, .07, .15), (.1, .5, 1., 2.)):
        if method.startswith("pseudo") and context_temperature != 1.:
            continue
        w = (similarities[:, :k] / temperature).softmax(-1)
        c_logits = (selected_logits[:, :k] * w[..., None]).sum(1)
        if method in {"probability_pool", "log_probability_pool"}:
            cp = ((selected_logits[:, :k] / context_temperature).softmax(-1) * w[..., None]).sum(1)
        elif method.startswith("pseudo"):
            cp = torch.zeros_like(q_logits).scatter_add_(1, selected_logits[:, :k].argmax(-1), w)
        else:
            cp = (c_logits / context_temperature).softmax(-1)
        for query_temperature, alpha, gate in itertools.product((.5, 1., 2.), (.1, .25, .4, .6),
                                                                ("fixed", "uncertainty", "relative_confidence")):
            config = FusionConfig(k=k, retrieval_temperature=temperature, alpha=alpha, method=method,
                                  gate=gate, context_temperature=context_temperature, query_temperature=query_temperature)
            probs = combine_predictions(q_logits, cp, config, c_logits)
            row = {"config": asdict(config), **metrics(probs, labels)}
            rows.append(row)
            if best is None or (row["accuracy"], -row["ce"]) > (best["accuracy"], -best["ce"]):
                best, best_probs = row, probs.clone()
    config = FusionConfig(**best["config"])
    if classifier is None:
        production_probs, _ = fuse_retrieved(q_logits, pool[indices[:, :config.k]], similarities[:, :config.k], weight, bias, config)
    else:
        production_probs, _ = fuse_context_logits(q_logits, selected_logits[:, :config.k], similarities[:, :config.k], config)
    max_error = float((production_probs - best_probs).abs().max())
    if max_error > 1e-5:
        raise ValueError("Fast scoring and production fusion differ")
    best_probs = production_probs
    best.update(metrics(best_probs, labels))
    gain = paired_difference(best_probs, baseline_probs, labels)
    locked = {"extension": "image_only_native_context_fusion", "candidate_labels_used_at_inference": False,
              "selection_split": "validation_first_2500", "confirmation_split": "validation_last_2500",
              "test_used_for_selection": False, "baseline": baseline, "best": best,
              "tuning_difference": gain, "configurations_evaluated": len(rows),
              "fast_vs_production_max_error": max_error}
    locked["classifier"] = classifier_metadata or {"type": "original_linear_no_context"}
    json_save({"locked": locked, "results": rows}, output / "tuning.json")
    json_save(locked, output / "locked_config.json")
    print(json.dumps(locked, indent=2), flush=True)


@torch.no_grad()
def confirm(cache, labels, weight, bias, output, device, classifier=None):
    locked = json.loads((output / "locked_config.json").read_text())
    config = FusionConfig(**locked["best"]["config"])
    q = cache["validation"]["features"][2500:].to(device)
    query_ids = cache["validation"]["ids"][2500:].to(device)
    labels = labels[2500:].to(device)
    pool, pool_ids = cache["pool"]["features"].to(device), cache["pool"]["ids"].to(device)
    if classifier is None:
        model = NativeContextFusion(pool, pool_ids, weight, bias, config).eval()
        probs = model(q, query_ids)["logits"].softmax(-1)
        base_probs = F.linear(q, weight, bias).softmax(-1)
    else:
        q_logits, pool_logits = classifier(q), classifier(pool)
        values, indices = retrieve(q, pool, pool_ids, config.k, query_ids)
        probs, _ = fuse_context_logits(q_logits, pool_logits[indices], values, config)
        base_probs = q_logits.softmax(-1)
    difference = paired_difference(probs, base_probs, labels)
    result = {"split": "validation_confirmation_last_2500", "configuration_changed": False,
              "baseline": metrics(base_probs, labels), "context": metrics(probs, labels),
              "paired_difference": difference, "passes_positive_ci_gate": difference["ci95_pp"][0] > 0}
    json_save(result, output / "confirmation.json")
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cache", "train_cache", "tune", "confirm"), default="tune")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", default="outputs/native_context_search")
    p.add_argument("--feature-cache", help="reuse native features from this directory")
    p.add_argument("--classifier", choices=("linear", "query_residual"), default="linear")
    p.add_argument("--residual-checkpoint", default="outputs/native_residual_v1/query_only_control/checkpoint_best.pt")
    args = p.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA required; use .venv-gpu/Scripts/python.exe")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    feature_dir = ROOT / args.feature_cache if args.feature_cache else output
    source = TrainingImages(ROOT / args.data_root)
    ids = list(range(len(source)))
    random.Random(42).shuffle(ids)
    val_ids, pool_ids = ids[:5000], ids[5000:14000]
    assert not set(val_ids) & set(pool_ids)
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    model, entry, cfg = load_model(archive, "no_context", device)
    metadata = {"checkpoint": entry, "crc": archive.getinfo(entry).CRC, "dtype": "float32",
                "image_size": 224, "seed": 42, "candidate_labels_used": False}
    cache = {}
    for split, selected in (("pool", pool_ids), ("validation", val_ids)):
        cache[split] = extract(model, source, selected, feature_dir / f"{split}.pt",
                               {**metadata, "split": split}, args.batch_size, args.workers, device)
    if args.stage == "train_cache":
        train_ids = ids[5000:]
        path = output / "train.pt"
        folder = output / "train_chunks"
        if not path.exists() and not folder.exists():
            folder.mkdir()
            json_save({"metadata": {**metadata, "split": "train"}, "ids": train_ids}, folder / "metadata.json")
            atomic_save({"features": cache["pool"]["features"], "ids": cache["pool"]["ids"]}, folder / "chunk_000000.pt")
        extract(model, source, train_ids, path, {**metadata, "split": "train"},
                args.batch_size, args.workers, device)
    weight = model.task_model.head.weight.detach()
    bias = model.task_model.head.bias.detach()
    classifier, classifier_metadata = None, None
    if args.classifier == "query_residual":
        checkpoint_path = ROOT / args.residual_checkpoint
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["plan"]["backbone_crc"] != metadata["crc"]:
            raise ValueError("Residual classifier encoder mismatch")
        classifier = NativeResidualClassifier(weight, bias, use_context=False,
                                              hidden=saved["plan"]["hidden"], dropout=saved["plan"]["dropout"]).to(device).eval()
        classifier.load_state_dict(saved["model"])
        classifier.requires_grad_(False)
        classifier_metadata = {"type": "query_only_residual", "checkpoint": args.residual_checkpoint,
                               "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
    labels = torch.tensor([source.targets[i] for i in val_ids])
    if args.stage == "tune":
        tune(cache, labels, weight, bias, output, device, classifier, classifier_metadata)
    elif args.stage == "confirm":
        locked = json.loads((output / "locked_config.json").read_text())
        if locked["classifier"] != (classifier_metadata or {"type": "original_linear_no_context"}):
            raise ValueError("Confirmation classifier differs from locked configuration")
        confirm(cache, labels, weight, bias, output, device, classifier)


if __name__ == "__main__":
    main()
