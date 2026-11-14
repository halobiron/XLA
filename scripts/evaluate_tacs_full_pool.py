"""Evaluate unchanged learned TACS with its entire fixed train candidate pool.

Selector embeddings can be cached at inference, so full-pool scoring need not
materialize 9,000 images per query. The categorical scores and hard argmax are
the existing selector, and downstream inference remains patch concatenation.
This logs the explicit C=4-train -> C=9000-inference protocol change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import zipfile

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from probe_v3_context import ROOT, TrainingImages, load_model
from search_native_context import atomic_save, json_save, paired_difference, metrics


class Images(Dataset):
    def __init__(self, source, ids, contexts=None):
        self.source, self.ids, self.contexts = source, ids, contexts

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image, label = self.source[self.ids[index]]
        if self.contexts is None:
            return image, label
        return image, self.source[self.contexts[index]][0], label


@torch.no_grad()
def encode(encoder, projector, source, ids, path, metadata, device, batch_size, workers):
    if path.exists():
        cached = torch.load(path, weights_only=False)
        if cached["metadata"] != metadata or cached["ids"].tolist() != ids:
            raise ValueError(f"Cache mismatch {path}")
        return cached["features"].to(device)
    loader = DataLoader(Images(source, ids), batch_size=batch_size, num_workers=workers, pin_memory=True)
    chunks = []
    for step, (images, _) in enumerate(loader):
        features = F.normalize(projector(encoder.encode_global(images.to(device))), dim=-1)
        chunks.append(features.cpu())
        if step % 20 == 0 or step + 1 == len(loader):
            print(f"{path.stem}: {sum(len(x) for x in chunks)}/{len(ids)}", flush=True)
    result = torch.cat(chunks)
    atomic_save({"features": result, "ids": torch.tensor(ids), "metadata": metadata}, path)
    return result.to(device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("validation", "test"), default="validation")
    p.add_argument("--output", default="outputs/tacs_full_pool_v1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--selector", choices=("learned", "frozen_dino"), default="learned")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required: use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    device = torch.device("cuda")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    full, full_entry, cfg = load_model(archive, "full_tacs", device)
    baseline, base_entry, _ = load_model(archive, "no_context", device)
    source = TrainingImages(ROOT / args.data_root)
    ids = list(range(len(source)))
    random.Random(42).shuffle(ids)
    pool_ids = ids[5000:14000]
    query_ids = ids[:5000] if args.stage == "validation" else list(range(50000, 60000))
    if set(pool_ids) & set(query_ids):
        raise ValueError("Query candidate overlap")
    protocol = {"method": "Full TACS, whole fixed-pool inference" if args.selector == "learned" else "Full task model with frozen DINO full-pool retrieval control",
                "selector": args.selector, "train_candidates_per_query": 4, "evaluation_candidates": len(pool_ids),
                "changes": ["Inference candidates increased from 4 sampled images to complete fixed 9000-image training pool"],
                "checkpoint_crc": {"full_tacs": archive.getinfo(full_entry).CRC,
                                   "no_context": archive.getinfo(base_entry).CRC},
                "pool_ids_sha256": hashlib.sha256(json.dumps(pool_ids).encode()).hexdigest(),
                "task_architecture": "unchanged query/context patch concatenation before transformer",
                "new_training": False, "candidate_labels_used_for_selection": False,
                "primary_metric": "top1 accuracy, learned full pool minus independently trained No-Context",
                "validation_selection": "no tunable hyperparameters; fixed learned scores and argmax",
                "split_seed": 42, "image_size": 224}
    lock = output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError("Changed protocol")
    json_save(protocol, lock)
    result_path = output / f"{args.stage}_result.json"
    if result_path.exists():
        print(result_path.read_text(), flush=True)
        return
    if args.stage == "test":
        val = json.loads((output / "validation_result.json").read_text())
        if val["paired_difference"]["ci95_pp"][0] <= 0:
            raise ValueError("No established positive full-validation difference")
        from evaluate_native_residual_test import TestImages
        query_source = TestImages(ROOT / args.data_root)
        local_query_ids = list(range(10000))
    else:
        query_source, local_query_ids = source, query_ids
    if args.selector == "learned":
        encoder, projector = full.selector.encoder, full.selector.projector
    else:
        from tacsx.models.dinov3_adapter import DINOv3Adapter
        encoder = DINOv3Adapter(cfg["backbone"]["repo_dir"], weights=cfg["backbone"]["weights"]).to(device).eval().requires_grad_(False)
        projector = nn.Identity()
    started = time.perf_counter()
    c = encode(encoder, projector, source, pool_ids, output / "pool_features.pt", protocol,
               device, args.batch_size, args.workers)
    q = encode(encoder, projector, query_source, local_query_ids, output / f"{args.stage}_query_features.pt", protocol,
               device, args.batch_size, args.workers)
    # Query and candidate vectors are exactly the native normalized MLP outputs.
    similarities, actions = (q @ c.T).max(-1)
    context_ids = torch.tensor(pool_ids)[actions.cpu()].tolist()
    if any(a == b for a, b in zip(query_ids, context_ids)):
        raise ValueError("Self retrieval")
    # Work around distinct train/test sources without exposing candidate labels.
    loader = DataLoader(Images(query_source, local_query_ids), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    probabilities, base_probabilities, labels, rows = [], [], [], []
    done = 0
    for images, y in loader:
        images = images.to(device)
        batch_ids = context_ids[done:done + len(y)]
        contexts = torch.stack([source[i][0] for i in batch_ids]).to(device)
        probs = full.task_model(images, contexts).softmax(-1)
        base_probs = baseline.task_model(images, None).softmax(-1)
        if done == 0 and args.selector == "learned":
            direct = full.selector(images, contexts[:, None])[:, 0]
            if not torch.allclose(direct, similarities[:len(y)], atol=1e-5, rtol=1e-5):
                raise ValueError("Cached learned scoring differs from selector forward")
        probabilities.append(probs.cpu())
        base_probabilities.append(base_probs.cpu())
        labels.append(y)
        for j, cid in enumerate(batch_ids):
            rows.append({"query_id": query_ids[done + j], "candidate_id": cid,
                         "selection_score": float(similarities[done + j]), "query_label": int(y[j]),
                         "prediction": int(probs[j].argmax()), "baseline_prediction": int(base_probs[j].argmax())})
        done += len(y)
        if done % 512 < args.batch_size or done == len(query_ids):
            print(f"{args.stage}: {done}/{len(query_ids)}, elapsed {time.perf_counter() - started:.1f}s", flush=True)
    probs, bp, labels = torch.cat(probabilities), torch.cat(base_probabilities), torch.cat(labels)
    if args.stage == "test" and int(bp.argmax(-1).eq(labels).sum()) != 8617:
        raise ValueError("Baseline test reproduction failed")
    result = {"split": args.stage, "n": len(labels), "full_tacs": metrics(probs, labels),
              "no_context": metrics(bp, labels), "paired_difference": paired_difference(probs, bp, labels),
              "only_tacs_correct": int((probs.argmax(-1).eq(labels) & ~bp.argmax(-1).eq(labels)).sum()),
              "only_baseline_correct": int((~probs.argmax(-1).eq(labels) & bp.argmax(-1).eq(labels)).sum()),
              "elapsed_seconds": time.perf_counter() - started, "protocol": protocol}
    atomic_save({"probabilities": probs, "baseline_probabilities": bp, "labels": labels,
                 "query_ids": torch.tensor(query_ids), "candidate_ids": torch.tensor(context_ids),
                 "protocol": protocol}, output / f"{args.stage}_predictions.pt")
    with (output / f"{args.stage}_selected_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(result, result_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
