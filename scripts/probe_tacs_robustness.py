"""Frozen original Full TACS vs No-Context under a locked corruption suite.

This is a new project robustness benchmark, NOT CIFAR-100-C or a benchmark
reported by the paper. All eight conditions are retained regardless of outcome.
No optimization, candidate labels, oracle selection, or test-driven tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F

from probe_v3_context import ROOT, TrainingImages, load_model
from search_native_context import atomic_save, json_save
from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem


CONDITIONS = (("clean", 0.), ("gaussian_noise", .08), ("gaussian_noise", .16),
              ("gaussian_blur", .5), ("gaussian_blur", 1.),
              ("contrast", .5), ("contrast", .2), ("brightness", 1.5), ("brightness", 2.))


def corrupt(array, name, severity, image_id):
    image = Image.fromarray(array)
    if name == "gaussian_noise":
        # Same perturbation on each image/model; independent of traversal order.
        rng = np.random.default_rng(20261114 + int(image_id))
        x = np.clip(array.astype(np.float32) / 255 + rng.normal(0, severity, array.shape), 0, 1)
        return Image.fromarray(np.rint(x * 255).astype(np.uint8))
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(severity))
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(severity)
    if name == "brightness":
        return ImageEnhance.Brightness(image).enhance(severity)
    if name != "clean":
        raise ValueError(name)
    return image


def paired_ci(differences, level=.95):
    """Bootstrap queries as clusters, preserving all their corruption conditions."""
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20261114)
    draws = []
    for _ in range(100):
        indices = rng.integers(0, len(values), (100, len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    return {"difference": float(values.mean()),
            "ci": np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2]).tolist(),
            "confidence_level": level}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("probe", "confirmation", "test"), default="probe")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="outputs/tacs_robustness_v1")
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required; use .venv-gpu/Scripts/python.exe")
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
    protocol = {"benchmark": "project_native32_corruptions_v1_not_CIFAR100C",
                "conditions": list(map(list, CONDITIONS)), "query_corrupted": True, "context_corrupted": False,
                "primary_metric": "mean top1 accuracy over all 8 corruptions, equal weights",
                "primary_difference": "Full TACS minus independently trained No-Context",
                "secondary_metric": "mean CE over same 8 corruptions",
                "uncertainty": "paired query-cluster bootstrap, 10000 draws; 8-condition CIs use Bonferroni",
                "candidate_count": 4, "pool_count": 9000, "candidate_seed": 42,
                "model_crc": {"full_tacs": archive.getinfo(full_entry).CRC,
                              "no_context": archive.getinfo(base_entry).CRC},
                "optimization_or_test_selection": False,
                "splits": {"probe": "first 512 validation", "confirmation": "last 2500 validation",
                           "test": "all 10000 test; permitted only after positive primary confirmation CI"}}
    lock = output / "protocol.json"
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise ValueError("Protocol changed; preserve this experiment")
    json_save(protocol, lock)
    if (output / f"{args.split}_result.json").exists():
        print((output / f"{args.split}_result.json").read_text(), flush=True)
        return
    if args.split == "test":
        confirmation = json.loads((output / "confirmation_result.json").read_text())
        if confirmation["aggregate_accuracy_full_minus_baseline"]["ci"][0] <= 0:
            raise ValueError("Primary confirmation has not established robustness improvement")
    source = TrainingImages(ROOT / args.data_root)
    ids = list(range(len(source)))
    random.Random(42).shuffle(ids)
    pool = FixedCandidatePool([PoolItem(i, source.targets[i]) for i in ids[5000:14000]], 42)
    query_source = source
    if args.split == "test":
        from evaluate_native_residual_test import TestImages
        query_source = TestImages(ROOT / args.data_root)
        query_ids, source_ids = list(range(50000, 60000)), list(range(10000))
    else:
        query_ids = ids[:512] if args.split == "probe" else ids[2500:5000]
        source_ids = query_ids
    started = time.perf_counter()
    logits = {mode: [[] for _ in CONDITIONS] for mode in ("full_tacs", "no_context")}
    labels, selected_ids, all_scores = [], [], []
    resume = output / f"{args.split}_progress.pt"
    done = 0
    if resume.exists():
        saved = torch.load(resume, weights_only=False)
        if saved["protocol_sha256"] != hashlib.sha256(lock.read_bytes()).hexdigest():
            raise ValueError("Progress protocol mismatch")
        logits, labels, selected_ids, all_scores, done = (saved[k] for k in ("logits", "labels", "selected_ids", "scores", "done"))
    for start in range(done, len(query_ids), args.batch_size):
        batch_ids, batch_source = query_ids[start:start + args.batch_size], source_ids[start:start + args.batch_size]
        candidate_ids = torch.tensor([pool.sample_indices(i, 4) for i in batch_ids])
        if (candidate_ids == torch.tensor(batch_ids)[:, None]).any():
            raise ValueError("Self retrieval")
        candidates = torch.stack([torch.stack([source[int(i)][0] for i in row]) for row in candidate_ids]).to(device)
        b = len(batch_ids)
        cfeat = full.selector.encoder.encode_global(candidates.flatten(0, 1)).reshape(b, 4, -1)
        cfeat = F.normalize(full.selector.projector(cfeat), dim=-1)
        batch_selected, batch_scores = [], []
        for j, (name, severity) in enumerate(CONDITIONS):
            q = torch.stack([query_source.transform(corrupt(query_source.images[i], name, severity, image_id))
                             for i, image_id in zip(batch_source, batch_ids)]).to(device)
            qfeat = F.normalize(full.selector.projector(full.selector.encoder.encode_global(q)), dim=-1)
            scores = torch.einsum("bd,bnd->bn", qfeat, cfeat)
            if start == 0 and j == 0 and not torch.allclose(scores, full.selector(q, candidates), atol=1e-5, rtol=1e-5):
                raise ValueError("Cached selector differs from original Full TACS")
            action = scores.argmax(-1)
            context = candidates[torch.arange(b, device=device), action]
            logits["full_tacs"][j].append(full.task_model(q, context).cpu())
            logits["no_context"][j].append(baseline.task_model(q, None).cpu())
            batch_selected.append(candidate_ids[torch.arange(b), action.cpu()])
            batch_scores.append(scores.cpu())
        selected_ids.append(torch.stack(batch_selected, dim=1))
        all_scores.append(torch.stack(batch_scores, dim=1))
        labels.extend([query_source.targets[i] for i in batch_source])
        done = start + b
        atomic_save({"logits": logits, "labels": labels, "selected_ids": selected_ids, "scores": all_scores,
                     "done": done, "protocol_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}, resume)
        print(f"{args.split}: {done}/{len(query_ids)} queries, 9 conditions each, {time.perf_counter() - started:.1f}s", flush=True)
    labels = torch.tensor(labels)
    packed = {mode: torch.stack([torch.cat(v) for v in conditions], dim=1) for mode, conditions in logits.items()}
    correct = {mode: values.argmax(-1).eq(labels[:, None]).float().numpy() for mode, values in packed.items()}
    ce = {mode: F.cross_entropy(values.flatten(0, 1), labels.repeat_interleave(len(CONDITIONS)), reduction="none")
                    .reshape(len(labels), len(CONDITIONS)).numpy() for mode, values in packed.items()}
    rows = []
    for j, (name, severity) in enumerate(CONDITIONS):
        rows.append({"condition": name, "severity": severity,
                     **{mode: {"accuracy": float(correct[mode][:, j].mean()), "ce": float(ce[mode][:, j].mean())}
                        for mode in packed},
                     "accuracy_full_minus_baseline": paired_ci(correct["full_tacs"][:, j] - correct["no_context"][:, j],
                                                               .95 if j == 0 else 1 - .05 / 8)})
    result = {"split": args.split, "queries": len(labels), "conditions": rows,
              "aggregate": {mode: {"accuracy": float(correct[mode][:, 1:].mean()), "ce": float(ce[mode][:, 1:].mean())}
                            for mode in packed},
              "aggregate_accuracy_full_minus_baseline": paired_ci((correct["full_tacs"][:, 1:] - correct["no_context"][:, 1:]).mean(1)),
              "aggregate_ce_full_minus_baseline": paired_ci((ce["full_tacs"][:, 1:] - ce["no_context"][:, 1:]).mean(1)),
              "elapsed_seconds": time.perf_counter() - started, "protocol": protocol}
    atomic_save({"logits": packed, "labels": labels, "query_ids": torch.tensor(query_ids),
                 "selected_candidate_ids": torch.cat(selected_ids), "scores": torch.cat(all_scores),
                 "protocol": protocol}, output / f"{args.split}_predictions.pt")
    json_save(result, output / f"{args.split}_result.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
