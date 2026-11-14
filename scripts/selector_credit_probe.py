"""Small selector-only diagnostic on fixed features and enumerated context losses.

This is NOT Full TACS or a paper reproduction result. Original zipped models are
read-only. Only cloned MLP heads train, exclusively on training query IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from probe_v3_context import ROOT, load_model
from diagnose_fidelity import TrainingImages
from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem
from tacsx.losses.policy import reinforce_loss, standardize_advantage


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def collect(model, baseline, source, query_ids, pool, path, metadata, batch_size, device):
    values = {}
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["metadata"] != metadata or cached["query_ids"] != query_ids:
            raise ValueError(f"Cache protocol mismatch: {path}")
        values = cached["values"]
    completed = len(values["q"]) if values else 0
    for start in range(completed, len(query_ids), batch_size):
        batch_ids = query_ids[start:start + batch_size]
        examples = [source[i] for i in batch_ids]
        q = torch.stack([image for image, _ in examples]).to(device)
        y = torch.tensor([label for _, label in examples], device=device)
        candidate_ids = [pool.sample_indices(i, 4) for i in batch_ids]
        if any(query == candidate for query, row in zip(batch_ids, candidate_ids) for candidate in row):
            raise ValueError("Self-retrieval detected")
        candidates = torch.stack([torch.stack([source[i][0] for i in row]) for row in candidate_ids]).to(device)
        n = len(q)
        qz = model.selector.encoder.encode_global(q)
        cz = model.selector.encoder.encode_global(candidates.flatten(0, 1)).reshape(n, 4, -1)
        logits = torch.stack([model.task_model(q, candidates[:, j]) for j in range(4)], dim=1)
        ce = F.cross_entropy(logits.flatten(0, 1), y.repeat_interleave(4), reduction="none").reshape(n, 4)
        qce = F.cross_entropy(model.task_model(q, None), y, reduction="none")
        batch = {"q": qz, "c": cz, "ce": ce, "query_ce": qce,
                 "correct": logits.argmax(-1) == y[:, None]}
        if baseline is not None:
            batch["baseline_correct"] = baseline.task_model(q, None).argmax(-1) == y
        for key, value in batch.items():
            value = value.cpu()
            values[key] = torch.cat((values[key], value)) if key in values else value
        if len(values["q"]) % 64 == 0 or len(values["q"]) == len(query_ids):
            atomic_save({"metadata": metadata, "query_ids": query_ids, "values": values}, path)
            print(f"{metadata['split']} cache {len(values['q'])}/{len(query_ids)}", flush=True)
    return {key: value.to(device) for key, value in values.items()}


def scores(head, data, indices=None):
    q, c = data["q"], data["c"]
    if indices is not None:
        q, c = q[indices], c[indices]
    q = F.normalize(head(q), dim=-1)
    c = F.normalize(head(c), dim=-1)
    return torch.einsum("bd,bnd->bn", q, c)


@torch.no_grad()
def measure(head, data):
    logits = scores(head, data)
    action = logits.argmax(-1)
    b = torch.arange(len(action), device=action.device)
    best = data["ce"].argmin(-1)
    pair_agree = count = 0
    for j in range(4):
        for k in range(j + 1, 4):
            a, u = logits[:, j] - logits[:, k], data["ce"][:, k] - data["ce"][:, j]
            valid = (a.abs() > 1e-8) & (u.abs() > 1e-8)
            pair_agree += int(((a * u > 0) & valid).sum())
            count += int(valid.sum())
    correct = data["correct"][b, action].float().cpu().numpy()
    ce = data["ce"][b, action].cpu().numpy()
    summary = {"accuracy": float(correct.mean()), "selected_ce": float(ce.mean()),
               "expected_ce": float((logits.softmax(-1) * data["ce"]).sum(-1).mean()),
               "oracle_match": float((action == best).float().mean()),
               "rank_agreement": pair_agree / count if count else None}
    return summary, correct, ce


def intervals(a, b):
    differences = np.asarray(a) - np.asarray(b)
    indices = np.random.default_rng(20261114).integers(0, len(a), (10000, len(a)))
    return {"difference": float(differences.mean()),
            "ci95": np.quantile(differences[indices].mean(1), [.025, .975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="outputs/diagnostics/selector_credit_gpu")
    args = p.parse_args()
    if min(args.train_samples, args.val_samples, args.epochs, args.batch_size) < 1 or args.val_samples > 5000 or args.train_samples > 36000:
        p.error("Positive sizes required; validation max 5000, training max 36000")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable. Run this with .venv-gpu/Scripts/python.exe; CPU fallback is disabled.")
    started = time.perf_counter()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(20261114)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    model, entry, cfg = load_model(archive, "full_tacs", device)
    baseline, baseline_entry, _ = load_model(archive, "no_context", device)
    source = TrainingImages(ROOT / args.data_root)
    ids = list(range(len(source)))
    random.Random(42).shuffle(ids)
    validation_ids = ids[:5000]
    pool_ids = ids[5000:14000]
    train_ids = random.Random(20261114).sample(ids[14000:], args.train_samples)
    val_ids = validation_ids[:args.val_samples]
    assert not set(train_ids) & set(validation_ids)
    assert not set(pool_ids) & set(validation_ids)
    pool = FixedCandidatePool([PoolItem(i, source.targets[i]) for i in pool_ids], 42)
    metadata = {"checkpoint_crc": archive.getinfo(entry).CRC,
                "baseline_crc": archive.getinfo(baseline_entry).CRC,
                "image_size": 224, "candidates": 4, "seed": 42,
                "device_type": device.type, "dtype": "float32"}
    train = collect(model, None, source, train_ids, pool, output / "train_cache.pt",
                    {**metadata, "split": "train"}, args.batch_size, device)
    validation = collect(model, baseline, source, val_ids, pool, output / "val_cache.pt",
                         {**metadata, "split": "validation"}, args.batch_size, device)
    initial = copy.deepcopy(model.selector.projector)
    initial_train, _, _ = measure(initial, train)
    initial_val, initial_correct, initial_ce = measure(initial, validation)
    report = {"diagnostic_only_not_full_tacs": True, "train_query_ids": train_ids,
              "validation_query_ids": val_ids, "train_validation_disjoint": True,
              "encoder_and_downstream_updated": False, "metadata": metadata,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "hyperparameters": {"epochs": args.epochs, "lr": cfg["training"]["lr_selector"],
                                    "batch_size": 32, "warm_start": "saved v3 MLP", "seed": 20261114},
              "initial_train": initial_train, "initial_validation": initial_val,
              "independent_baseline_accuracy": float(validation["baseline_correct"].float().mean()),
              "arms": {}}
    for method in ("exact_expected_ce", "reinforce_cached_reward"):
        head = copy.deepcopy(initial).requires_grad_(True).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["training"]["lr_selector"]),
                                     weight_decay=float(cfg["training"]["weight_decay"]))
        generator = torch.Generator().manual_seed(20261114)
        torch.manual_seed(20261114)
        history = []
        for epoch in range(args.epochs):
            order = torch.randperm(len(train_ids), generator=generator).to(device)
            for batch in order.split(32):
                optimizer.zero_grad()
                logits = scores(head, train, batch)
                ce = train["ce"][batch]
                if method == "exact_expected_ce":
                    loss = (logits.softmax(-1) * ce).sum(-1).mean()
                else:
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    reward = train["query_ce"][batch] - ce.gather(1, action[:, None]).squeeze(1)
                    loss = reinforce_loss(dist.log_prob(action), standardize_advantage(reward))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {method} loss")
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
                metrics, _, _ = measure(head, train)
                history.append({"epoch": epoch + 1, **metrics})
                print(method, history[-1], flush=True)
        # Fixed final epoch; validation never chooses epochs or hyperparameters.
        train_metrics, _, _ = measure(head, train)
        val_metrics, correct, ce = measure(head, validation)
        report["arms"][method] = {
            "train": train_metrics, "validation": val_metrics, "history": history,
            "accuracy_vs_initial": intervals(correct, initial_correct),
            "accuracy_vs_independent_baseline": intervals(correct, validation["baseline_correct"].cpu().numpy().astype(float)),
            "ce_vs_initial": intervals(ce, initial_ce),
        }
        atomic_save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                     "diagnostic_only": True, "metadata": metadata}, output / f"{method}_head.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
        report["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    report["elapsed_seconds"] = time.perf_counter() - started
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("query_ids")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
