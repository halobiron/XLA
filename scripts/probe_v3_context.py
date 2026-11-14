"""Validation-only diagnostic of existing zipped checkpoints; no training/downloads.

Oracle selections use validation labels solely to measure the potential of the
four fixed candidates. They are not deployable selectors or benchmark results.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import random
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
import numpy as np

from diagnose_fidelity import TrainingImages
from tacsx.cli import build_model
from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem


def score_variants(selector, query, candidates):
    """Post-hoc scores on fixed learned features; never update any parameters."""
    b, n, c, h, w = candidates.shape
    q = selector.encoder.encode_global(query)
    contexts = selector.encoder.encode_global(candidates.reshape(b * n, c, h, w)).reshape(b, n, -1)
    projected_q, projected_c = selector.projector(q), selector.projector(contexts)
    dot = lambda a, b: torch.einsum("bd,bnd->bn", a, b)
    cosine = lambda a, b: dot(F.normalize(a, dim=-1), F.normalize(b, dim=-1))
    current = cosine(projected_q, projected_c)
    return {"current": current, "scaled_cosine_x10": current * 10,
            "raw_dot_same_projector": dot(projected_q, projected_c),
            "cosine_without_projector": cosine(q, contexts),
            "raw_dot_without_projector": dot(q, contexts)}


def paired_intervals(records):
    """Exploratory, query-paired percentile bootstrap, including ranking clusters."""
    n = len(records)
    indices = np.random.default_rng(20261114).integers(0, n, (10000, n))
    baseline = np.asarray([r["baseline_correct"] for r in records], dtype=float)
    current = np.asarray([r["selected_correct"] for r in records], dtype=float)
    quantities = {
        "current_minus_baseline": current - baseline,
        "current_minus_expected_random": current - np.asarray([sum(r["candidate_correct"]) / 4 for r in records]),
        "oracle_min_ce_minus_baseline": np.asarray([r["oracle_min_ce_correct"] for r in records], dtype=float) - baseline,
    }
    for name in records[0]["variants"]:
        values = np.asarray([r["variants"][name]["correct"] for r in records], dtype=float)
        quantities[name + "_minus_baseline"] = values - baseline
        quantities[name + "_minus_current"] = values - current
    result = {}
    for name, differences in quantities.items():
        means = differences[indices].mean(1)
        result[name] = {"difference_pp": float(differences.mean() * 100),
                        "ci95_pp": (np.quantile(means, [.025, .975]) * 100).tolist()}
    rank = np.asarray([r["rank_agreement"] for r in records])
    result["ranking_agreement"] = {"mean": float(rank.mean()),
                                    "ci95": np.quantile(rank[indices].mean(1), [.025, .975]).tolist()}
    return result


def load_model(archive, mode, device):
    entry = next(name for name in archive.namelist()
                 if name.endswith("/checkpoint_best.pt") and
                 f"cifar100_{mode}_c4_e5_unf4v3/" in name)
    saved = torch.load(io.BytesIO(archive.read(entry)), map_location="cpu", weights_only=False)
    cfg = saved["config"]
    cfg["backbone"]["repo_dir"] = str(ROOT / "third_party/dinov3")
    cfg["backbone"]["weights"] = str(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    if not Path(cfg["backbone"]["weights"]).is_file():
        raise FileNotFoundError("Local official DINOv3 weights required; this probe never downloads them")
    model = build_model(cfg, False, include_selector=mode != "no_context")
    model.load_state_dict(saved["model"])
    return model.to(device).eval().requires_grad_(False), entry, cfg


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default="outputs.zip")
    p.add_argument("--data-root", default="data")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="outputs/diagnostics/v3_context_probe.json")
    args = p.parse_args()
    if not 1 <= args.samples <= 5000 or args.batch_size < 1:
        p.error("samples must be 1..5000 and batch size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    archive = zipfile.ZipFile(ROOT / args.archive)
    full, full_entry, cfg = load_model(archive, "full_tacs", device)
    baseline, baseline_entry, _ = load_model(archive, "no_context", device)
    if cfg["data"]["image_size"] != 224 or cfg["data"]["candidates_per_query"] != 4:
        raise ValueError("This probe expects the saved 224px C=4 protocol")
    source = TrainingImages(ROOT / args.data_root)
    ids = list(range(len(source)))
    random.Random(42).shuffle(ids)
    val_ids, train_ids = ids[:5000], ids[5000:]
    pool = FixedCandidatePool([PoolItem(i, source.targets[i]) for i in train_ids[:9000]], 42)
    records = []
    pair_agree = pair_count = 0
    for start in range(0, args.samples, args.batch_size):
        query_ids = val_ids[start:min(start + args.batch_size, args.samples)]
        samples = [source[i] for i in query_ids]
        q = torch.stack([image for image, _ in samples]).to(device)
        labels = torch.tensor([label for _, label in samples], device=device)
        candidate_ids = [pool.sample_indices(i, 4) for i in query_ids]
        candidates = torch.stack([torch.stack([source[i][0] for i in row]) for row in candidate_ids]).to(device)
        variants = score_variants(full.selector, q, candidates)
        scores = variants["current"]
        probs = scores.softmax(-1)
        logits = torch.stack([full.task_model(q, candidates[:, j]) for j in range(4)], dim=1)
        ce = F.cross_entropy(logits.flatten(0, 1), labels.repeat_interleave(4), reduction="none").reshape(-1, 4)
        correct = logits.argmax(-1) == labels[:, None]
        baseline_logits = baseline.task_model(q, None)
        baseline_ce = F.cross_entropy(baseline_logits, labels, reduction="none")
        full_query_logits = full.task_model(q, None)
        full_query_ce = F.cross_entropy(full_query_logits, labels, reduction="none")
        actions = scores.argmax(-1)
        oracle = ce.argmin(-1)
        query_agree = torch.zeros(len(q), device=device)
        query_count = torch.zeros(len(q), device=device)
        for j in range(4):
            for k in range(j + 1, 4):
                score_delta, utility_delta = scores[:, j] - scores[:, k], ce[:, k] - ce[:, j]
                valid = (score_delta.abs() > 1e-8) & (utility_delta.abs() > 1e-8)
                pair_count += valid.sum().item()
                pair_agree += ((score_delta * utility_delta > 0) & valid).sum().item()
                query_count += valid
                query_agree += ((score_delta * utility_delta > 0) & valid)
        for i, query_id in enumerate(query_ids):
            action, best = int(actions[i]), int(oracle[i])
            records.append({
                "query_id": query_id, "candidate_ids": candidate_ids[i],
                "scores": scores[i].cpu().tolist(), "probs": probs[i].cpu().tolist(),
                "candidate_ce": ce[i].cpu().tolist(), "candidate_correct": correct[i].cpu().tolist(),
                "baseline_correct": bool(baseline_logits[i].argmax() == labels[i]),
                "baseline_ce": float(baseline_ce[i]),
                "full_query_only_correct": bool(full_query_logits[i].argmax() == labels[i]),
                "full_query_only_ce": float(full_query_ce[i]),
                "selected_action": action, "selected_correct": bool(correct[i, action]),
                "selected_ce": float(ce[i, action]),
                "oracle_min_ce_action": best, "oracle_min_ce_correct": bool(correct[i, best]),
                "oracle_min_ce": float(ce[i, best]),
                "rank_agreement": float(query_agree[i] / query_count[i]) if query_count[i] else .5,
                "variants": {name: {"action": int(value[i].argmax()),
                                      "correct": bool(correct[i, value[i].argmax()]),
                                      "ce": float(ce[i, value[i].argmax()])}
                             for name, value in variants.items()},
            })
        print(f"validation queries {len(records)}/{args.samples}", flush=True)
    n = len(records)
    mean = lambda values: sum(values) / n
    summary = {
        "baseline_accuracy": mean([r["baseline_correct"] for r in records]),
        "full_query_only_accuracy": mean([r["full_query_only_correct"] for r in records]),
        "full_selector_accuracy": mean([r["selected_correct"] for r in records]),
        "expected_random_context_accuracy": mean([sum(r["candidate_correct"]) / 4 for r in records]),
        "oracle_min_ce_accuracy": mean([r["oracle_min_ce_correct"] for r in records]),
        "oracle_any_correct_upper_bound": mean([any(r["candidate_correct"]) for r in records]),
        "selector_matches_min_ce_candidate": mean([r["selected_action"] == r["oracle_min_ce_action"] for r in records]),
        "score_utility_pairwise_agreement": pair_agree / pair_count if pair_count else None,
        "mean_train_probability_of_inference_action": mean([max(r["probs"]) for r in records]),
        "mean_score_spread": mean([max(r["scores"]) - min(r["scores"]) for r in records]),
        "mean_entropy": mean([-sum(p * math.log(max(p, 1e-12)) for p in r["probs"]) for r in records]),
        "fraction_any_context_beats_independent_baseline_ce": mean([r["oracle_min_ce"] < r["baseline_ce"] for r in records]),
        "baseline_ce": mean([r["baseline_ce"] for r in records]),
        "selected_ce": mean([r["selected_ce"] for r in records]),
        "oracle_min_ce": mean([r["oracle_min_ce"] for r in records]),
        "full_query_only_ce": mean([r["full_query_only_ce"] for r in records]),
        "posthoc_variants": {name: {
            "accuracy": mean([r["variants"][name]["correct"] for r in records]),
            "ce": mean([r["variants"][name]["ce"] for r in records]),
            "changed_action_fraction": mean([r["variants"][name]["action"] != r["selected_action"] for r in records])}
            for name in records[0]["variants"]},
    }
    result = {
        "split": "validation_only", "samples": n, "seed": 42,
        "warning": "Small post-training diagnostic, not a new benchmark. Oracle uses validation labels and is never a trained/deployed selector.",
        "full_checkpoint": full_entry, "full_checkpoint_zip_crc": archive.getinfo(full_entry).CRC,
        "baseline_checkpoint": baseline_entry,
        "baseline_checkpoint_zip_crc": archive.getinfo(baseline_entry).CRC,
        "selector_and_task_share_encoder_object": full.selector.encoder is full.task_model.backbone,
        "summary": summary, "paired_bootstrap": paired_intervals(records), "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
