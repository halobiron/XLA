"""Prospectively registered clean/robustness evaluation of matched TACS models.

Three family-wise controlled comparisons to independently trained No-Context:
clean accuracy, mean accuracy across eight locked corruptions, clean NLL.
All three are reported; no per-class/per-condition winner becomes the claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import chain
import random
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from probe_v3_context import ROOT, TrainingImages
from probe_tacs_robustness import CONDITIONS, corrupt, paired_ci
from search_native_context import atomic_save, json_save
from train_tacs_shortlist import build_task, fingerprints
from tacsx.models.retrieval_fusion import retrieve
from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap


def summarize(packed, labels):
    correct = {mode: value.argmax(-1).eq(labels[:, None]).float().numpy() for mode, value in packed.items()}
    ce = {mode: F.cross_entropy(value.flatten(0, 1), labels.repeat_interleave(len(CONDITIONS)), reduction="none")
          .reshape(len(labels), len(CONDITIONS)).numpy() for mode, value in packed.items()}
    delta = correct["full_tacs"] - correct["no_context"]
    quantities = {"clean_accuracy_pp": delta[:, 0] * 100,
                  "corrupted_mean_accuracy_pp": delta[:, 1:].mean(1) * 100,
                  "clean_nll_reduction": ce["no_context"][:, 0] - ce["full_tacs"][:, 0]}
    differences = {key: paired_ci(value, 1 - .05 / 3) for key, value in quantities.items()}
    completion_quantities = dict(quantities)
    for name, columns in {"gaussian_noise": (1, 2), "gaussian_blur": (3, 4),
                          "contrast": (5, 6), "brightness": (7, 8)}.items():
        completion_quantities[name + "_mean_accuracy_pp"] = delta[:, columns].mean(1) * 100
    completion = {key: paired_ci(value, 1 - .05 / 7) for key, value in completion_quantities.items()}
    return {"metrics": {mode: {"clean_accuracy": float(correct[mode][:, 0].mean()),
                               "corrupted_mean_accuracy": float(correct[mode][:, 1:].mean()),
                               "clean_nll": float(ce[mode][:, 0].mean()),
                               "corrupted_mean_nll": float(ce[mode][:, 1:].mean())} for mode in packed},
            "tacs_improvement": differences,
            "completion_evidence_seven_comparisons": completion,
            "positive_familywise_metrics": [key for key, value in completion.items() if value["ci"][0] > 0],
            "all_conditions": [{"name": name, "severity": severity,
                                **{mode: {"accuracy": float(correct[mode][:, j].mean()), "nll": float(ce[mode][:, j].mean())}
                                   for mode in packed}} for j, (name, severity) in enumerate(CONDITIONS)]}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("register", "validation", "test"), default="register")
    p.add_argument("--run", default="outputs/tacs_shortlist_fullft_clean_v3")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    run = ROOT / args.run
    output = run / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / "plan.json").read_text())
    protocol = {"training_plan_sha256": fingerprints(run / "plan.json"),
                "models": ["no_context", "full_tacs"],
                "prospective_metrics": ["clean top1 accuracy", "mean top1 accuracy over all eight corruptions", "clean NLL"],
                "comparisons": "Full TACS vs independent No-Context, same task initialization/epochs/data/optimizer",
                "family_confidence": 1 - .05 / 3,
                "uncertainty": "10000 paired query-cluster bootstrap draws, Bonferroni over 3 metrics",
                "positive_sign": "TACS minus No-Context for accuracy; No-Context minus TACS for NLL",
                "conditions": list(map(list, CONDITIONS)), "benchmark": "project native32 corruptions; not CIFAR100C or a paper benchmark",
                "context": "clean original fixed train pool; shortlist recomputed from each corrupted query without labels",
                "identity_exclusion": "block pixel-identical candidates; validation excludes any training duplicates and repeats; test excludes any original CIFAR train-file duplicates and repeats",
                "inference_precision": "float32, TF32 disabled; both models same precision",
                "checkpoint_selection": "best clean validation accuracy, clean NLL tie-break, independently for each arm",
                "test_gate": "training complete for both arms, at least one positive family-controlled validation metric; lock checkpoint hashes before test",
                "report": "all metrics/conditions including negative differences, one training seed; query CIs do not estimate seed variation"}
    path = output / "prospective_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Changed evaluation protocol")
    json_save(protocol, path)
    supplemental_path = output / "prospective_corruption_groups.json"
    supplemental_spec = {"groups": {"gaussian_noise": [1, 2], "gaussian_blur": [3, 4],
                                     "contrast": [5, 6], "brightness": [7, 8]},
                         "metric": "query-paired mean top1 accuracy across both fixed severities of each corruption type",
                         "completion_family": "all 3 original comparisons plus all 4 corruption-type means",
                         "completion_confidence": 1 - .05 / 7,
                         "multiplicity": "Bonferroni over 7; original 3-comparison intervals retained descriptively, never used alone for completion",
                         "registration_scope": "before any robustness evaluation of these trained shortlist models"}
    if supplemental_path.exists():
        supplemental = json.loads(supplemental_path.read_text())
        if supplemental["specification"] != supplemental_spec:
            raise ValueError("Changed supplemental protocol")
    else:
        if (output / "validation_result.json").exists() or (output / "validation_progress.pt").exists():
            raise ValueError("Cannot register corruption groups after observing new robustness outcomes")
        supplemental = {"registered_utc": datetime.now(timezone.utc).isoformat(), "specification": supplemental_spec}
        json_save(supplemental, supplemental_path)
    if args.stage == "register":
        print(json.dumps(protocol, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        p.error("CUDA required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    checkpoints = {mode: run / mode / "checkpoint_best.pt" for mode in protocol["models"]}
    hashes = {mode: fingerprints(path) for mode, path in checkpoints.items()}
    for mode in checkpoints:
        history = json.loads((run / mode / "history.json").read_text())
        if len(history) != plan["epochs"]:
            raise ValueError("Finish the matched training budget before evaluating")
    lock_path = output / "checkpoint_lock.json"
    lock = {"checkpoint_sha256": hashes, "evaluation_protocol_sha256": fingerprints(path),
            "supplemental_protocol_sha256": fingerprints(supplemental_path), "training_plan": plan}
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Checkpoint/evaluation lock changed")
    json_save(lock, lock_path)
    result_path = output / f"{args.stage}_result.json"
    if result_path.exists():
        print(result_path.read_text(), flush=True)
        return
    if args.stage == "test":
        val = json.loads((output / "validation_result.json").read_text())
        if not val["positive_familywise_metrics"] or val["lock"] != lock:
            raise ValueError("No locked positive validation evidence")
    source = TrainingImages(ROOT / "data")
    indices = list(range(50000))
    random.Random(42).shuffle(indices)
    query_source = source
    query_ids, local_ids = indices[:5000], indices[:5000]
    if args.stage == "test":
        from evaluate_native_residual_test import TestImages
        query_source = TestImages(ROOT / "data")
        query_ids, local_ids = list(range(50000, 60000)), list(range(10000))
        identity_ids = pixel_identity_ids(chain(source.images, query_source.images))
        query_ids, excluded = heldout_without_identity_overlap(list(range(50000)), query_ids, identity_ids)
        local_ids = [i - 50000 for i in query_ids]
    else:
        identity_ids = pixel_identity_ids(source.images)
        query_ids, excluded = heldout_without_identity_overlap(indices[5000:], query_ids, identity_ids)
        local_ids = query_ids
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    models = {}
    for mode in checkpoints:
        models[mode] = build_task(archive, mode, device, plan["task_unfreeze_last_blocks"]).eval().requires_grad_(False)
        saved = torch.load(checkpoints[mode], map_location=device, weights_only=False)
        if saved["plan"] != plan:
            raise ValueError("Training/checkpoint mismatch")
        models[mode].load_state_dict(saved["model"])
    full = models["full_tacs"]
    cache = torch.load(run / "train_shortlist.pt", weights_only=False)
    pool_ids, pool = cache["pool_ids"].to(device), cache["pool_features"].to(device)
    if not torch.equal(cache["identity_ids"], identity_ids[:50000]):
        raise ValueError("Source identities changed")
    pool_identity = identity_ids[pool_ids.cpu()].to(device)
    output_logits = {mode: [[] for _ in CONDITIONS] for mode in models}
    labels, selected = [], []
    progress = output / f"{args.stage}_progress.pt"
    done = 0
    if progress.exists():
        saved = torch.load(progress, weights_only=False)
        if saved["lock"] != lock:
            raise ValueError("Progress belongs to different models")
        output_logits, labels, selected, done = (saved[key] for key in ("logits", "labels", "selected", "done"))
    started = time.perf_counter()
    for start in range(done, len(query_ids), args.batch_size):
        batch_ids, batch_local = query_ids[start:start + args.batch_size], local_ids[start:start + args.batch_size]
        batch_selected = []
        for j, (name, severity) in enumerate(CONDITIONS):
            q = torch.stack([query_source.transform(corrupt(query_source.images[local], name, severity, qid))
                             for local, qid in zip(batch_local, batch_ids)]).to(device)
            # Official frozen DINO proposal + the trained TACS utility projection.
            qfeatures = full.selector.encoder.encode_global(q)
            batch_identity = identity_ids[torch.tensor(batch_ids)].to(device)
            _, positions = retrieve(qfeatures, pool, pool_ids, plan["shortlist_k"], torch.tensor(batch_ids, device=device),
                                    batch_identity, pool_identity)
            qproj = F.normalize(full.selector.projector(qfeatures), dim=-1)
            cproj = F.normalize(full.selector.projector(pool[positions]), dim=-1)
            scores = torch.einsum("bd,bnd->bn", qproj, cproj) * plan["selector_logit_scale"]
            action = scores.argmax(-1)
            context_ids = pool_ids[positions[torch.arange(len(q), device=device), action]]
            if identity_ids[context_ids.cpu()].eq(batch_identity.cpu()).any():
                raise ValueError("Self retrieval")
            c = torch.stack([source[int(i)][0] for i in context_ids.cpu()]).to(device)
            output_logits["full_tacs"][j].append(full.task_model(q, c).cpu())
            output_logits["no_context"][j].append(models["no_context"].task_model(q, None).cpu())
            batch_selected.append(context_ids.cpu())
        labels.extend([query_source.targets[i] for i in batch_local])
        selected.append(torch.stack(batch_selected, dim=1))
        done = start + len(batch_ids)
        atomic_save({"logits": output_logits, "labels": labels, "selected": selected, "done": done,
                     "lock": lock}, progress)
        print(f"{args.stage}: {done}/{len(query_ids)} queries x9 conditions, {time.perf_counter() - started:.1f}s", flush=True)
    labels = torch.tensor(labels)
    packed = {mode: torch.stack([torch.cat(condition) for condition in values], dim=1) for mode, values in output_logits.items()}
    result = {"split": args.stage, "count": len(labels), "excluded_duplicate_query_ids": excluded,
              **summarize(packed, labels), "lock": lock,
              "elapsed_seconds": time.perf_counter() - started}
    atomic_save({"logits": packed, "labels": labels, "query_ids": torch.tensor(query_ids),
                 "candidate_ids": torch.cat(selected), "lock": lock}, output / f"{args.stage}_predictions.pt")
    rows = [{"model": mode, **value} for mode, value in result["metrics"].items()]
    with (output / f"{args.stage}_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(result, result_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
