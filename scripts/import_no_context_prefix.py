"""Reuse an identical No-Context optimization prefix after heldout de-duplication.

Only supports the unambiguous one-epoch case: last==best. No optimization or
tensor changes; validation predictions are filtered by precomputed image IDs.
"""
import argparse
import copy
import json

import torch

from train_tacs_shortlist import ROOT, atomic_save, json_save, fingerprints, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="outputs/tacs_shortlist_fullft_v2")
    p.add_argument("--destination", default="outputs/tacs_shortlist_fullft_clean_v3")
    args = p.parse_args()
    source, destination = ROOT / args.source, ROOT / args.destination
    old_plan = json.loads((source / "plan.json").read_text())
    plan = json.loads((destination / "plan.json").read_text())
    keys = ("seed", "epochs", "batch_size", "lr_backbone", "lr_head", "weight_decay",
            "task_unfreeze_last_blocks", "common_task_initialization_crc", "official_weights_sha256",
            "downstream", "training_augmentation")
    if any(old_plan[key] != plan[key] for key in keys):
        raise ValueError("Optimization prefix differs")
    old = source / "no_context"
    run = destination / "no_context"
    run.mkdir(exist_ok=True)
    if (run / "checkpoint_last.pt").exists():
        raise ValueError("Destination already has a checkpoint")
    last = torch.load(old / "checkpoint_last.pt", weights_only=False)
    best = torch.load(old / "checkpoint_best.pt", weights_only=False)
    predictions = torch.load(old / "validation_best.pt", weights_only=False)
    if len(last["history"]) != 1 or best["best"]["epoch"] != 1 or predictions["epoch"] != 1:
        raise ValueError("Only a unique first-epoch best can be reused safely")
    if any(not torch.equal(value, best["model"][key]) for key, value in last["model"].items()):
        raise ValueError("Last and best weights differ")
    old_ids = torch.load(source / "validation_shortlist.pt", weights_only=False)["query_ids"]
    ids = torch.load(destination / "validation_shortlist.pt", weights_only=False)["query_ids"]
    lookup = {qid: index for index, qid in enumerate(old_ids)}
    positions = [lookup[qid] for qid in ids]
    probs, labels = predictions["probabilities"][positions], predictions["labels"][positions]
    row = copy.deepcopy(last["history"][0])
    row["validation_before_identity_filter"] = row["validation"]
    row["validation"] = {**metrics(probs, labels), "count": len(labels), "filtered_existing_predictions": True}
    provenance = {"source_checkpoint_sha256": fingerprints(old / "checkpoint_last.pt"),
                  "unchanged_optimization_keys": list(keys), "training_repeated": False,
                  "validation_before": len(old_ids), "validation_after": len(ids),
                  "excluded_ids": sorted(set(old_ids) - set(ids)), "weights_unchanged": True,
                  "optimizer_scheduler_scaler_unchanged": True}
    last.update(plan=plan, history=[row], best=row, imported_prefix=provenance)
    best.update(plan=plan, best=row, imported_prefix=provenance)
    atomic_save(last, run / "checkpoint_last.pt")
    atomic_save(best, run / "checkpoint_best.pt")
    atomic_save({"probabilities": probs, "labels": labels, "epoch": 1}, run / "validation_best.pt")
    json_save([row], run / "history.json")
    json_save(provenance, run / "prefix_provenance.json")
    print(json.dumps({"validation": row["validation"], "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
