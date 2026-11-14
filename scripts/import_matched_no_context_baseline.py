"""Reuse an already completed, exactly matched No-Context arm in a new TACS run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parents[1]
MATCHED_FIELDS = (
    "seed", "epochs", "batch_size", "lr_backbone", "lr_head", "weight_decay",
    "task_unfreeze_last_blocks", "common_task_initialization_crc",
    "official_weights_sha256", "candidate_pool", "identity_rule",
    "training_augmentation", "selection_metric",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/tacs_shortlist_fullft_clean_v3/no_context")
    parser.add_argument("--target", default="outputs/tacs_shared_backbone_k8_v1")
    args = parser.parse_args()
    src, dst = ROOT / args.source, ROOT / args.target
    target_plan = json.loads((dst / "plan.json").read_text())
    source_plan = json.loads((src.parent / "plan.json").read_text())
    if target_plan["arms"] != ["no_context", "full_tacs"]:
        raise ValueError("Target plan does not contain the matched two-arm comparison")
    if any(source_plan[key] != target_plan[key] for key in MATCHED_FIELDS):
        raise ValueError("No-Context training protocol differs; refusing checkpoint reuse")
    history = json.loads((src / "history.json").read_text())
    if len(history) != target_plan["epochs"]:
        raise ValueError("Source No-Context training budget is incomplete")
    source_checkpoint = src / "checkpoint_best.pt"
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["arm"] != "no_context" or checkpoint["plan"]["seed"] != target_plan["seed"]:
        raise ValueError("Source checkpoint identity mismatch")
    if checkpoint["best"] != max(history, key=lambda row: (row["validation"]["accuracy"], -row["validation"]["ce"])):
        raise ValueError("Saved best checkpoint does not match the completed history")
    out = dst / "no_context"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint["plan"] = target_plan
    torch.save(checkpoint, out / "checkpoint_best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(src / "validation_best.pt", out / "validation_best.pt")
    provenance = {
        "reused_completed_training": True,
        "source_run": args.source,
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_history_sha256": sha256(src / "history.json"),
        "matched_plan_fields": list(MATCHED_FIELDS),
        "source_best_epoch": checkpoint["best"]["epoch"],
        "completed_epochs": len(history),
        "reason_selector_plan_fields_are_ignored": "The No-Context arm has no selector, candidate, or context computation; its task model, data, optimizer, seed, and schedule match exactly.",
        "optimizer_state_required": False,
        "validation_data_count": int(torch.load(out / "validation_best.pt", map_location="cpu", weights_only=False)["labels"].numel()),
    }
    (out / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
