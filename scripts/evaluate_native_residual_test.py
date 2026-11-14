"""One locked final test of the native residual extension and matched control.

No fitting or hyperparameter selection. Checkpoints and train-only candidate
pool are fingerprinted before the test file is opened. Repeated calls resume
feature extraction and evaluate the same locked models only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from probe_v3_context import ROOT, load_model
from search_native_context import extract, atomic_save, json_save, metrics, paired_difference
from train_native_residual import contexts
from tacsx.data.cifar100 import _transform
from tacsx.models.retrieval_fusion import NativeResidualClassifier


class TestImages:
    def __init__(self, root):
        with (root / "cifar-100-python" / "test").open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        self.images = np.asarray(data["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = data["fine_labels"]
        self.transform = _transform(224, False)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), self.targets[index]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/native_residual_final_test")
    p.add_argument("--residual-run", default="outputs/native_residual_v1")
    p.add_argument("--feature-cache", default="outputs/native_context_search")
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required: use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.perf_counter()
    output, run = ROOT / args.output, ROOT / args.residual_run
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / "plan.json").read_text())
    checkpoints = {name: run / name / "checkpoint_best.pt" for name in plan["arms"]}
    pool_path = ROOT / args.feature_cache / "pool.pt"
    locked = {
        "purpose": "Final fixed-model evaluation; no selection on test",
        "primary_comparison": "query_only_control vs original_no_context",
        "secondary_comparison": "image_context vs same-capacity query_only_control",
        "checkpoint_sha256": {name: sha256(path) for name, path in checkpoints.items()},
        "pool_sha256": sha256(pool_path), "training_plan": plan,
        "epoch_selection": "validation first 2500; unchanged after confirmation",
        "validation_disclosure": "Multiple context families explored; confirmation reused across families",
        "candidate_labels_used_at_inference": False,
        "deviations": ["Frozen saved partially-finetuned native backbone", "Additional residual MLP trained on cached features for 20 epochs", "Late feature fusion rather than paper token concatenation", "Single seed; extra head training budget relative to original baseline"],
    }
    lock_path = output / "locked_protocol.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != locked:
        raise ValueError("Protocol/checkpoints changed after locking")
    json_save(locked, lock_path)
    if (output / "results.json").exists():
        print((output / "results.json").read_text(), flush=True)
        return
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    base, entry, _ = load_model(archive, "no_context", device)
    crc = archive.getinfo(entry).CRC
    if crc != plan["backbone_crc"]:
        raise ValueError("Backbone mismatch")
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    if pool["metadata"]["crc"] != crc or pool["metadata"]["split"] != "pool":
        raise ValueError("Candidate encoder/split mismatch")
    # Everything above happens before opening any test data.
    source = TestImages(ROOT / args.data_root)
    # Namespace test IDs separately from train IDs, including in saved pairs.
    ids = list(range(len(source)))
    test = extract(base, source, ids, output / "test_features.pt",
                   {"checkpoint": entry, "crc": crc, "dtype": "float32", "image_size": 224,
                    "split": "test", "protocol_sha256": sha256(lock_path)},
                   args.batch_size, args.workers, device)
    q, labels = test["features"].to(device), torch.tensor(source.targets, device=device)
    weight, bias = base.task_model.head.weight.detach(), base.task_model.head.bias.detach()
    predictions = {"original_no_context": F.linear(q, weight, bias).softmax(-1)}
    original_correct = int((predictions["original_no_context"].argmax(-1) == labels).sum())
    if original_correct != 8617:
        raise ValueError(f"Native baseline reproduction failed: {original_correct}/10000, expected 8617")
    c = contexts(q, test["ids"].to(device) + 50000, pool["features"].to(device),
                 pool["ids"].to(device), output / "test_context_pairs.pt", plan["k"], plan["temperature"])
    rows = []
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["plan"] != plan:
            raise ValueError("Residual training plan mismatch")
        model = NativeResidualClassifier(weight, bias, use_context=name == "image_context",
                                          hidden=plan["hidden"], dropout=plan["dropout"]).to(device).eval()
        model.load_state_dict(saved["model"])
        predictions[name] = model(q, c).softmax(-1)
        rows.append({"mode": name, "best_epoch": saved["best_epoch"], **metrics(predictions[name], labels)})
    rows.insert(0, {"mode": "original_no_context", "best_epoch": 0, **metrics(predictions["original_no_context"], labels)})
    comparisons = {}
    for a, b in (("query_only_control", "original_no_context"), ("image_context", "original_no_context"),
                 ("image_context", "query_only_control")):
        ca, cb = predictions[a].argmax(-1).eq(labels), predictions[b].argmax(-1).eq(labels)
        comparisons[f"{a}_vs_{b}"] = {**paired_difference(predictions[a], predictions[b], labels),
            "only_new_correct": int((ca & ~cb).sum()), "only_baseline_correct": int((~ca & cb).sum())}
    atomic_save({"probabilities": {k: v.cpu() for k, v in predictions.items()}, "labels": labels.cpu(),
                 "test_ids": test["ids"], "protocol": locked}, output / "predictions.pt")
    result = {"split": "test", "count": len(labels), "metrics": rows, "comparisons": comparisons,
              "device": torch.cuda.get_device_name(), "elapsed_seconds": time.perf_counter() - started,
              "protocol": locked}
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_save(result, output / "results.json")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
