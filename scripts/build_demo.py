"""Bake the saved runs in outputs.zip into the offline demo/data.js.

Usage: python scripts/build_demo.py
Reads only archived artifacts (metrics, history, selected_pairs, CIFAR-100 images).
Nothing is recomputed: every number shown by the demo comes from these files.
"""
from __future__ import annotations

import base64
import argparse
import csv
import hashlib
import io
import json
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
CIFAR = ROOT / "data" / "cifar-100-python"
TEST_OFFSET = 50000  # query_id = 50000 + index into the CIFAR-100 test split

# Matched v3 protocol: C=4, seed 42, last four blocks fine-tuned.
MAIN = [
    f"cifar100_{mode}_c4_e5_unf4v3" for mode in
    ("no_context", "random", "dino_sim", "gumbel", "policy", "full_tacs", "topk_k2", "topk_k4")
]
LABELS = {
    "no_context": "No-Context", "random_context": "Random", "dino_similarity": "DINO-similarity",
    "gumbel_only": "Gumbel-only", "policy_only": "Policy-only", "full_tacs": "Full TACS",
    "topk_tacs": "Top-K TACS-X", "adaptive_topk_tacs": "Adaptive Top-K",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs.zip")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        artifacts = {name: archive.read(name).decode("utf8") for name in archive.namelist()
                     if name.endswith((".json", ".yaml", ".csv"))}
    def read(run, filename):
        return artifacts[f"outputs/{run}/{filename}"]
    def rows(run, filename):
        return list(csv.DictReader(io.StringIO(read(run, filename))))

    train = pickle.load(open(CIFAR / "train", "rb"), encoding="latin1")
    test = pickle.load(open(CIFAR / "test", "rb"), encoding="latin1")
    meta = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")
    names = meta["fine_label_names"]

    def img_b64(split: str, idx: int) -> str:
        d = train if split == "train" else test
        arr = np.array(d["data"][idx]).reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr).resize((96, 96), Image.NEAREST).save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    runs = []
    for run in MAIN:
        m = json.loads(read(run, "metrics.json"))
        source_cfg = yaml.safe_load(read(run, "config.yaml"))
        cfg = {f"{section}.{key}": str(value).lower() for section, values in source_cfg.items()
               if isinstance(values, dict) for key, value in values.items()}
        hist = rows(run, "history.csv")
        runs.append({
            "id": run, "mode": m["method"], "modeLabel": LABELS.get(m["method"], m["method"]),
            "seed": m.get("seed"), "C": int(cfg.get("data.candidates_per_query", 0) or 0),
            "K": int(cfg.get("topk.k", 0) or 0) if m["method"] == "topk_tacs" else None,
            "thr": float(cfg["topk.adaptive_threshold"]) if m["method"] == "adaptive_topk_tacs" else None,
            "frozen": cfg.get("backbone.freeze") == "true" and int(cfg.get("backbone.unfreeze_last_n_blocks", 0)) == 0,
            "unfreezeBlocks": int(cfg.get("backbone.unfreeze_last_n_blocks", 0)),
            "trainerVersion": m["trainer_version"],
            "trainableParameters": m["trainable_parameters"],
            "deviations": source_cfg["experiment"].get("deviations", []),
            "test": m["test_acc"], "val": m["best_val_acc"], "loss": m["avg_loss"],
            "entropy": m.get("mean_selection_entropy"), "cross": m.get("cross_class_rate"),
            "cos": m.get("mean_query_context_cosine"), "reward": m.get("mean_reward"),
            "valCurve": [round(float(h["val_accuracy"]), 4) for h in hist],
            "hasPairs": f"outputs/{run}/selected_pairs.csv" in artifacts,
        })

    # Same test queries across every main mode, so the committee can compare like with like.
    pairs = {r: {int(x["query_id"]): x for x in rows(r, "selected_pairs.csv")}
             for r in MAIN if f"outputs/{r}/selected_pairs.csv" in artifacts}
    common = sorted(set.intersection(*[set(v) for v in pairs.values()]))
    rng = random.Random(7)
    # one query per class-diverse pick: shuffle, keep first 12 with distinct labels
    rng.shuffle(common)
    queries, seen = [], set()
    first = next(iter(pairs.values()))
    for q in common:
        lab = int(first[q]["query_label"])
        if lab in seen:
            continue
        seen.add(lab)
        queries.append(q)
        if len(queries) == 12:
            break

    images: dict[str, str] = {}
    qdata = []
    for q in queries:
        qi = q - TEST_OFFSET
        images[f"t{qi}"] = img_b64("test", qi)
        entry = {"q": q, "img": f"t{qi}", "label": names[int(first[q]["query_label"])], "modes": {}}
        for r, rows in pairs.items():
            x = rows[q]
            ids = json.loads(x.get("topk_candidate_ids")) if x.get("topk_candidate_ids") else [int(x["candidate_id"])]
            labs = json.loads(x.get("topk_candidate_labels")) if x.get("topk_candidate_labels") else [int(x["candidate_label"])]
            wts = json.loads(x.get("topk_weights")) if x.get("topk_weights") else [1.0]
            for i in ids:
                images.setdefault(f"c{i}", img_b64("train", i))
            entry["modes"][r] = [{"img": f"c{i}", "label": names[l], "same": l == int(x["query_label"]), "w": round(w, 4)}
                                 for i, l, w in zip(ids, labs, wts)]
        qdata.append(entry)

    with args.archive.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    payload = {"source": {"archive": args.archive.name, "sha256": digest, "trainerVersion": 3},
               "runs": runs, "main": MAIN, "queries": qdata, "images": images}
    DEMO.mkdir(exist_ok=True)
    (DEMO / "data.js").write_text("window.TACS_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf8")
    print(f"runs={len(runs)} queries={len(qdata)} images={len(images)} size={(DEMO / 'data.js').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
