"""Bake outputs/ into demo/data.js so demo/index.html opens offline by double-click.

Usage: python scripts/build_demo.py
Reads only existing artifacts (metrics, history, selected_pairs, CIFAR-100 images).
Nothing is recomputed: every number shown by the demo comes from these files.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEMO = ROOT / "demo"
CIFAR = ROOT / "data" / "cifar-100-python"
TEST_OFFSET = 50000  # query_id = 50000 + index into the CIFAR-100 test split

# Runs that share the main protocol (group A, C=4, seed 42, frozen backbone).
MAIN = [
    "cifar100_no_context_c4_e5", "cifar100_random_c4_e5", "cifar100_dino_sim_c4_e5",
    "cifar100_gumbel_c4_e5_actionfix", "cifar100_policy_c4_e5_actionfix",
    "cifar100_full_tacs_c4_e5_actionfix", "cifar100_topk_k2_c4_e5",
    "cifar100_topk_k3_c4_e5", "cifar100_topk_k4_c4_e5", "cifar100_adaptive_topk_c4_e5",
]
THRESH = ["cifar100_adaptive_thr50_c4_e5", "cifar100_adaptive_thr70_c4_e5", "cifar100_adaptive_topk_c4_e5"]
LABELS = {
    "no_context": "No-Context", "random_context": "Random", "dino_similarity": "DINO-similarity",
    "gumbel_only": "Gumbel-only", "policy_only": "Policy-only", "full_tacs": "Full TACS",
    "topk_tacs": "Top-K TACS-X", "adaptive_topk_tacs": "Adaptive Top-K",
}


def read_csv(p: Path):
    with open(p, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x in (None, "", "None") else float(x)


def parse_cfg(p: Path) -> dict:
    """Tiny YAML reader for the flat keys we need (avoids a PyYAML dependency)."""
    cfg, sec = {}, None
    for line in p.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        if not line.startswith(" "):
            sec = line.split(":")[0]
            continue
        k, _, v = line.strip().partition(":")
        cfg[f"{sec}.{k}"] = v.strip()
    return cfg


def main() -> None:
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
    for d in sorted(OUT.iterdir()):
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text(encoding="utf8"))
        cfg = parse_cfg(d / "config.yaml")
        hist = read_csv(d / "history.csv") if (d / "history.csv").exists() else []
        runs.append({
            "id": d.name, "mode": m["method"], "modeLabel": LABELS.get(m["method"], m["method"]),
            "seed": m.get("seed"), "C": int(cfg.get("data.candidates_per_query", 0) or 0),
            "K": int(cfg.get("topk.k", 0) or 0) if m["method"] == "topk_tacs" else None,
            "thr": float(cfg["topk.adaptive_threshold"]) if m["method"] == "adaptive_topk_tacs" else None,
            "frozen": cfg.get("backbone.freeze") == "true",
            "test": m["test_acc"], "val": m["best_val_acc"], "loss": m["avg_loss"],
            "entropy": m.get("mean_selection_entropy"), "cross": m.get("cross_class_rate"),
            "cos": m.get("mean_query_context_cosine"), "reward": m.get("mean_reward"),
            "valCurve": [round(float(h["val_accuracy"]), 4) for h in hist],
            "hasPairs": (d / "selected_pairs.csv").exists(),
        })

    # Same test queries across every main mode, so the committee can compare like with like.
    pairs = {r: {int(x["query_id"]): x for x in read_csv(OUT / r / "selected_pairs.csv")}
             for r in MAIN if (OUT / r / "selected_pairs.csv").exists()}
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

    payload = {"runs": runs, "main": MAIN, "thresh": THRESH, "queries": qdata, "images": images}
    DEMO.mkdir(exist_ok=True)
    (DEMO / "data.js").write_text("window.TACS_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf8")
    print(f"runs={len(runs)} queries={len(qdata)} images={len(images)} size={(DEMO / 'data.js').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
