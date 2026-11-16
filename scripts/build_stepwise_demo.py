"""Bake step-by-step No-Context vs Full-TACS traces (clean_v3) into demo/steps_data.js.

    .venv-gpu/Scripts/python.exe scripts/build_stepwise_demo.py --device cuda

* Uses only the validation split carved from CIFAR-100 train (the locked test file is never read).
* First verifies that the reconstructed models reproduce the saved validation predictions.
* Examples are a seeded random sample from each outcome group (no hand-picking); group sizes
  over the whole validation set are reported next to them.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tacsx.stepwise import RUN, StepwiseEngine  # noqa: E402

GROUPS = {"fixed": 10, "broken": 6, "both": 6, "neither": 4}


def baseline_evidence(run=RUN, archive_path=ROOT / "outputs.zip"):
    """Read training provenance only; never load models or the held-out test set."""
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    history = json.loads((run / "no_context/history.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive_path) as archive:
        entry, = [info.filename for info in archive.infolist()
                  if info.filename.endswith("checkpoint_best.pt")
                  and info.CRC == plan["common_task_initialization_crc"]]
        prefix = entry.rsplit("/", 1)[0]
        warm_history = list(csv.DictReader(io.StringIO(archive.read(prefix + "/history.csv").decode("utf-8"))))
        warm_metrics = json.loads(archive.read(prefix + "/metrics.json"))
    best = max(history, key=lambda row: (row["validation"]["accuracy"], -row["validation"]["ce"]))
    return {
        "warmStart": {"archive": "outputs.zip", "entry": prefix,
                      "checkpointCRC": plan["common_task_initialization_crc"],
                      "epochs": len(warm_history), "validationCount": 5000,
                      "bestValidationAccuracy": warm_metrics["best_val_acc"]},
        "continuation": {"epochs": len(history), "bestEpoch": best["epoch"],
                         "lrBackbone": plan["lr_backbone"], "lrHead": plan["lr_head"],
                         "weightDecay": plan["weight_decay"],
                         "validationHistory": [{"epoch": row["epoch"], "accuracy": row["validation"]["accuracy"]}
                                               for row in history]},
    }


def png(arr) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify", type=int, default=96, help="validation queries to re-run against saved predictions")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default=str(ROOT / "demo/steps_data.js"))
    p.add_argument("--refresh-baseline-only", action="store_true",
                   help="refresh baseline provenance in the existing demo, without model inference")
    args = p.parse_args()
    if args.refresh_baseline_only:
        out = Path(args.output)
        data = json.loads(out.read_text(encoding="utf-8").removeprefix("window.TACS_STEPS = ").strip().removesuffix(";"))
        data["summary"]["baseline"] = baseline_evidence()
        out.write_text("window.TACS_STEPS = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
        print(f"Refreshed baseline evidence in {out}; no inference or test access")
        return
    torch.manual_seed(0)
    engine = StepwiseEngine(args.device).load()
    saved = {arm: torch.load(RUN / arm / "validation_best.pt", weights_only=False) for arm in ("no_context", "full_tacs")}
    labels = saved["no_context"]["labels"]
    if not torch.equal(labels, saved["full_tacs"]["labels"]) or len(labels) != len(engine.val_ids):
        raise ValueError("validation order mismatch")
    for arm, value in saved.items():
        if value["epoch"] != engine.models[arm].best_epoch:
            raise ValueError(f"{arm}: validation_best epoch {value['epoch']} != checkpoint_best epoch")
    pred = {arm: v["probabilities"].argmax(1) for arm, v in saved.items()}
    # Outcome groups / confusion counts come from the locked validation evaluation (clean
    # condition = index 0), so they agree exactly with the headline accuracies.
    locked = torch.load(RUN / "evaluation/validation_predictions.pt", weights_only=False)
    if locked["query_ids"].tolist() != engine.val_ids or not torch.equal(locked["labels"], labels):
        raise ValueError("locked evaluation order mismatch")
    ok = {arm: locked["logits"][arm][:, 0].argmax(-1) == labels for arm in pred}

    # 1) faithfulness check against the predictions written during training.
    rng = random.Random(args.seed)
    check_rows = rng.sample(range(len(labels)), args.verify)
    agree = {"no_context": 0, "full_tacs": 0}
    max_diff = {"no_context": 0., "full_tacs": 0.}
    for row in check_rows:
        qid = engine.val_ids[row]
        trace, _ = engine.trace(engine.images[qid], engine.targets[qid], qid)
        for arm, key in (("no_context", "noContext"), ("full_tacs", "tacs")):
            top = trace[key]["top"][0]
            agree[arm] += int(top["class"] == int(pred[arm][row]))
            max_diff[arm] = max(max_diff[arm], abs(top["p"] - float(saved[arm]["probabilities"][row, top["class"]])))
    print(f"verification on {args.verify} validation queries: top-1 agreement {agree}, max top-prob diff {max_diff}", flush=True)
    if min(agree.values()) < args.verify - 2:
        raise ValueError("Reconstructed models do not reproduce saved validation predictions")

    # 2) whole-validation statistics.
    masks = {"fixed": ~ok["no_context"] & ok["full_tacs"], "broken": ok["no_context"] & ~ok["full_tacs"],
             "both": ok["no_context"] & ok["full_tacs"], "neither": ~ok["no_context"] & ~ok["full_tacs"]}
    counts = {k: int(v.sum()) for k, v in masks.items()}
    with (RUN / "full_tacs/validation_selected_pairs.csv").open(encoding="utf-8") as handle:
        picked = {int(r["query_id"]): int(r["candidate_id"]) for r in csv.DictReader(handle)}
    top1_ids = engine.pool_ids[engine.val_positions[:, 0]].tolist()
    differs = same_learned = same_dino = 0
    for row, qid in enumerate(engine.val_ids):
        differs += picked[qid] != top1_ids[row]
        same_learned += engine.targets[picked[qid]] == engine.targets[qid]
        same_dino += engine.targets[top1_ids[row]] == engine.targets[qid]
    n = len(engine.val_ids)
    evaluation = json.loads((RUN / "evaluation/validation_result.json").read_text(encoding="utf-8"))
    imp = evaluation["tacs_improvement"]
    summary = {
        "baseline": baseline_evidence(),
        "count": n, "counts": counts,
        "metrics": evaluation["metrics"],
        "improvement": {k: {"difference": v["difference"], "ci": v["ci"], "level": v["confidence_level"]} for k, v in imp.items()},
        "selector": {"differsFromDinoTop1": round(differs / n, 4), "sameClassLearned": round(same_learned / n, 4),
                     "sameClassDinoTop1": round(same_dino / n, 4)},
        "bestEpochs": {arm: engine.models[arm].best_epoch for arm in engine.models},
        "verification": {"queries": args.verify, "agreement": agree, "maxProbDiff": {k: round(v, 4) for k, v in max_diff.items()}},
        "plan": {k: engine.plan[k] for k in ("seed", "epochs", "batch_size", "tau", "lambda_policy", "shortlist_k",
                                              "candidate_pool", "deviations", "selector", "downstream")},
    }

    # 3) seeded random examples per outcome group.
    images, examples = {}, []
    for group, k in GROUPS.items():
        rows = torch.nonzero(masks[group]).flatten().tolist()
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            if taken == k:
                break
            qid = engine.val_ids[row]
            trace, cand_ids = engine.trace(engine.images[qid], engine.targets[qid], qid)
            # fp32 re-run can flip a borderline fp16 training-time prediction; keep groups truthful.
            if (trace["noContext"]["correct"], trace["tacs"]["correct"]) != (bool(ok["no_context"][row]), bool(ok["full_tacs"][row])):
                print(f"skip {qid}: re-run outcome differs from saved group {group}", flush=True)
                continue
            taken += 1
            for i in [qid, *cand_ids]:
                images.setdefault(str(i), png(engine.images[i]))
            trace.update(id=qid, group=group)
            examples.append(trace)
            print(f"{group}: query {qid} nc={trace['noContext']['pred']} tacs={trace['tacs']['pred']} true={trace['trueLabel']}", flush=True)
    data = {"summary": summary, "examples": examples, "images": images}
    out = Path(args.output)
    out.write_text("window.TACS_STEPS = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(examples)} examples)", flush=True)


if __name__ == "__main__":
    main()
