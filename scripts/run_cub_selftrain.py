"""Confirmatory self-training run at K=2 (2 labeled images per class), official test evaluated once.

Teachers are the locked checkpoints of outputs/cub200_unlabeled_v1 (seeds 10-13). Per seed and arm, the
teacher labels the unlabeled training images (its own primary readout), the K most confident images
per predicted class join the 400 labeled images, and a student of the same arm is retrained from that
seed's warm-up. Settings were chosen on validation with dev seed 42 only (scripts/dev_cub_pseudo.py);
official test results of seeds 10-13 were never used to choose anything.

Arms
  nc_pl    No-Context student, teacher = No-Context            (readout: query_only)
  tacs_pl  TACS-X student (unlabeled pool), teacher = TACS-X    (readout: forced_context)
PRIMARY: tacs_pl forced_context minus nc_pl query_only, mean over seeds, 95% paired bootstrap over test.
Secondary: tacs_pl vs the teacher TACS-X (cub200_unlabeled_v1), nc_pl vs No-Context, tacs_pl query_only.

Stages: register (once) -> run (validation only, test pixels blocked) -> test (refuses a second run).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "src"))
import numpy as np
import torch
from PIL import Image

import run_cub_anchored as R

ARMS = {"nc_pl": {"arm": "no_context", "teacher": "no_context", "readout": "query_only"},
        "tacs_pl": {"arm": "anchored_tacs", "teacher": "anchored_tacs_unlabeled", "readout": "forced_context"}}


def make_plan(a):
    return {"protocol": "cub_selftrain_v1", "teacher_run": a.teacher_run, "seeds": a.seeds, "arms": ARMS,
            "per_class": a.per_class, "epochs": a.epochs, "batch_size": 16, "train_per_class": 2,
            "student_init": "the seed's warm-up checkpoint of the teacher run",
            "pseudo_labels": "teacher primary readout on every unlabeled training image (validation identities "
                             "excluded); keep the per_class most confident images per predicted class",
            "context_pool": "tacs_pl selects from every training-split image (labels never read), identity-masked",
            "flags": {"anchored_tacs": {"aux_query_weight": 1.0, "abstain_gate": True}, "tau": .1,
                      "lambda_policy": 1.0},
            "checkpoint_selection": "best validation accuracy of each arm's readout, earliest epoch on ties",
            "primary_endpoint": "official test: tacs_pl forced_context minus nc_pl query_only, mean over seeds, "
                                "95% paired bootstrap over test queries (seed-averaged correctness)",
            "secondary": ["tacs_pl minus teacher TACS-X (unlabeled pool, same seeds)",
                          "nc_pl minus teacher No-Context", "tacs_pl query_only readout", "per-seed differences"],
            "disclosure": ["third use of the official test in this project (new protocol, registered before training)",
                           "settings chosen on validation with dev seed 42 only; seeds 10-13 reuse the teacher run",
                           "true labels of unlabeled images are never read for training or selection"],
            "smoke": {"max_steps": a.max_steps, "eval_limit": a.eval_limit} if a.max_steps or a.eval_limit else None}


def paths(a):
    root = HERE.parents[1]
    data = Path(a.data_root)
    data = data / "CUB_200_2011" if (data / "CUB_200_2011").is_dir() else data
    return data, Path(a.weights), Path(a.dinov3_repo), root


def register(a, out, teacher):
    path = out / "registration.json"
    plan = make_plan(a)
    lock = json.loads((teacher / "test/test_lock.json").read_text())
    teachers = {}
    for s in a.seeds:
        for arm in ("warmup", "no_context", "anchored_tacs_unlabeled"):
            ckpt = teacher / f"seed_{s}" / arm / "checkpoint_best.pt"
            digest = R.digest(ckpt)
            if arm != "warmup":
                assert digest == lock["checkpoints"][f"{s}/{arm}"]["sha256"], f"teacher changed: {ckpt}"
            teachers[f"{s}/{arm}"] = digest
    if path.exists():
        old = json.loads(path.read_text())
        if old["plan"] != plan or old["teacher_sha256"] != teachers:
            raise ValueError("Arguments or teachers differ from the registered protocol")
        return old
    record = {"registered_utc": datetime.now(timezone.utc).isoformat(), "plan": plan, "teacher_sha256": teachers,
              "source_sha256": {p.name: R.digest(p) for p in (HERE, HERE.parent / "run_cub_anchored.py")}}
    R.dump(path, record)
    print(f"REGISTERED {path}", flush=True)
    return record


def load(arm, seed, weights, repo, state_path):
    warm = R.build_warm_model(weights, repo, seed)
    model = R.make_arm(arm, warm, seed)
    if state_path is not None:
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False)["model"])
    return model, warm


@torch.no_grad()
def teacher_probs(model, pixels, pool_ids, ident, positions):
    model.eval()
    is_nc = model.selector is None
    feats = None if is_nc else R.candidate_features(model, pixels)
    out = []
    for chunk in positions.split(64):
        q = pixels[chunk]
        with torch.autocast("cuda", dtype=torch.float16):
            if is_nc:
                logits = model.task_model(q)
            else:
                s = R.utility(model, q, feats, pool_ids[chunk].cpu(), pool_ids, ident)
                logits = model.task_model(q, pixels[s.argmax(-1)])
        out.append(logits.float().softmax(-1))
    return torch.cat(out)


def run_stage(a, out, teacher):
    data, weights, repo, _ = paths(a)
    register(a, out, teacher)
    original_open, audit = R.protect_test_images(data)
    try:
        train, val, _, pools, ident, split = R.prepare(data, 16, 0, train_per_class=2, unlabeled=True)
        R.dump(out / "split_audit.json", {k: v for k, v in split.items() if not isinstance(v, list)})
        pixels, pool_ids = pools["unlabeled"]
        labeled = list(train.dataset.query_ids)
        unl_pos = torch.tensor([j for j, i in enumerate(pool_ids.tolist()) if i not in set(labeled)]).cuda()
        subset = train.dataset.base
        base_images = subset.dataset
        clean = train.dataset.candidate_base
        R.ARM_FLAGS["anchored_tacs"] = make_plan(a)["flags"]["anchored_tacs"]
        for seed in a.seeds:
            for name, spec in ARMS.items():
                folder = out / f"seed_{seed}" / name
                if (folder / "final.json").exists():
                    print(f"seed {seed} {name} already complete", flush=True)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                t_model, _ = load(spec["arm"], seed, weights, repo,
                                  teacher / f"seed_{seed}" / spec["teacher"] / "checkpoint_best.pt")
                probs = teacher_probs(t_model, pixels, pool_ids, ident, unl_pos)
                del t_model
                torch.cuda.empty_cache()
                conf, pred = probs.max(-1)
                take = torch.zeros_like(conf, dtype=torch.bool)
                for c in pred.unique():
                    idx = (pred == c).nonzero().squeeze(1)
                    take[idx[conf[idx].argsort(descending=True)[:a.per_class]]] = True
                if (len(labeled) + int(take.sum())) % 16 == 1:
                    # amendment 1: a final batch of one image cannot train the gate's BatchNorm;
                    # drop the least confident pseudo-label (a no-op for every other split size)
                    sel = take.nonzero().squeeze(1)
                    take[sel[conf[sel].argmin()]] = False
                pseudo_ids = pool_ids[unl_pos[take]].tolist()
                pseudo_labels = [int(y) for y in pred[take].tolist()]
                assert not set(pseudo_ids) & set(val.dataset.query_ids) and not set(pseudo_ids) & set(labeled)
                images = copy.copy(base_images)
                images.targets = list(base_images.targets)
                for i, y in zip(pseudo_ids, pseudo_labels):
                    images.targets[i] = y
                subset.dataset = images
                train.dataset.query_ids = labeled + pseudo_ids
                subset.indices = train.dataset.query_ids
                train.dataset.seed = seed
                precision = float(np.mean([y == clean.targets[i] for i, y in zip(pseudo_ids, pseudo_labels)]))
                R.dump(folder / "pseudo_labels.json", {"count": len(pseudo_ids), "classes": len(set(pseudo_labels)),
                                                       "precision_diagnostic_only": precision,
                                                       "ids": pseudo_ids, "labels": pseudo_labels})
                print(f"seed {seed} {name}: {len(pseudo_ids)} pseudo-labels, precision {precision:.3f} "
                      f"(diagnostic)", flush=True)
                warm =R.build_warm_model(weights, repo, seed)
                warm.load_state_dict(torch.load(teacher / f"seed_{seed}" / "warmup" / "checkpoint_best.pt",
                                                map_location="cpu", weights_only=False)["model"])
                student = R.make_arm(spec["arm"], warm, seed)
                args = argparse.Namespace(seeds=a.seeds, arms=["no_context", "anchored_tacs"], warmup_epochs=5,
                                          epochs=a.epochs, batch_size=16, train_per_class=2, max_steps=a.max_steps,
                                          eval_limit=a.eval_limit, output_path=out)
                R.STATUS_NAME = "status.json"
                R.run_phase(spec["arm"], student, a.epochs, train, val, pixels, pool_ids, ident, seed, 5, args, folder)
                del student, warm
                torch.cuda.empty_cache()
                subset.dataset = base_images
                train.dataset.query_ids = labeled
                subset.indices = labeled
        R.dump(out / "image_access_audit.json", audit)
        R.status(out, state="complete", stage="run", test_pixels_read=False, **audit)
    finally:
        Image.open = original_open


def test_stage(a, out, teacher):
    data, weights, repo, _ = paths(a)
    record = register(a, out, teacher)
    test_dir = out / "test"
    if (test_dir / "test_lock.json").exists() or (test_dir / "test_result.json").exists():
        raise RuntimeError("Official test already evaluated or started for this run; it is evaluated once only")
    ckpts = {}
    for seed in a.seeds:
        for name in ARMS:
            final = out / f"seed_{seed}" / name / "final.json"
            info = json.loads(final.read_text())
            ckpt = final.parent / "checkpoint_best.pt"
            assert R.digest(ckpt) == info["checkpoint_sha256"], f"checkpoint changed: {ckpt}"
            ckpts[f"{seed}/{name}"] = {"sha256": info["checkpoint_sha256"], "validation_epoch": info["best"]["epoch"],
                                       "validation_accuracy": info["best"]["accuracy"]}
    smoke = record["plan"]["smoke"] is not None
    R.dump(test_dir / "test_lock.json", {"locked_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": ckpts,
                                         "primary_endpoint": record["plan"]["primary_endpoint"]})
    if smoke:
        print("SMOKE: validation used in place of test", flush=True)
        original_open, _ = R.protect_test_images(data)
        try:
            _, test, _, pools, ident, split = R.prepare(data, 16, 0, train_per_class=2, unlabeled=True)
        finally:
            Image.open = original_open
        split["test_identical_to_training"] = []
    else:
        _, _, test, pools, ident, split = R.prepare(data, 16, 0, include_test=True, train_per_class=2, unlabeled=True)
    R.dump(test_dir / "test_split_audit.json", {k: v for k, v in split.items() if not isinstance(v, list)})
    pixels, pool_ids = pools["unlabeled"]
    correct, query_ids = {}, None
    for seed in a.seeds:
        for name, spec in ARMS.items():
            model, _ = load(spec["arm"], seed, weights, repo, out / f"seed_{seed}" / name / "checkpoint_best.pt")
            readouts, preds = R.evaluate(model, test, pixels, pool_ids, ident, a.eval_limit if smoke else 0)
            torch.save(preds, test_dir / f"predictions_seed{seed}_{name}.pt")
            query_ids = preds["query_id"] if query_ids is None else query_ids
            assert torch.equal(query_ids, preds["query_id"])
            for r in ("query_only", "forced_context"):
                if f"{r}_correct" in preds:
                    correct.setdefault(f"{name}:{r}", []).append(preds[f"{r}_correct"].numpy())
            print(f"TEST seed {seed} {name} {json.dumps(readouts)}", flush=True)
            del model
            torch.cuda.empty_cache()
    if not smoke:  # teacher-run predictions on the same queries, for the secondary comparisons
        for seed in a.seeds:
            for arm, r in (("no_context", "query_only"), ("anchored_tacs_unlabeled", "forced_context")):
                p = torch.load(teacher / "test" / f"predictions_seed{seed}_{arm}.pt", weights_only=False)
                assert torch.equal(p["query_id"], query_ids)
                correct.setdefault(f"teacher_{arm}:{r}", []).append(p[f"{r}_correct"].numpy())
    mean = {k: np.mean(v, axis=0) for k, v in correct.items()}
    per_seed = {k: [float(x.mean()) for x in v] for k, v in correct.items()}

    def compare(a_key, b_key):
        res = R.paired_bootstrap(mean[a_key], mean[b_key])
        d = np.array([(b - x) * 100 for x, b in zip(per_seed[a_key], per_seed[b_key])])
        half = 3.182 * d.std(ddof=1) / 2 if len(d) == 4 else float("nan")
        return {**res, "per_seed_pp": d.tolist(), "seed_t95_pp": [float(d.mean() - half), float(d.mean() + half)]}

    result = {"scope": "SMOKE: validation used in place of test" if smoke else
              "official CUB test, evaluated once on validation-selected checkpoints",
              "seeds": a.seeds, "queries": int(len(query_ids)),
              "accuracy_seed_mean": {k: float(v.mean()) for k, v in mean.items()}, "accuracy_per_seed": per_seed,
              "primary": compare("nc_pl:query_only", "tacs_pl:forced_context"), "secondary": {}}
    if not smoke:
        result["secondary"] = {
            "tacs_pl_vs_teacher_tacs": compare("teacher_anchored_tacs_unlabeled:forced_context", "tacs_pl:forced_context"),
            "nc_pl_vs_teacher_nc": compare("teacher_no_context:query_only", "nc_pl:query_only"),
            "tacs_pl_vs_teacher_nc": compare("teacher_no_context:query_only", "tacs_pl:forced_context"),
            "tacs_pl_query_only_vs_nc_pl": compare("nc_pl:query_only", "tacs_pl:query_only")}
    result["primary_win"] = bool(result["primary"]["ci95_pp"][0] > 0)
    R.dump(test_dir / "test_result.json", result)
    print("TEST RESULT " + json.dumps({"primary": result["primary"], "acc": result["accuracy_seed_mean"]}), flush=True)


def main():
    root = HERE.parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("register", "run", "test"), default="run")
    p.add_argument("--output", default="outputs/cub200_selftrain_v1")
    p.add_argument("--teacher-run", default="outputs/cub200_unlabeled_v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13])
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--epochs", type=int, default=21)
    p.add_argument("--data-root", default=str(root / "data/CUB_200_2011"))
    p.add_argument("--weights", default=str(root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    p.add_argument("--dinov3-repo", default=str(root / "third_party/dinov3"))
    p.add_argument("--max-steps", type=int, default=0, help="smoke only")
    p.add_argument("--eval-limit", type=int, default=0, help="smoke only")
    a = p.parse_args()
    out, teacher = Path(a.output), Path(a.teacher_run)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if a.stage == "register":
        register(a, out, teacher)
    elif a.stage == "run":
        started = time.perf_counter()
        run_stage(a, out, teacher)
        print(f"RUN DONE {(time.perf_counter() - started) / 3600:.2f} h", flush=True)
    else:
        test_stage(a, out, teacher)


if __name__ == "__main__":
    main()
