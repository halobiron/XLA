"""Independent verification of the confirmatory CUB K=2 Kaggle run (no GPU, no model).

Recomputes every reported number from raw artifacts instead of trusting
test_result.json: checkpoint hashes vs the pre-test lock, labels from the
official CUB metadata, per-seed/seed-mean accuracy, paired bootstrap and
seed-level t interval, fixed/broken counts, and retrieval statistics of the
selected contexts (spec section 4). Writes verification.json, comparison.csv
and comparison.json next to the run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(a, b, draws=10000, seed=42):
    d = (np.asarray(b, float) - np.asarray(a, float)) * 100
    rng = np.random.default_rng(seed)
    boot = np.concatenate([d[rng.integers(len(d), size=(500, len(d)))].mean(1) for _ in range(draws // 500)])
    return float(d.mean()), [float(x) for x in np.quantile(boot, [.025, .975])]


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(root / "outputs/cub200_lowdata2_v1"))
    p.add_argument("--data-root", default=str(root / "data/CUB_200_2011"))
    args = p.parse_args()
    run, data = Path(args.run), Path(args.data_root)
    checks, notes = {}, []

    reg = json.loads((run / "registration.json").read_text())
    plan = reg["plan"]
    lock = json.loads((run / "test" / "test_lock.json").read_text())
    reported = json.loads((run / "test" / "test_result.json").read_text())
    seeds, arms = plan["seeds"], plan["arms"]
    checks["plan_matches_preregistered_regime"] = (plan.get("train_per_class") == 2 and seeds == [0, 1, 2, 3]
                                                   and plan["epochs"] == 75 and arms == ["no_context", "anchored_tacs"])

    # 1. checkpoint integrity: final.json, test lock, and file bytes agree
    hashes_ok = True
    for key, info in lock["checkpoints"].items():
        actual = sha(run / info["path"])
        final = json.loads((run / Path(info["path"]).parent / "final.json").read_text())
        hashes_ok &= actual == info["sha256"] == final["checkpoint_sha256"]
    checks["checkpoint_hashes_match_test_lock"] = bool(hashes_ok)

    # 2. execution source recorded at registration is intact
    exe = run / "execution"
    checks["registered_source_intact"] = all(sha(exe / rel) == h for rel, h in reg["source_sha256"].items())
    local = root / "scripts" / "run_cub_anchored.py"
    checks["local_script_identical_to_executed"] = sha(local) == sha(exe / "run_cub_anchored.py")

    # 3. no test access during training
    audits = [json.loads(f.read_text()) for f in run.glob("image_access_audit*.json")]
    statuses = [json.loads(f.read_text()) for f in run.glob("status*.json")]
    checks["no_test_image_access_during_training"] = (bool(audits) and all(a["forbidden_image_attempts"] == 0 for a in audits)
                                                      and all(s.get("test_pixels_read") is False for s in statuses))

    # 4. official labels and split
    paths = dict(l.split(maxsplit=1) for l in (data / "images.txt").read_text().splitlines())
    labels = {int(i) - 1: int(c) - 1 for i, c in (l.split() for l in (data / "image_class_labels.txt").read_text().splitlines())}
    is_train = {int(i) - 1: int(f) for i, f in (l.split() for l in (data / "train_test_split.txt").read_text().splitlines())}
    test_ids = sorted(i for i, f in is_train.items() if not f)
    split = json.loads((run / "split_audit.json").read_text())

    correct, contexts, qids = {}, {}, None
    for seed in seeds:
        for arm in arms:
            pr = torch.load(run / "test" / f"predictions_seed{seed}_{arm}.pt", weights_only=False)
            q = pr["query_id"].tolist()
            if qids is None:
                qids = q
            assert q == qids, "query order differs between files"
            y = np.array([labels[i] for i in q])
            assert np.array_equal(pr["label"].numpy(), y), "stored labels differ from official metadata"
            for readout in ("query_only", "forced_context", "gated"):
                if f"{readout}_logits" in pr:
                    c = pr[f"{readout}_logits"].argmax(-1).numpy() == y
                    assert np.array_equal(c, pr[f"{readout}_correct"].numpy())
                    if arm == "no_context" and readout != "query_only":
                        continue
                    correct.setdefault(f"{arm}:{readout}", []).append(c)
            if "context_id" in pr:
                contexts[seed] = pr["context_id"].numpy()
    checks["test_queries_are_exactly_official_test_split"] = sorted(qids) == test_ids and len(qids) == 5794
    checks["labels_from_official_metadata"] = True

    # 5. selected contexts: training split only, retrieval stats
    y =np.array([labels[i] for i in qids])
    retrieval = {}
    for seed, ctx in contexts.items():
        ctx_labels = np.array([labels[int(i)] for i in ctx])
        retrieval[seed] = {"contexts_from_training_split": bool(all(is_train[int(i)] for i in ctx)),
                           "same_class_rate": float((ctx_labels == y).mean()),
                           "cross_class_rate": float((ctx_labels != y).mean()),
                           "distinct_contexts_used": int(len(set(ctx.tolist())))}
    checks["contexts_only_from_training_split"] = all(r["contexts_from_training_split"] for r in retrieval.values())

    # 6. recompute the statistics
    nc = np.mean(correct["no_context:query_only"], axis=0)
    rows, comparisons = [], {}
    for key, vals in correct.items():
        per_seed = [float(v.mean()) for v in vals]
        row = {"model": key, "accuracy_seed_mean": float(np.mean(per_seed)),
               **{f"seed{s}": a for s, a in zip(seeds, per_seed)}}
        if key != "no_context:query_only":
            diff, ci = bootstrap(nc, np.mean(vals, axis=0))
            d_seed = (np.array(per_seed) - np.array([v.mean() for v in correct["no_context:query_only"]])) * 100
            se = d_seed.std(ddof=1) / np.sqrt(len(d_seed))
            t = T975[len(d_seed) - 1]
            fixed = int(np.sum([(v & ~n).sum() for v, n in zip(vals, correct["no_context:query_only"])]))
            broken = int(np.sum([(n & ~v).sum() for v, n in zip(vals, correct["no_context:query_only"])]))
            comparisons[key] = {"difference_pp": diff, "query_bootstrap_ci95_pp": ci,
                                "per_seed_difference_pp": [float(x) for x in d_seed],
                                "seed_t_ci95_pp": [float(d_seed.mean() - t * se), float(d_seed.mean() + t * se)],
                                "fixed_total_over_seeds": fixed, "broken_total_over_seeds": broken}
            row.update(difference_pp=diff, ci95_low=ci[0], ci95_high=ci[1])
        rows.append(row)
    primary = comparisons["anchored_tacs:forced_context"]
    checks["recomputed_primary_matches_reported"] = (abs(primary["difference_pp"] - reported["primary"]["difference_pp"]) < 1e-9
                                                     and np.allclose(primary["query_bootstrap_ci95_pp"], reported["primary"]["ci95_pp"]))
    checks["primary_win_query_ci"] = primary["query_bootstrap_ci95_pp"][0] > 0
    checks["primary_win_every_seed"] = all(x > 0 for x in primary["per_seed_difference_pp"])
    checks["primary_win_seed_t_ci"] = primary["seed_t_ci95_pp"][0] > 0
    ctx_gain = (np.mean(correct["anchored_tacs:forced_context"], 0) - np.mean(correct["anchored_tacs:query_only"], 0))
    decomposition = {"anchored_query_only_minus_no_context_pp": comparisons["anchored_tacs:query_only"]["difference_pp"],
                     "context_over_own_query_path_pp": float(ctx_gain.mean() * 100),
                     "context_over_own_query_path_ci95_pp": bootstrap(np.mean(correct["anchored_tacs:query_only"], 0),
                                                                      np.mean(correct["anchored_tacs:forced_context"], 0))[1]}
    if not checks["local_script_identical_to_executed"]:
        notes.append("scripts/run_cub_anchored.py differs from the executed copy; the executed copy in execution/ is authoritative.")

    result = {"run": str(run), "all_checks_pass": all(checks.values()), "checks": checks, "notes": notes,
              "split": {k: (len(v) if isinstance(v, list) else v) for k, v in split.items()},
              "comparisons": comparisons, "mechanism_decomposition": decomposition, "retrieval": retrieval}
    (run / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    with (run / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (run / "comparison.json").write_text(json.dumps({"rows": rows, "comparisons": comparisons}, indent=2), encoding="utf8")
    print(json.dumps({"checks": checks, "notes": notes, "primary": primary, "decomposition": decomposition,
                      "retrieval": retrieval}, indent=1))


if __name__ == "__main__":
    main()
