"""Bake demo/cub_data.js for the offline TACS-X CUB demo (demo/index.html).

Every number is read from saved artifacts, never typed by hand:
  * headline result: outputs/cub200_selftrain_v1 (No-Context and TACS-X after one round of
    class-balanced self-training; seeds 10-13, official test once);
  * its teachers outputs/cub200_unlabeled_v1 (No-Context, TACS-X with the labeled and the unlabeled pool);
  * earlier confirmatory run outputs/cub200_lowdata2_v1 and the validation-only regimes.
The walkthrough re-runs one seed's self-trained No-Context and TACS-X checkpoints on
a seeded random sample of official-test images per outcome group (never
hand-picked) and checks the reconstruction against the saved Kaggle
predictions before writing anything.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import run_cub_anchored as rca

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("fixed", "broken", "both", "neither")
PER_GROUP = 3
SAMPLE_SEED = 42
TOPK_CANDIDATES = 8


def jpeg(image, size, quality=82):
    if isinstance(size, int):
        size = (size, size)
    buf = io.BytesIO()
    image.convert("RGB").resize(size, Image.BICUBIC).save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def aspect_thumb(image, longest=260):
    w, h = image.size
    s = longest / max(w, h)
    return jpeg(image, (max(1, round(w * s)), max(1, round(h * s))))


def class_names(data):
    names = {}
    for line in (data / "classes.txt").read_text().splitlines():
        i, raw = line.split(maxsplit=1)
        names[int(i) - 1] = raw.split(".", 1)[1].replace("_", " ")
    return names


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def regimes():
    """Every regime explored for anchored TACS-X, in the order it was run."""
    out = ROOT / "outputs"
    rows = []

    def from_summary(run, label, scope, train):
        s = load_json(out / run / "validation_summary.json")["per_seed"]["42"]
        c = s["anchored_tacs:forced_context"]
        rows.append({"label": label, "scope": scope, "train": train, "split": "validation · 594 ảnh",
                     "seeds": "42 (dev)", "no_context": s["no_context"], "tacs": c["accuracy"],
                     "difference_pp": c["difference_pp"], "ci95_pp": c["ci95_pp"], "kind": "exploratory"})

    from_summary("cub200_anchored_dev_s42", "Đủ dữ liệu", "clean", "5.400 ảnh")
    probe = load_json(out / "cub200_anchored_dev_s42/corruption_probe/probe_result.json")
    acc = probe["accuracy"]
    rows.append({"label": "Query bị nhiễu", "scope": "Gaussian noise σ 0,08 / 0,16, context sạch", "train": "5.400 ảnh",
                 "split": "validation · 594 ảnh", "seeds": "42 (dev)",
                 "no_context": acc["no_context:query_only"]["primary_noise_mean"],
                 "tacs": acc["anchored_tacs:forced_context"]["primary_noise_mean"],
                 "difference_pp": probe["primary"]["difference_pp"], "ci95_pp": probe["primary"]["ci95_pp"],
                 "kind": "exploratory"})
    from_summary("cub200_lowdata5_dev_s42", "5 ảnh / loài", "clean", "1.000 ảnh")
    from_summary("cub200_lowdata3_dev_s42", "3 ảnh / loài", "clean", "600 ảnh")
    from_summary("cub200_lowdata2_dev_s42", "2 ảnh / loài", "clean", "400 ảnh")
    test = load_json(out / "cub200_lowdata2_v1/test/test_result.json")
    rows.append({"label": "2 ảnh / loài", "scope": "clean · xác nhận", "train": "400 ảnh",
                 "split": "official test · 5.794 ảnh", "seeds": "0, 1, 2, 3 (mới)",
                 "no_context": test["accuracy_seed_mean"]["no_context:query_only"],
                 "tacs": test["accuracy_seed_mean"]["anchored_tacs:forced_context"],
                 "difference_pp": test["primary"]["difference_pp"], "ci95_pp": test["primary"]["ci95_pp"],
                 "kind": "confirmatory"})
    return rows


TEACHER_RUN = ROOT / "outputs/cub200_unlabeled_v1"
STUDENTS = {"nc_pl": "no_context", "tacs_pl": "anchored_tacs"}  # run_cub_selftrain arm -> model arm


def seed_t95(diffs):
    """95% t interval over seeds (t_{0.975, n-1}; n=4 -> 3.182)."""
    d = np.asarray(diffs, float)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}[len(d)]
    half = tcrit * d.std(ddof=1) / math.sqrt(len(d))
    return [float(d.mean() - half), float(d.mean() + half)]


def results(run):
    test = load_json(run / "test/test_result.json")
    teacher = load_json(TEACHER_RUN / "test/test_result.json")
    seeds, per, acc = test["seeds"], test["accuracy_per_seed"], test["accuracy_seed_mean"]
    assert seeds == teacher["seeds"]
    correct, distinct, pseudo = {}, [], {k: [] for k in STUDENTS}
    for s in seeds:
        pr = torch.load(run / "test" / f"predictions_seed{s}_tacs_pl.pt", weights_only=False)
        assert not (pr["context_id"] == pr["query_id"]).any()
        distinct.append(len(set(pr["context_id"].tolist())))
        for r in ("query_only", "forced_context"):
            correct.setdefault(r, []).append(pr[f"{r}_correct"].numpy().astype(float))
        for k in STUDENTS:
            pseudo[k].append(load_json(run / f"seed_{s}" / k / "pseudo_labels.json"))
    mean = {k: np.mean(v, 0) for k, v in correct.items()}
    context = rca.paired_bootstrap(mean["query_only"], mean["forced_context"])
    nc, tx = "nc_pl:query_only", "tacs_pl:forced_context"
    diffs = [(per[tx][i] - per[nc][i]) * 100 for i in range(len(seeds))]
    ta = teacher["accuracy_seed_mean"]
    return {
        "seeds": seeds, "queries": test["queries"], "accuracy": acc,
        "ladder": {"no_context": ta["no_context:query_only"], "tacs_labeled": ta["anchored_tacs:forced_context"],
                   "tacs_unlabeled": ta["anchored_tacs_unlabeled:forced_context"], "tacs_selftrain": acc[tx],
                   "nc_selftrain": acc[nc],
                   "step_labeled": teacher["comparisons"]["anchored_tacs:forced_context"]["all"],
                   "step_unlabeled": teacher["comparisons"]["anchored_tacs_unlabeled_vs_anchored_tacs"]["all"],
                   "step_selftrain": test["secondary"]["tacs_pl_vs_teacher_tacs"],
                   "nc_step_selftrain": test["secondary"]["nc_pl_vs_teacher_nc"]},
        "perSeed": [{"seed": s, "no_context": per[nc][i], "tacs": per[tx][i], "difference_pp": diffs[i]}
                    for i, s in enumerate(seeds)],
        "primary": {**test["primary"], "positive": sum(d > 0 for d in diffs)},
        "decomposition": {"training_pp": test["secondary"]["tacs_pl_query_only_vs_nc_pl"]["difference_pp"],
                          "training_ci95_pp": test["secondary"]["tacs_pl_query_only_vs_nc_pl"]["ci95_pp"],
                          "context_pp": context["difference_pp"], "context_ci95_pp": context["ci95_pp"]},
        "distinctContexts": float(np.mean(distinct)),
        "distinctTeacherContexts": {arm: float(np.mean([len(set(torch.load(TEACHER_RUN / "test" / f"predictions_seed{s}_{arm}.pt",
                                                                          weights_only=False)["context_id"].tolist()))
                                                        for s in seeds]))
                                    for arm in ("anchored_tacs", "anchored_tacs_unlabeled")},
        "pseudo": {k: {"count": float(np.mean([p["count"] for p in v])),
                       "precision": float(np.mean([p["precision_diagnostic_only"] for p in v])), "perClass": 5}
                   for k, v in pseudo.items()},
        "poolSize": load_json(TEACHER_RUN / "split_audit.json")["unlabeled_pool_count"],
        "lockedUtc": load_json(run / "test/test_lock.json")["locked_utc"],
    }


@torch.no_grad()
def trace(nc, tacs, image_tensor, qid, label, pixels, pool_ids, identities, pool_labels, names, features):
    q = image_tensor[None].cuda()
    y = torch.tensor([label]).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        nc_logits = nc.task_model(q).float()
        base = tacs.task_model(q).float()
        scores = rca.utility(tacs, q, features, torch.tensor([qid]), pool_ids, identities).float()
        action = scores.argmax(-1)
        ctx = tacs.task_model(q, pixels[action]).float()
        tokens_q = tacs.task_model.backbone.prepare_tokens(q).tokens.shape[1]
    probs = scores.softmax(-1)[0]
    finite = torch.isfinite(scores[0])
    entropy = float(-(probs[finite] * probs[finite].clamp_min(1e-12).log()).sum())
    top = torch.topk(scores[0], TOPK_CANDIDATES)

    def topn(logits, n=5):
        p = logits.softmax(-1)[0]
        v, i = p.topk(n)
        return [{"label": names[int(c)], "p": float(x), "cls": int(c)} for x, c in zip(v, i)]

    ce = lambda lg: float(F.cross_entropy(lg, y))
    return {
        "nc_logits_argmax": int(nc_logits.argmax()), "ctx_argmax": int(ctx.argmax()),
        "context_id": int(pool_ids[action].item()), "tokens_query": int(tokens_q),
        "candidates": [{"pos": int(j), "id": int(pool_ids[j]), "label": names[pool_labels[int(j)]],
                        "same": pool_labels[int(j)] == label, "score": float(scores[0, j]), "p": float(probs[j])}
                       for j in top.indices.tolist()],
        "pick": 0, "entropy": entropy, "maxEntropy": math.log(int(finite.sum())), "poolSize": int(finite.sum()),
        "noContext": {"top": topn(nc_logits), "correct": int(nc_logits.argmax()) == label, "ce": ce(nc_logits)},
        "tacs": {"top": topn(ctx), "correct": int(ctx.argmax()) == label, "ce": ce(ctx),
                 "queryOnly": {"top": topn(base, 3), "correct": int(base.argmax()) == label, "ce": ce(base)}},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "outputs/cub200_selftrain_v1"))
    p.add_argument("--seed", type=int, default=10, help="which confirmatory seed's checkpoints to trace")
    p.add_argument("--data-root", default=str(ROOT / "data/CUB_200_2011"))
    p.add_argument("--out", default=str(ROOT / "demo/cub_data.js"))
    args = p.parse_args()
    run, data, seed = Path(args.run), Path(args.data_root), args.seed
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    names = class_names(data)
    plan = {"train_per_class": load_json(run / "registration.json")["plan"]["train_per_class"]}

    saved = {STUDENTS[k]: torch.load(run / "test" / f"predictions_seed{seed}_{k}.pt", weights_only=False)
             for k in STUDENTS}
    BEST = "anchored_tacs"
    qids = saved["no_context"]["query_id"].tolist()
    assert qids == saved[BEST]["query_id"].tolist()
    nc_ok = saved["no_context"]["query_only_correct"].numpy()
    tc_ok = saved[BEST]["forced_context_correct"].numpy()
    group_of = np.where(~nc_ok & tc_ok, 0, np.where(nc_ok & ~tc_ok, 1, np.where(nc_ok & tc_ok, 2, 3)))
    counts = {g: int((group_of == i).sum()) for i, g in enumerate(GROUPS)}
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = [(g, int(j)) for i, g in enumerate(GROUPS)
              for j in rng.choice(np.flatnonzero(group_of == i), PER_GROUP, replace=False)]

    train, _, test, pools, identities, _ = rca.prepare(data, 16, 0, include_test=True,
                                                   train_per_class=plan["train_per_class"], unlabeled=True)
    pixels, pool_ids = pools["unlabeled"]
    clean = test.dataset.candidate_base
    pool_labels = {j: clean.targets[int(i)] for j, i in enumerate(pool_ids.tolist())}
    warm = rca.build_warm_model(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                                ROOT / "third_party/dinov3", seed)
    models = {}
    for name, arm in STUDENTS.items():
        m = rca.make_arm(arm, warm, seed)
        m.load_state_dict(torch.load(run / f"seed_{seed}" / name / "checkpoint_best.pt", map_location="cpu",
                                     weights_only=False)["model"])
        models[arm] = m.eval()
    features = rca.candidate_features(models[BEST], pixels)

    images, examples, mismatches = {}, [], []
    position = {q: i for i, q in enumerate(test.dataset.query_ids)}

    def img_ref(image_id, size):
        key = f"{image_id}@{size}"
        if key not in images:
            with Image.open(clean.root / "images" / clean.paths[image_id]) as im:
                images[key] = jpeg(im, size)
        return key

    for group, j in sample:
        qid = qids[j]
        label = clean.targets[qid]
        tensor, _ = test.dataset.base[position[qid]]
        t = trace(models["no_context"], models[BEST], tensor, qid, label, pixels, pool_ids, identities,
                  pool_labels, names, features)
        expect = (int(saved["no_context"]["query_only_logits"][j].argmax()),
                  int(saved[BEST]["forced_context_logits"][j].argmax()),
                  int(saved[BEST]["context_id"][j]))
        got = (t.pop("nc_logits_argmax"), t.pop("ctx_argmax"), t["context_id"])
        if got != expect:
            mismatches.append({"query_id": qid, "saved": expect, "rebuilt": got})
        with Image.open(clean.root / "images" / clean.paths[qid]) as im:
            original = aspect_thumb(im)
            size = list(im.size)
        for c in t["candidates"]:
            c["img"] = img_ref(c["id"], 112)
        examples.append({"id": qid, "group": group, "trueLabel": names[label], "original": original,
                         "originalSize": size, "query": img_ref(qid, 224), "context": img_ref(t["context_id"], 224),
                         "contextLabel": names[clean.targets[t["context_id"]]],
                         "contextSame": clean.targets[t["context_id"]] == label, **t})
    if mismatches:
        raise SystemExit(f"Reconstruction differs from saved Kaggle predictions: {mismatches}")

    dev = load_json(ROOT / "outputs/cub200_anchored_dev_s42/validation_summary.json")["per_seed"]["42"]
    history = {"paper_tacs_full_data": {k: dev["paper_tacs:forced_context"][k] for k in ("accuracy", "difference_pp", "ci95_pp")},
               "no_context_full_data": dev["no_context"]}
    curves = {}
    for arm in ("no_context", "paper_tacs", "anchored_tacs"):
        h = load_json(ROOT / f"outputs/cub200_anchored_dev_s42/seed_42/{arm}/history.json")
        curves[arm] = [r["validation"]["query_only"] for r in h]
    history["query_only_curves_full_data"] = curves
    # Step-3 illustration: one seeded-random species, its 2 labeled images and the pseudo-labels the
    # seed's TACS-X teacher gave it (true labels are read only to mark each pseudo-label right/wrong).
    pseudo = load_json(run / f"seed_{seed}" / "tacs_pl" / "pseudo_labels.json")
    by_class = {}
    for i, y in zip(pseudo["ids"], pseudo["labels"]):
        by_class.setdefault(y, []).append(i)
    full = sorted(c for c, v in by_class.items() if len(v) == 5)
    species = int(np.random.default_rng(SAMPLE_SEED).choice(full))
    labeled_ids = [i for i in train.dataset.query_ids if clean.targets[i] == species]
    selftrain_example = {
        "species": names[species], "labeled": [img_ref(i, 112) for i in labeled_ids],
        "pseudo": [{"img": img_ref(i, 112), "correct": clean.targets[i] == species, "trueLabel": names[clean.targets[i]]}
                   for i in by_class[species]],
        "seed": seed, "classesWithFive": len(full)}

    payload = {"results": results(run), "selftrainExample": selftrain_example, "regimes": regimes(), "history": history,
               "paper": {"no_context": .821, "tacs": .852, "source": "Bảng 2, CUB, đủ dữ liệu, 100 epoch, 3 seed"},
               "walkthrough": {"seed": seed, "counts": counts, "perGroup": PER_GROUP, "sampleSeed": SAMPLE_SEED,
                               "poolSize": int(len(pool_ids)), "rebuiltMatchesSaved": f"{len(examples)}/{len(examples)}",
                               "examples": examples},
               "images": images}
    Path(args.out).write_text("window.TACSX = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                              encoding="utf8")
    size = Path(args.out).stat().st_size / 2**20
    print(f"Wrote {args.out} ({size:.2f} MiB): {len(examples)} examples, counts {counts}, "
          f"{len(images)} images, reconstruction {len(examples)}/{len(examples)} matches saved predictions")


if __name__ == "__main__":
    main()
