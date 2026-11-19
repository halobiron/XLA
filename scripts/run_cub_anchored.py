"""Matched multi-seed CUB comparison: No-Context vs anchored TACS-X (and optional paper TACS).

Stages
  register  write registration.json (protocol, input/source hashes) once.
  run   train every requested seed/arm, select checkpoints on validation only.
        Official-test images are blocked at the PIL level and audited.
  test  evaluate the locked validation-selected checkpoints on the official
        test split exactly once (refuses to run twice).

Protocol (fixed in registration.json before training, checked on every rerun):
  * Split/pool are fixed by data seed 42 (same validation/pool as all earlier
    CUB runs). The training seed changes head init, data order, augmentation
    and Gumbel/categorical draws; arms within a seed are paired.
  * Per seed: query-only warm-up from official DINOv3, then every arm continues
    from that identical warm-start with fresh matched AdamW + cosine schedules.
  * anchored_tacs = paper hybrid selection + trained query-only CE
    (aux weight 1) + per-query abstain gate (TACS-X extension).
  * PRIMARY endpoint: anchored_tacs "forced context" readout (always use the
    argmax context) minus No-Context, top-1 on the official test split,
    averaged over seeds; 95% paired bootstrap over test queries.
    The gated readout and paper TACS are secondary.
  * Checkpoint per arm/seed = best validation accuracy of its primary readout
    (earliest epoch on ties). No test-based selection.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve()
for candidate in (HERE.parents[1] / "src", HERE.parent / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tacsx.cli import build_model, param_groups
from tacsx.data import make_cub200_loaders
from tacsx.data.identity import heldout_without_identity_overlap
from tacsx.engine import _set_training_mode, set_seed
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier

DATA_SEED = 42
ARM_FLAGS = {"paper_tacs": {"aux_query_weight": 0.0, "abstain_gate": False},
             "anchored_tacs": {"aux_query_weight": 1.0, "abstain_gate": True},
             "anchored_tacs_unlabeled": {"aux_query_weight": 1.0, "abstain_gate": True}}
PRIMARY_READOUT = {"no_context": "query_only", "paper_tacs": "forced_context", "anchored_tacs": "forced_context",
                   "anchored_tacs_unlabeled": "forced_context"}
# Candidate pool per arm. "unlabeled": every image of the training split (labels never read),
# identity-masked against queries; only defined in the low-data regime.
ARM_POOL = {"anchored_tacs_unlabeled": "unlabeled"}


def uses_unlabeled(args):
    return any(ARM_POOL.get(a) == "unlabeled" for a in args.arms)


# ----------------------------------------------------------------------------- utilities
def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


STATUS_NAME = "status.json"


def status(output, **fields):
    dump(output / STATUS_NAME, {"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **fields})


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def paired_bootstrap(a, b, seed=42, draws=10000):
    """Mean (b-a) in pp and 95% CI; a/b are per-query correctness (possibly seed-averaged)."""
    d = (np.asarray(b, float) - np.asarray(a, float)) * 100
    rng = np.random.default_rng(seed)
    boot = np.concatenate([d[rng.integers(len(d), size=(500, len(d)))].mean(1) for _ in range(draws // 500)])
    return {"difference_pp": float(d.mean()), "ci95_pp": [float(x) for x in np.quantile(boot, [.025, .975])],
            "queries": int(len(d))}


def protect_test_images(data_root):
    """Fail on any PIL open of an official-test image (run stage)."""
    paths = {int(i): p for i, p in (l.split(maxsplit=1) for l in (data_root / "images.txt").read_text().splitlines())}
    allowed = {(data_root / "images" / paths[int(i)]).resolve() for i, flag in
               (l.split() for l in (data_root / "train_test_split.txt").read_text().splitlines()) if int(flag)}
    image_root, original = (data_root / "images").resolve(), Image.open
    audit = {"training_image_opens": 0, "forbidden_image_attempts": 0}

    def guarded(fp, *args, **kwargs):
        if isinstance(fp, (str, os.PathLike)):
            path = Path(fp).resolve()
            if path.is_relative_to(image_root):
                if path not in allowed:
                    audit["forbidden_image_attempts"] += 1
                    raise RuntimeError("Official test image access is prohibited in the run stage")
                audit["training_image_opens"] += 1
        return original(fp, *args, **kwargs)

    Image.open = guarded
    return original, audit


# ----------------------------------------------------------------------------- protocol
def make_plan(args):
    return {"protocol": "cub_anchored_v1", "data_seed": DATA_SEED, "seeds": args.seeds, "arms": args.arms,
            "warmup_epochs": args.warmup_epochs, "epochs": args.epochs, "batch_size": args.batch_size,
            "image_size": 224, "pool_ratio": .2, "validation_ratio": .1,
            "candidate_scope": "entire fixed 1080-image training pool, identity-masked",
            **({"train_per_class": args.train_per_class,
                "low_data_pool": "the K-per-class training subset itself (every labeled training image), identity-masked"}
               if args.train_per_class else {}),
            "candidate_features": "stop-gradient shared-backbone cache, refreshed every 50 steps and before validation",
            "score_scale": 1 / .07, "tau": .1, "lambda_policy": 1.0,
            "lr": {"lr_backbone": 1e-5, "lr_head": 3e-4, "lr_selector": 1e-4, "lr_gate": 1e-3},
            "weight_decay": .05, "scheduler": "cosine per phase", "amp": "fp16, TF32 off",
            "arm_flags": {a: ARM_FLAGS.get(a, {}) for a in args.arms},
            "primary_endpoint": ("official test: anchored_tacs_unlabeled forced_context minus anchored_tacs forced_context"
                                 if uses_unlabeled(args) else
                                 "official test: anchored_tacs forced_context minus no_context") +
                                ", mean over seeds, 95% paired bootstrap over test queries (seed-averaged correctness)",
            "secondary": ["anchored_tacs gated readout", "paper_tacs forced_context", "per-seed differences",
                          "identity-filtered test subset"],
            "checkpoint_selection": "best validation accuracy of each arm's primary readout, earliest epoch on ties",
            **({"unlabeled_pool": "anchored_tacs_unlabeled selects context from every training-split image "
                                  "(validation excluded by identity); training labels outside the K-per-class "
                                  "subset are never read; No-Context does not see these images",
                "primary_comparison": "anchored_tacs_unlabeled:forced_context minus anchored_tacs:forced_context",
                "key_secondary": "anchored_tacs_unlabeled:forced_context minus no_context"}
               if uses_unlabeled(args) else {}),
            "smoke": {"max_steps": args.max_steps, "eval_limit": args.eval_limit} if args.max_steps or args.eval_limit else None,
            "deviations": ["TACS-X anchored extension (aux query-only CE + abstain gate) is not in the paper",
                           f"{args.warmup_epochs}+{args.epochs} epochs instead of 100; batch {args.batch_size} instead of 64; 224px instead of 256",
                           "query-only warm-up shared by all arms of a seed",
                           "stop-gradient cached candidate features; cosine utility scaled by 1/.07; horizontal pair RoPE",
                           "lambda_policy 1.0 (AGENTS.md); paper section 4.2 says 0.5",
                           "validation carved from official training split"]}


def register(output, args, data_root, weights):
    path = output / "registration.json"
    plan = make_plan(args)
    if path.exists():
        record = json.loads(path.read_text())
        if record["plan"] != plan:
            raise ValueError("Protocol differs from registration.json; use a new --output")
        return record
    execution = output / "execution"
    execution.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE, execution / HERE.name)
    src = next(p for p in (HERE.parents[1] / "src/tacsx", HERE.parent / "src/tacsx") if p.is_dir())
    shutil.copytree(src, execution / "src/tacsx", ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    inputs = {"weights": weights, **{n: data_root / n for n in
              ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")}}
    record = {"registered_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "input_sha256": {k: digest(p) for k, p in inputs.items()},
              "source_sha256": {str(p.relative_to(execution)): digest(p) for p in execution.rglob("*.py")}}
    dump(path, record)
    log(f"REGISTERED {path}")
    return record


# ----------------------------------------------------------------------------- data
def prepare(data_root, batch_size, workers, include_test=False, train_per_class=0, unlabeled=False):
    train, val, test, pool = make_cub200_loaders(data_root, 224, .2, 1, batch_size, workers, DATA_SEED,
                                                 include_context=True)
    for loader in (train, val, test):
        loader.dataset.include_context = False  # full-pool selection; no sampled candidates
    clean = train.dataset.candidate_base
    full_train = list(train.dataset.query_ids)
    if unlabeled and not train_per_class:
        raise ValueError("the unlabeled pool is defined only for the low-data regime")
    if train_per_class:
        # Low-data regime: first K ids per class of the (data-seed-shuffled)
        # training list; the context pool is exactly this labeled subset.
        taken, subset = {}, []
        for qid in train.dataset.query_ids:
            label = clean.targets[qid]
            if taken.get(label, 0) < train_per_class:
                taken[label] = taken.get(label, 0) + 1
                subset.append(qid)
        train.dataset.query_ids = subset
        train.dataset.base.indices = subset
        pool_override = subset
    else:
        pool_override = None
    ids = (full_train if unlabeled else train.dataset.query_ids) + val.dataset.query_ids + \
        (test.dataset.query_ids if include_test else [])
    identities, groups = torch.full((len(clean),), -1, dtype=torch.long), {}
    for qid in ids:
        with Image.open(clean.root / "images" / clean.paths[qid]) as image:
            image = image.convert("RGB")
            key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        identities[qid] = groups.setdefault(key, qid)
    kept, excluded = heldout_without_identity_overlap(train.dataset.query_ids, val.dataset.query_ids, identities)
    val.dataset.query_ids = kept
    val.dataset.base.indices = kept
    pool_ids = torch.tensor(pool_override if pool_override is not None else [item.index for item in pool.items])
    assert not set(pool_ids.tolist()) & set(kept)
    pixels = torch.stack([clean[int(i)][0] for i in pool_ids]).half().cuda()
    pools = {"labeled": (pixels, pool_ids.cuda())}
    split = {"train_count": len(train.dataset), "validation_count": len(kept), "pool_count": len(pool_ids),
             "excluded_validation_duplicates": excluded}
    if unlabeled:
        val_groups = {int(identities[v]) for v in kept}
        unl = [i for i in full_train if int(identities[i]) not in val_groups]
        assert not set(unl) & set(kept) and set(train.dataset.query_ids) <= set(unl)
        pools["unlabeled"] = (torch.stack([clean[int(i)][0] for i in unl]).half().cuda(), torch.tensor(unl).cuda())
        split["unlabeled_pool_count"] = len(unl)
    if include_test:
        seen = {int(identities[i]) for i in (full_train if unlabeled else train.dataset.query_ids) + val.dataset.query_ids}
        split["test_count"] = len(test.dataset)
        split["test_identical_to_training"] = [i for i in test.dataset.query_ids if int(identities[i]) in seen]
    return train, val, test, pools, identities.cuda(), split


# ----------------------------------------------------------------------------- models
def build_warm_model(weights, repo, seed):
    cfg = {"backbone": {"weights": str(weights), "repo_dir": str(repo), "freeze": False, "pair_rope_layout": "horizontal"},
           "data": {"num_classes": 200}, "selector": {"projection": False}, "policy": {"lambda_policy": 1.}}
    set_seed(seed)
    return build_model(cfg, False, include_selector=False).cuda()


def make_arm(arm, warm, seed):
    task = copy.deepcopy(warm.task_model)
    set_seed(seed + 7)  # paired gate initialization
    if arm == "no_context":
        return TACSClassifier(None, task).cuda()
    return TACSClassifier(TACSSelector(task.backbone, projection=False), task, tau=.1, lambda_policy=1.,
                          **ARM_FLAGS[arm]).cuda()


@torch.no_grad()
def candidate_features(model, pixels):
    encoder, was = model.selector.encoder, model.selector.encoder.training
    encoder.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        feats = torch.cat([F.normalize(encoder.encode_global(p).float(), dim=-1) for p in pixels.split(64)])
    encoder.train(was)
    return feats


def utility(model, query, features, query_ids, pool_ids, identities):
    q = F.normalize(model.selector.encoder.encode_global(query).float(), dim=-1)
    scores = (q @ features.T) * (1 / .07)
    same = identities[query_ids.cuda()][:, None] == identities[pool_ids][None, :]
    return scores.masked_fill(same, -torch.inf)


def gather_from(pixels):
    return lambda m: (m.to(pixels.dtype) @ pixels.flatten(1)).reshape(len(m), *pixels.shape[1:])


# ----------------------------------------------------------------------------- loops
def train_epoch(model, loader, optimizer, scaler, pixels, pool_ids, identities, max_steps=0):
    is_nc = model.selector is None
    _set_training_mode(model)
    features = None if is_nc else candidate_features(model, pixels)
    sums, count, started = {}, 0, time.perf_counter()
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        q, y = batch["query"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True)
        if features is not None and step and step % 50 == 0:
            features = candidate_features(model, pixels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if is_nc:
                logits = model.task_model(q)
                loss = F.cross_entropy(logits.float(), y)
                parts = {"task_loss": loss}
            else:
                scores = utility(model, q, features, batch["query_id"], pool_ids, identities)
                out = model.anchored_step(q, y, scores, gather_from(pixels))
                loss = out.task_loss + model.lambda_policy * out.policy_loss
                parts = {"task_loss": out.task_loss, "policy_loss": out.policy_loss, "mean_reward": out.reward.mean(),
                         "positive_reward_rate": (out.reward > 0).float().mean()}
                if out.aux_loss is not None:
                    loss = loss + out.aux_loss; parts["aux_loss"] = out.aux_loss
                if out.gate_loss is not None:
                    loss = loss + out.gate_loss; parts["gate_loss"] = out.gate_loss
                    parts["query_only_rate"] = out.null_selected.float().mean()
                logits = out.logits
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        parts["accuracy"] = logits.argmax(-1).eq(y).float().mean()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(y)
        count += len(y)
    return {**{k: v / count for k, v in sums.items()}, "seconds": time.perf_counter() - started}


@torch.no_grad()
def evaluate(model, loader, pixels, pool_ids, identities, eval_limit=0):
    """Per-query readouts: query_only, forced_context, gated (gate arms)."""
    model.eval()
    is_nc = model.selector is None
    features = None if is_nc else candidate_features(model, pixels)
    rows = {"query_id": [], "label": [], "action": []}
    logits_store = {"query_only": [], "forced_context": [], "gated": []}
    for i, batch in enumerate(loader):
        if eval_limit and i * loader.batch_size >= eval_limit:
            break
        q, y = batch["query"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            base = model.task_model(q).float()
            logits_store["query_only"].append(base.cpu())
            if not is_nc:
                scores = utility(model, q, features, batch["query_id"], pool_ids, identities)
                action = scores.argmax(-1)
                ctx = model.task_model(q, pixels[action]).float()
                logits_store["forced_context"].append(ctx.cpu())
                rows["action"].append(pool_ids[action].cpu())
                if model.abstain_gate:
                    _, query_only = model._deploy(base, ctx, scores)
                    logits_store["gated"].append(torch.where(query_only[:, None], base, ctx).cpu())
        rows["query_id"].append(batch["query_id"]); rows["label"].append(y.cpu())
    labels = torch.cat(rows["label"])
    result = {"query_id": torch.cat(rows["query_id"]), "label": labels}
    if rows["action"]:
        result["context_id"] = torch.cat(rows["action"])
    readouts = {}
    for name, chunks in logits_store.items():
        if chunks:
            logits = torch.cat(chunks)
            result[f"{name}_logits"] = logits
            correct = logits.argmax(-1).eq(labels)
            result[f"{name}_correct"] = correct
            readouts[name] = float(correct.float().mean())
    return readouts, result


def run_phase(name, model, epochs, train, val, pixels, pool_ids, identities, seed, epoch_offset, args, folder):
    """Train `epochs` epochs; keep the best validation state of the primary readout."""
    optimizer = torch.optim.AdamW(param_groups(model, make_plan(args)["lr"]), weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    primary = PRIMARY_READOUT.get(name, "query_only")
    history, best = [], {"accuracy": -1.0}
    for epoch in range(1, epochs + 1):
        absolute = epoch_offset + epoch
        set_seed(seed + absolute * 1009)
        train.dataset.set_epoch(absolute)
        train.generator.manual_seed(seed + absolute * 101)
        status(args.output_path, state="running", seed=seed, arm=name, epoch=epoch, of=epochs)
        tm = train_epoch(model, train, optimizer, scaler, pixels, pool_ids, identities, args.max_steps)
        readouts, predictions = evaluate(model, val, pixels, pool_ids, identities, args.eval_limit)
        schedule.step()
        history.append({"epoch": epoch, "absolute_epoch": absolute, "train": tm, "validation": readouts})
        dump(folder / "history.json", history)
        log(f"seed {seed} {name} epoch {epoch}/{epochs} train_acc={tm['accuracy']:.4f} "
            f"val={json.dumps({k: round(v, 4) for k, v in readouts.items()})} ({tm['seconds']:.0f}s)")
        if readouts[primary] > best["accuracy"]:
            best = {"accuracy": readouts[primary], "epoch": epoch, "readouts": readouts}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = predictions
    torch.save({"model": best_state, "best": best, "arm": name, "seed": seed}, folder / "checkpoint_best.pt")
    torch.save(best_predictions, folder / "validation_predictions.pt")
    dump(folder / "final.json", {"best": best, "primary_readout": primary,
                                 "checkpoint_sha256": digest(folder / "checkpoint_best.pt")})
    model.load_state_dict(best_state)
    return best


def run_stage(args, data_root, weights, repo):
    output = args.output_path
    register(output, args, data_root, weights)
    original_open, audit = protect_test_images(data_root)
    started = time.perf_counter()
    try:
        status(output, state="preparing")
        train, val, _, pools, identities, split = prepare(data_root, args.batch_size, args.workers,
                                                          train_per_class=args.train_per_class,
                                                          unlabeled=uses_unlabeled(args))
        pixels, pool_ids = pools["labeled"]
        dump(output / "split_audit.json", split)
        log(f"split {json.dumps({k: v for k, v in split.items() if not isinstance(v, list)})}")
        for seed in args.train_seeds:
            seed_dir = output / f"seed_{seed}"
            train.dataset.seed = seed  # training-seed-specific augmentation, paired across arms
            warm = build_warm_model(weights, repo, seed)
            warm_path = seed_dir / "warmup" / "checkpoint_best.pt"
            if (seed_dir / "warmup" / "final.json").exists():
                warm.load_state_dict(torch.load(warm_path, map_location="cpu", weights_only=False)["model"])
                log(f"seed {seed}: reuse warm-up checkpoint")
            else:
                run_phase("warmup", warm, args.warmup_epochs, train, val, pixels, pool_ids, identities, seed, 0,
                          args, seed_dir / "warmup")
            for arm in args.arms:
                if (seed_dir / arm / "final.json").exists():
                    log(f"seed {seed}: {arm} already complete, skipping")
                    continue
                model = make_arm(arm, warm, seed)
                arm_pixels, arm_pool = pools[ARM_POOL.get(arm, "labeled")]
                run_phase(arm, model, args.epochs, train, val, arm_pixels, arm_pool, identities, seed,
                          args.warmup_epochs, args, seed_dir / arm)
                del model
                torch.cuda.empty_cache()
            del warm
            torch.cuda.empty_cache()
            summarize_validation(output, args)
        dump(output / f"image_access_audit{args.suffix}.json", audit)
        status(output, state="complete", stage="run", seeds=args.train_seeds, elapsed_hours=(time.perf_counter() - started) / 3600,
               test_pixels_read=False, **audit)
    finally:
        Image.open = original_open
        dump(output / f"image_access_audit{args.suffix}.json", audit)


def summarize_validation(output, args):
    per_seed, pooled = {}, {}
    for seed in args.seeds:
        preds = {}
        for arm in args.arms:
            path = output / f"seed_{seed}" / arm / "validation_predictions.pt"
            if path.exists():
                preds[arm] = torch.load(path, weights_only=False)
        if "no_context" not in preds:
            continue
        nc = preds["no_context"]["query_only_correct"].numpy()
        entry = {"no_context": float(nc.mean())}
        for arm, p in preds.items():
            if arm == "no_context":
                continue
            assert torch.equal(p["query_id"], preds["no_context"]["query_id"]), "unpaired validation"
            for readout in ("forced_context", "gated", "query_only"):
                if f"{readout}_correct" in p:
                    key = f"{arm}:{readout}"
                    c = p[f"{readout}_correct"].numpy()
                    entry[key] = {"accuracy": float(c.mean()), **paired_bootstrap(nc, c),
                                  "fixed": int((c & ~nc).sum()), "broken": int((nc & ~c).sum())}
                    pooled.setdefault(key, []).append(c)
        pooled.setdefault("no_context", []).append(nc)
        per_seed[seed] = entry
    summary = {"scope": "validation only (594 queries, reused); NOT the confirmatory endpoint", "per_seed": per_seed}
    if pooled.get("no_context"):
        nc_mean = np.mean(pooled["no_context"], axis=0)
        summary["seed_averaged"] = {k: paired_bootstrap(nc_mean, np.mean(v, axis=0)) for k, v in pooled.items()
                                    if k != "no_context" and len(v) == len(pooled["no_context"])}
    dump(output / f"validation_summary{getattr(args, 'suffix', '')}.json", summary)
    log(f"VALIDATION {json.dumps(summary['per_seed'], default=str)[:2000]}")


def test_stage(args, data_root, weights, repo):
    output = args.output_path
    record = json.loads((output / "registration.json").read_text())
    if record["plan"] != make_plan(args):
        raise ValueError("Arguments differ from the registered protocol")
    test_dir = output / "test"
    if (test_dir / "test_result.json").exists() or (test_dir / "test_lock.json").exists():
        raise RuntimeError("Official test already evaluated or started for this run; it is evaluated once only")
    checkpoints = {}
    for seed in args.seeds:
        for arm in args.arms:
            final = output / f"seed_{seed}" / arm / "final.json"
            if not final.exists():
                raise RuntimeError(f"Incomplete run: {final}")
            ckpt = final.parent / "checkpoint_best.pt"
            info = json.loads(final.read_text())
            assert digest(ckpt) == info["checkpoint_sha256"], f"checkpoint changed: {ckpt}"
            checkpoints[f"{seed}/{arm}"] = {"path": str(ckpt.relative_to(output)), "sha256": info["checkpoint_sha256"],
                                            "validation_epoch": info["best"]["epoch"]}
    args.suffix = ""
    summarize_validation(output, args)  # all seeds, written before any test access
    dump(test_dir / "test_lock.json", {"locked_utc": datetime.now(timezone.utc).isoformat(),
                                       "checkpoints": checkpoints, "primary_endpoint": record["plan"]["primary_endpoint"]})
    smoke = record["plan"]["smoke"] is not None
    if smoke:
        # Exercise the full test code path on VALIDATION; never reads test pixels.
        log("SMOKE protocol: test stage evaluates validation in place of the official test")
        original_open, _ = protect_test_images(data_root)
        try:
            _, test, _, pools, identities, split = prepare(data_root, args.batch_size, args.workers,
                                                           train_per_class=args.train_per_class,
                                                           unlabeled=uses_unlabeled(args))
        finally:
            Image.open = original_open
        split["test_identical_to_training"] = []
    else:
        log("TEST LOCK written; evaluating official test once")
        _, _, test, pools, identities, split = prepare(data_root, args.batch_size, args.workers, include_test=True,
                                                       train_per_class=args.train_per_class,
                                                       unlabeled=uses_unlabeled(args))
    dump(test_dir / "test_split_audit.json", {k: v for k, v in split.items()})
    identical = set(split["test_identical_to_training"])
    correct = {}
    for seed in args.seeds:
        warm = build_warm_model(weights, repo, seed)
        for arm in args.arms:
            model = make_arm(arm, warm, seed)
            model.load_state_dict(torch.load(output / f"seed_{seed}" / arm / "checkpoint_best.pt",
                                             map_location="cpu", weights_only=False)["model"])
            pixels, pool_ids = pools[ARM_POOL.get(arm, "labeled")]
            readouts, preds = evaluate(model, test, pixels, pool_ids, identities, args.eval_limit if smoke else 0)
            torch.save(preds, test_dir / f"predictions_seed{seed}_{arm}.pt")
            for name in ("query_only", "forced_context", "gated"):
                if f"{name}_correct" in preds:
                    correct.setdefault(f"{arm}:{name}", []).append(preds[f"{name}_correct"].numpy())
            query_ids = preds["query_id"].tolist()
            log(f"TEST seed {seed} {arm} {json.dumps(readouts)}")
            del model
        del warm
        torch.cuda.empty_cache()
    keep = np.array([q not in identical for q in query_ids])
    nc = np.mean(correct["no_context:query_only"], axis=0)
    result = {"scope": "SMOKE: validation used in place of test" if smoke else
              "official CUB test, evaluated once on validation-selected checkpoints",
              "seeds": args.seeds, "queries": int(len(nc)), "identity_filtered_queries": int(keep.sum()),
              "accuracy_seed_mean": {k: float(np.mean(v)) for k, v in correct.items()},
              "accuracy_per_seed": {k: [float(x.mean()) for x in v] for k, v in correct.items()}, "comparisons": {}}
    for key, values in correct.items():
        if key.startswith("no_context"):
            continue
        arm_mean = np.mean(values, axis=0)
        result["comparisons"][key] = {"all": paired_bootstrap(nc, arm_mean),
                                      "identity_filtered": paired_bootstrap(nc[keep], arm_mean[keep])}
    if "anchored_tacs_unlabeled:forced_context" in correct and "anchored_tacs:forced_context" in correct:
        ref = np.mean(correct["anchored_tacs:forced_context"], axis=0)
        new = np.mean(correct["anchored_tacs_unlabeled:forced_context"], axis=0)
        result["comparisons"]["anchored_tacs_unlabeled_vs_anchored_tacs"] = {
            "all": paired_bootstrap(ref, new), "identity_filtered": paired_bootstrap(ref[keep], new[keep])}
        primary = result["comparisons"]["anchored_tacs_unlabeled_vs_anchored_tacs"]["all"]
        result["key_secondary_vs_no_context"] = result["comparisons"]["anchored_tacs_unlabeled:forced_context"]["all"]
    else:
        primary = result["comparisons"].get("anchored_tacs:forced_context", {}).get("all")
    result["primary"] = primary
    result["primary_win"] = bool(primary and primary["ci95_pp"][0] > 0)
    dump(test_dir / "test_result.json", result)
    log(f"TEST RESULT {json.dumps({'primary': primary, 'win': result['primary_win'], 'acc': result['accuracy_seed_mean']})}")


def main():
    root = HERE.parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("register", "run", "test"), default="run",
                   help="register once before launching parallel per-GPU run processes")
    p.add_argument("--output", default="outputs/cub200_anchored_v1")
    p.add_argument("--data-root", default=str(root / "data/CUB_200_2011"))
    p.add_argument("--weights", default=str(root / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
    p.add_argument("--dinov3-repo", default=str(root / "third_party/dinov3"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=["no_context", "anchored_tacs"],
                   choices=["no_context", "paper_tacs", "anchored_tacs", "anchored_tacs_unlabeled"])
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--only-seeds", type=int, nargs="+",
                   help="train only this subset of the registered --seeds (e.g. one process per GPU)")
    p.add_argument("--train-per-class", type=int, default=0,
                   help="low-data regime: K training images per class (0 = all); pool = that subset")
    p.add_argument("--max-steps", type=int, default=0, help="smoke only: cap train steps per epoch")
    p.add_argument("--eval-limit", type=int, default=0, help="smoke only: cap evaluated queries")
    args = p.parse_args()
    if "no_context" not in args.arms:
        p.error("no_context is the reference arm and must be included")
    args.train_seeds = args.only_seeds or args.seeds
    if not set(args.train_seeds) <= set(args.seeds):
        p.error("--only-seeds must be a subset of --seeds")
    args.suffix = "" if args.only_seeds is None else "_seeds" + "-".join(map(str, args.train_seeds))
    global STATUS_NAME
    STATUS_NAME = f"status{args.suffix}.json"
    args.output_path = Path(args.output)
    args.output_path.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if (data_root / "CUB_200_2011").is_dir():
        data_root = data_root / "CUB_200_2011"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    dump(args.output_path / f"environment{args.suffix}.json", {"python": sys.executable, "torch": torch.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "argv": sys.argv})
    if args.stage == "register":
        register(args.output_path, args, data_root, Path(args.weights))
        return
    try:
        (run_stage if args.stage == "run" else test_stage)(args, data_root, Path(args.weights), Path(args.dinov3_repo))
    except Exception as error:
        status(args.output_path, state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
