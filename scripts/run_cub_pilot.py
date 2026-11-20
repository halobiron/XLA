"""Validation-only paired CUB pilot with a preregistered budget and futility gate."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get("TACS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

ARMS = ("no_context", "full_tacs")


def plan():
    return {"seed": 42, "max_epochs": 6, "batch_size": 16, "image_size": 224,
            "pool_ratio": .2, "validation_ratio": .1, "candidate_scope": "entire fixed training pool; no nearest shortlist",
            "candidate_features": "stop-gradient shared-backbone cache refreshed every50 steps and before validation",
            "query_features": "differentiable shared backbone; normalized dot product",
            "score_scale": 1 / .07, "tau": .1, "lambda_policy": 1.,
            "lr_backbone": 1e-5, "lr_head": 3e-4, "lr_selector": 1e-4, "weight_decay": .05,
            "scheduler": "cosine over6 epochs", "augmentation": "random resized crop; horizontal and vertical flips; paired per query/epoch",
            "initialization": "identical official pretrained DINOv3 task weights and random200-class head; no supervised warm-start",
            "endpoint": "clean validation top1 Full minus No-Context at epoch6 unless futility gate stops earlier",
            "futility_gate": "after each paired epoch>=3: difference<=-2pp AND descriptive95% paired CI upper<0 => stop both arms",
            "uncertainty": "10000 paired query bootstraps; descriptive one-seed pilot with repeated validation looks, not confirmatory",
            "test_pixels": "never read", "candidate_labels": "metadata only, not selection inputs",
            "deviations": ["6 epochs instead of100; batch16 instead of64;224px instead of256",
                           "candidate image encoder stop-gradient, features stale up to50 steps",
                           "cosine utility scaled by1/.07; project RoPE layout horizontal",
                           "lambda1 follows AGENTS.md; paper section4.2 specifies0.5",
                           "validation carved from official training; one seed; early futility stopping"]}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def status(output, **fields):
    dump(output / "status.json", {"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **fields})


def register(output):
    path = output / "registration.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["plan"] != plan():
            raise ValueError("Pilot plan changed; do not reuse this output")
        return
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, execution / "scripts/run_cub_pilot.py")
    shutil.copytree(ROOT / "src/tacsx", execution / "src/tacsx", ignore=shutil.ignore_patterns("__pycache__"))
    inputs = [ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    inputs += [ROOT / "data/CUB_200_2011" / name for name in
               ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]
    dump(path, {"registered_utc": datetime.now(timezone.utc).isoformat(), "plan": plan(),
                "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
                "source_sha256": {str(p.relative_to(execution)): digest(p) for p in execution.rglob("*.py")}})
    print(f"REGISTERED {path}", flush=True)


def paired_stats(no_context, full_tacs):
    differences = (np.asarray(full_tacs, dtype=float) - np.asarray(no_context, dtype=float)) * 100
    rng = np.random.default_rng(42)
    boot = np.concatenate([differences[rng.integers(len(differences), size=(250, len(differences)))].mean(1)
                           for _ in range(40)])
    return {"difference_pp": float(differences.mean()), "ci95_pp": np.quantile(boot, [.025, .975]).tolist()}


def futile(epoch, stats):
    return epoch >= 3 and stats["difference_pp"] <= -2 and stats["ci95_pp"][1] < 0


def select_context(scores, pixels):
    # One GEMM avoids allocating [batch,pool,channels,height,width].
    mask = F.gumbel_softmax(scores, tau=.1, hard=True, dim=-1)
    selected = (mask.to(pixels.dtype) @ pixels.flatten(1)).reshape(len(scores), *pixels.shape[1:])
    return selected, mask.argmax(-1)


def prepare(output):
    loaders = make_cub200_loaders(ROOT / "data/CUB_200_2011", 224, .2, 1, 16, 0, 42, include_context=True)
    train, val, _, pool = loaders
    # Query loaders do not materialize random candidates: selection uses the full pool.
    train.dataset.include_context = val.dataset.include_context = False
    clean = train.dataset.candidate_base
    all_train = train.dataset.query_ids + val.dataset.query_ids
    identities, groups = torch.full((len(clean),), -1, dtype=torch.long), {}
    hashes = []
    for qid in all_train:
        with Image.open(clean.root / "images" / clean.paths[qid]) as image:
            image = image.convert("RGB")
            key = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        identities[qid] = groups.setdefault(key, qid)
        hashes.append((qid, key))
    kept, excluded = heldout_without_identity_overlap(train.dataset.query_ids, val.dataset.query_ids, identities)
    val.dataset.query_ids = kept
    val.dataset.base.indices = kept
    pool_ids = torch.tensor([item.index for item in pool.items])
    assert not set(pool_ids.tolist()) & set(kept)
    pixels = torch.stack([clean[int(i)][0] for i in pool_ids]).half().cuda()
    dump(output / "split_audit.json", {"train_ids": train.dataset.query_ids, "validation_ids": kept,
         "pool_ids": pool_ids.tolist(), "excluded_validation_duplicates": excluded,
         "train_count": len(train.dataset), "validation_count": len(kept), "pool_count": len(pool_ids),
         "training_image_sha256": hashes, "test_pixels_read": False})
    return train, val, pixels, pool_ids.cuda(), identities.cuda()


def models():
    cfg = {"backbone": {"weights": str(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
                         "repo_dir": str(ROOT / "third_party/dinov3"), "freeze": False, "pair_rope_layout": "horizontal"},
           "data": {"num_classes": 200}, "selector": {}, "policy": {"lambda_policy": 1.}}
    set_seed(42)
    baseline = build_model(cfg, False, include_selector=False).cuda()
    task = copy.deepcopy(baseline.task_model)
    full = TACSClassifier(TACSSelector(task.backbone), task, tau=.1, lambda_policy=1.).cuda()
    assert all(torch.equal(a, b) for a, b in zip(baseline.task_model.state_dict().values(), task.state_dict().values()))
    return {"no_context": baseline, "full_tacs": full}


@torch.no_grad()
def candidate_features(model, pixels):
    encoder = model.selector.encoder
    was_training = encoder.training
    encoder.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        chunks = [F.normalize(encoder.encode_global(part).float(), dim=-1) for part in pixels.split(64)]
    encoder.train(was_training)
    return torch.cat(chunks)


def utility(model, query, features, query_ids, pool_ids, identities):
    q = F.normalize(model.selector.encoder.encode_global(query).float(), dim=-1)
    scores = (q @ features.T) * (1 / .07)
    excluded = identities[query_ids.to("cuda")][:, None] == identities[pool_ids][None, :]
    return scores.masked_fill(excluded, -torch.inf)


def forward_train(model, query, labels, scores, pixels):
    context, action = select_context(scores, pixels)
    logits = model.task_model(query, context)
    item_loss = F.cross_entropy(logits, labels, reduction="none")
    policy, reward, _ = model.policy_terms(query, pixels.unsqueeze(0).expand(len(query), -1, -1, -1, -1),
                                         labels, scores, action=action, context_loss=item_loss.detach())
    return logits, item_loss.mean(), policy, reward, action


def epoch_run(model, loader, arm, pixels, pool_ids, identities, optimizer=None, scaler=None):
    training = optimizer is not None
    _set_training_mode(model) if training else model.eval()
    started = time.perf_counter()
    features = candidate_features(model, pixels) if arm == "full_tacs" else None
    corrects, rows = [], []
    total_task = total_policy = total_reward = total_entropy = 0.
    count = 0
    for step, batch in enumerate(loader):
        q, y = batch["query"].cuda(), batch["label"].cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
            if features is not None and step and step % 50 == 0:
                features = candidate_features(model, pixels)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            if arm == "no_context":
                logits = model.task_model(q)
                task, policy = F.cross_entropy(logits, y), logits.new_zeros(())
                reward = action = scores = None
            else:
                scores = utility(model, q, features, batch["query_id"], pool_ids, identities)
                if training:
                    logits, task, policy, reward, action = forward_train(model, q, y, scores, pixels)
                else:
                    action = scores.argmax(-1)
                    logits = model.task_model(q, pixels[action])
                    task, policy, reward = F.cross_entropy(logits, y), logits.new_zeros(()), None
            loss = task + policy
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite pilot loss")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        corrects.extend(logits.argmax(-1).eq(y).detach().cpu().tolist())
        n = len(y)
        total_task += float(task.detach()) * n
        total_policy += float(policy.detach()) * n
        total_reward += float(reward.mean()) * n if reward is not None else 0
        if scores is not None:
            p = scores.detach().float().softmax(-1)
            total_entropy += float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()) * n
        if not training:
            for i, qid in enumerate(batch["query_id"].tolist()):
                row = {"query_id": qid, "query_label": int(y[i]), "prediction": int(logits[i].argmax())}
                if action is not None:
                    cid = int(pool_ids[action[i]])
                    row.update(candidate_id=cid, candidate_label=int(loader.dataset.candidate_base.targets[cid]),
                               selection_score=float(scores[i, action[i]]), reward="")
                rows.append(row)
        count += n
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader):
            print(f"{arm} {'train' if training else 'val'} {step+1}/{len(loader)} accuracy={np.mean(corrects):.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return {"accuracy": float(np.mean(corrects)), "task_loss": total_task / count, "policy_loss": total_policy / count,
            "mean_reward": total_reward / count if training and arm == "full_tacs" else None,
            "selection_entropy": total_entropy / count if arm == "full_tacs" else None,
            "elapsed_seconds": time.perf_counter() - started}, corrects, rows


def run(output, smoke):
    record = json.loads((output / "registration.json").read_text())
    for relative, expected in record["source_sha256"].items():
        assert digest(output / "execution" / relative) == expected, "Execution source changed"
    for relative, expected in record["input_sha256"].items():
        assert digest(ROOT / relative) == expected, "Registered input changed"
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(42)
    dump(output / "environment.json", {"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
                                       "gpu": torch.cuda.get_device_name(0), "precision": "fp16 AMP, TF32 disabled"})
    status(output, state="preparing")
    train, val, pixels, pool_ids, identities = prepare(output)
    arms = models()
    dump(output / "initialization_audit.json", {"task_weights_identical": True, "shared_trainable_backbone": True,
         "supervised_checkpoint_loaded": False, "pool_candidates": len(pool_ids)})
    if smoke:
        full = arms["full_tacs"]
        _set_training_mode(full)
        features = candidate_features(full, pixels)
        batch = next(iter(train))
        q, y = batch["query"].cuda(), batch["label"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            scores = utility(full, q, features, batch["query_id"], pool_ids, identities)
            _, task, policy, reward, _ = forward_train(full, q, y, scores, pixels)
        gt = torch.autograd.grad(task, scores, retain_graph=True)[0]
        gp = torch.autograd.grad(policy, scores, retain_graph=True)[0]
        (task + policy).backward()
        audit = {"task_gradient_to_scores_norm": float(gt.norm()), "policy_gradient_to_scores_norm": float(gp.norm()),
                 "finite": bool(torch.isfinite(gt).all() and torch.isfinite(gp).all()), "reward_detached": not reward.requires_grad,
                 "full_pool_size": len(pool_ids), "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3, "optimizer_steps": 0}
        assert audit["finite"] and audit["reward_detached"] and min(float(gt.norm()), float(gp.norm())) > 0
        dump(output / "gradient_audit.json", audit)
        status(output, state="smoke_passed")
        print(json.dumps(audit), flush=True)
        return
    optimizers = {a: torch.optim.AdamW(param_groups(m, plan()), weight_decay=.05) for a, m in arms.items()}
    schedules = {a: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=6) for a, o in optimizers.items()}
    scalers = {a: torch.amp.GradScaler("cuda") for a in arms}
    histories = {a: [] for a in arms}
    for epoch in range(1, 7):
        outcomes, orders = {}, {}
        for arm in ARMS:
            folder = output / arm
            folder.mkdir(exist_ok=True)
            if (folder / "checkpoint_last.pt").exists() and epoch == 1:
                raise ValueError("This fixed-budget pilot does not silently restart or resume existing training")
            set_seed(42 + epoch * 1009)
            train.dataset.set_epoch(epoch)
            train.generator.manual_seed(42 + epoch * 101)
            status(output, state="running", stage="training", epoch=epoch, arm=arm)
            tm, _, _ = epoch_run(arms[arm], train, arm, pixels, pool_ids, identities, optimizers[arm], scalers[arm])
            status(output, state="running", stage="validation", epoch=epoch, arm=arm)
            vm, outcomes[arm], pairs = epoch_run(arms[arm], val, arm, pixels, pool_ids, identities)
            orders[arm] = [(row["query_id"], row["query_label"]) for row in pairs]
            schedules[arm].step()
            row = {"epoch": epoch, "train": tm, "validation": vm}
            histories[arm].append(row)
            dump(folder / "history.json", histories[arm])
            dump(folder / "config.json", plan())
            torch.save({"model": arms[arm].state_dict(), "history": histories[arm], "config": plan(),
                        "registration_sha256": digest(output / "registration.json")}, folder / "checkpoint_last.pt")
            with (folder / f"validation_epoch_{epoch:03d}.csv").open("w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=pairs[0].keys()); writer.writeheader(); writer.writerows(pairs)
            print(f"FINISH {arm} {json.dumps(row)}", flush=True)
        assert orders["no_context"] == orders["full_tacs"], "Unpaired validation predictions"
        stats = paired_stats(outcomes["no_context"], outcomes["full_tacs"])
        result = {"epoch": epoch, "scope": "one-seed exploratory CUB pilot, clean validation only", **stats,
                  "accuracy": {a: float(np.mean(v)) for a, v in outcomes.items()}, "futility_stop": futile(epoch, stats), "test_pixels_read": False}
        dump(output / f"comparison_epoch_{epoch:03d}.json", result)
        dump(output / "comparison.json", result)
        with (output / "comparison.csv").open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=["arm", "epoch", "validation_accuracy"])
            writer.writeheader(); writer.writerows({"arm": a, "epoch": epoch, "validation_accuracy": v} for a, v in result["accuracy"].items())
        print(f"COMPARISON {json.dumps(result)}", flush=True)
        if result["futility_stop"]:
            status(output, state="stopped_futility", epoch=epoch, test_pixels_read=False)
            return
    status(output, state="complete", epoch=6, test_pixels_read=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("register", "smoke", "run"), default="register")
    p.add_argument("--output", default="outputs/cub200_fullpool_pilot_s42_v1")
    args = p.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "register":
        register(output)
        return
    if Path(__file__).resolve() != (output / "execution/scripts/run_cub_pilot.py").resolve():
        raise ValueError("Execute the registered source copy with TACS_PROJECT_ROOT set")
    try:
        run(output, args.stage == "smoke")
    except Exception as error:
        status(output, state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
