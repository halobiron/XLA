"""Train-only cached utility learning on the frozen final CUB TACS checkpoint.

An explicit TACS-X second stage, not original joint Gumbel/REINFORCE training.
Select heads/epochs on a train-split holdout before loading validation predictions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(os.environ.get("TACS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image

import run_cub_quick_check as quick
from tacsx.data.context import ContextDataset
from tacsx.models.selector import ResidualSelectorProjection
from tacsx.models.utility_selector import UtilitySelector, expected_context_loss

pilot = quick.pilot
PARENT = Path("outputs/cub200_quick_warmstart_s42_v1")
ORACLE = Path("outputs/cub200_final_oracle_s42_v1")
VARIANTS = ("cosine_metric", "utility16", "statistics16", "utility17_fallback")


def plan():
    return {"seed": 42, "parent": str(PARENT), "checkpoint": "fixed final continuation epoch4, absolute epoch9",
            "fit_queries_per_class": 12, "holdout_queries_per_class": 3, "classes": 200,
            "query_split": "stratified train-only; query identities disjoint from pool and across splits",
            "fit_views": "clean and one fixed original-training augmentation, grouped by query identity",
            "holdout_views": "clean only", "candidate_count": 16, "pool_count": 1080,
            "proposals": "frozen current TACS top16; self/pixel identities excluded; no candidate labels",
            "variants": list(VARIANTS), "epochs": 40, "batch_size": 128, "weight_decay": .001,
            "learning_rates": {"cosine_metric": .0003, "others": .001},
            "loss": "enumerated expected per-query-standardized CE; detached frozen task predictions",
            "selection": "best train-holdout accuracy, NLL tie-break; epoch0 allowed; choose one primary head before validation predictions are deserialized",
            "validation": "all594 existing clean validation queries; every fixed variant disclosed; no validation-driven follow-up",
            "controls": ["independent_no_context", "original_tacs", "internal_query_only", "two_query_only_ensemble", "max_confidence16", "min_entropy16", "mean_probability16"],
            "fallback": "utility17 may choose independent No-Context logits; uses an additional model, explicitly not pure TACS",
            "compute": "utility heads inspect16 pair predictions; cosine head ranks from image features only; additional head training is not a matched-budget original-TACS win",
            "uncertainty": "descriptive one-seed paired bootstrap95%; reused validation; multiple disclosed comparisons, no confirmatory claim",
            "frozen_task_optimizer_steps": 0, "test_access": False}


def make_splits(train_ids, pool_ids, identities, targets, fit_per_class=12, holdout_per_class=3):
    pool_identity = {int(identities[i]) for i in pool_ids}
    seen, eligible = set(pool_identity), {}
    for q in sorted(train_ids):
        identity = int(identities[q])
        if identity in seen:
            continue
        seen.add(identity)
        eligible.setdefault(int(targets[q]), []).append(q)
    rng, fit, holdout = random.Random(20261117), [], []
    for label in sorted(set(int(targets[q]) for q in train_ids)):
        ids = eligible.get(label, []).copy()
        rng.shuffle(ids)
        if len(ids) < fit_per_class + holdout_per_class:
            raise ValueError(f"Insufficient identity-safe queries for class{label}")
        holdout += ids[:holdout_per_class]
        fit += ids[holdout_per_class:holdout_per_class + fit_per_class]
    rng.shuffle(fit); rng.shuffle(holdout)
    assert not {int(identities[q]) for q in fit} & {int(identities[q]) for q in holdout}
    assert not {int(identities[q]) for q in fit + holdout} & pool_identity
    return {"fit": fit, "holdout": holdout}


class CosineMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = ResidualSelectorProjection(384)

    def forward(self, query_features, candidate_features):
        q = F.normalize(self.projection(query_features.float()), dim=-1)
        c = F.normalize(self.projection(candidate_features.float()), dim=-1)
        return torch.einsum("bd,bkd->bk", q, c) / .07


def build_head(name):
    if name == "cosine_metric":
        return CosineMetric()
    return UtilitySelector(embedding_dim=384, hidden=32 if name == "statistics16" else 64,
                           allow_null=name == "utility17_fallback", use_embeddings=name != "statistics16")


def head_forward(name, model, data, pool_features, indices):
    q, c = data["query_features"][indices], pool_features[data["positions"][indices]]
    pairs = data["pair_logits"][indices].float()
    if name == "cosine_metric":
        return model(q, c), pairs
    query = data["baseline_logits" if name == "utility17_fallback" else "internal_query_logits"][indices].float()
    return model(q, c, pairs, query, data["similarities"][indices].float())


@torch.no_grad()
def evaluate_head(name, model, data, pool_features):
    model.eval()
    logits, actions = [], []
    for indices in torch.arange(len(data["query_ids"]), device=pool_features.device).split(128):
        scores, candidates = head_forward(name, model, data, pool_features, indices)
        action = scores.argmax(-1)
        logits.append(candidates[torch.arange(len(indices), device=action.device), action].cpu())
        actions.append(action.cpu())
    logits, actions = torch.cat(logits), torch.cat(actions)
    labels = data["labels"].cpu()
    return {"accuracy": float(logits.argmax(-1).eq(labels).double().mean()),
            "nll": float(F.cross_entropy(logits, labels)),
            "fallback_fraction": float(actions.eq(0).double().mean()) if name == "utility17_fallback" else 0.}, logits, actions


def register(output):
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    source = ROOT / ORACLE
    source_record = json.loads((source / "registration.json").read_text())
    for relative, expected in source_record["source_sha256"].items():
        assert pilot.digest(source / "execution" / relative) == expected
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True)
    shutil.copytree(source / "execution/src", execution / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("run_cub_quick_check.py", "run_cub_pilot.py"):
        shutil.copy2(source / "execution/scripts" / name, execution / "scripts" / name)
    shutil.copy2(__file__, execution / "scripts" / Path(__file__).name)
    inputs = [ROOT / PARENT / "registration.json", ROOT / PARENT / "split_audit.json",
              source / "registration.json", source / "oracle_predictions.pt"]
    inputs += [ROOT / PARENT / arm / n for arm in pilot.ARMS for n in ("checkpoint_last.pt", "validation_epoch_004.csv")]
    inputs += [ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    inputs += [ROOT / "data/CUB_200_2011" / n for n in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]
    pilot.dump(output / "registration.json", {"plan": plan(),
               "input_sha256": {str(p.relative_to(ROOT)): pilot.digest(p) for p in inputs},
               "source_sha256": {str(p.relative_to(execution)): pilot.digest(p) for p in execution.rglob("*.py")}})
    print(f"REGISTERED {output}", flush=True)


def query_loader(source, ids, epoch=0):
    ds = ContextDataset(Subset(source, ids), source, include_context=False, query_ids=ids, seed=42)
    ds.set_epoch(epoch)
    return DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(42))


@torch.inference_mode()
def cache_training(output, name, ids, sources, model, baseline, pool_features, pixels, pool_ids, identities):
    target = output / f"{name}.pt"
    if target.exists():
        return
    chunks = output / "cache_chunks" / name
    chunks.mkdir(parents=True, exist_ok=True)
    lock = pilot.digest(output / "registration.json")
    for view, source in enumerate(sources):
        loader = query_loader(source, ids, epoch=101 if view else 0)
        for step, batch in enumerate(loader):
            path = chunks / f"view_{view}_batch_{step:04d}.pt"
            if path.exists():
                old = torch.load(path, map_location="cpu", weights_only=False)
                assert old["lock"] == lock and torch.equal(old["query_ids"], batch["query_id"])
                continue
            q, qids = batch["query"].cuda(), batch["query_id"].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                qf = F.normalize(model.selector.encoder.encode_global(q).float(), dim=-1)
                scores = (qf @ pool_features.T) / .07
                scores = scores.masked_fill(identities[qids][:, None] == identities[pool_ids][None], -torch.inf)
                values, positions = scores.topk(16, dim=-1)
                assert torch.isfinite(values).all()
                pairq = q[:, None].expand(-1, 16, -1, -1, -1).reshape(-1, *q.shape[1:])
                pairc = pixels[positions.flatten()]
                # Frozen inference can batch different query/candidate pairs together.
                pairs = torch.cat([model.task_model(a, b).float() for a, b in
                                   zip(pairq.split(32), pairc.split(32))]).reshape(len(q), 16, -1)
                internal = model.task_model(q).float()
                base = baseline.task_model(q).float()
            torch.save({"lock": lock, "query_ids": batch["query_id"], "labels": batch["label"],
                        "view": torch.full_like(batch["query_id"], view), "query_features": qf.cpu(),
                        "positions": positions.cpu(), "similarities": (values * .07).float().cpu(),
                        "pair_logits": pairs.cpu(), "internal_query_logits": internal.cpu(), "baseline_logits": base.cpu()}, path)
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == len(loader):
                pilot.status(output, state="running", stage="cache", split=name, view=view, batches=step+1, total_batches=len(loader))
                print(f"CACHE {name} view{view} {step+1}/{len(loader)}", flush=True)
    rows = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(chunks.glob("*.pt"))]
    assert all(r["lock"] == lock for r in rows)
    data = {key: torch.cat([r[key] for r in rows]) for key in rows[0] if key != "lock"}
    assert data["query_ids"].tolist() == ids * len(sources)
    torch.save({**data, "lock": lock, "split": name}, target)


def device_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def fit_heads(output, pool_features):
    fit, holdout = device_data(output / "fit.pt"), device_data(output / "holdout.pt")
    lock = pilot.digest(output / "registration.json")
    assert fit["lock"] == holdout["lock"] == lock
    assert not set(fit["query_ids"].tolist()) & set(holdout["query_ids"].tolist())
    choices = []
    for name in VARIANTS:
        folder = output / name
        folder.mkdir(exist_ok=True)
        pilot.set_seed(42)
        model = build_head(name).cuda()
        lr = .0003 if name == "cosine_metric" else .001
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        best, history = None, []
        for epoch in range(41):
            loss_sum = 0.
            if epoch:
                model.train()
                for indices in torch.randperm(len(fit["query_ids"]), device="cuda").split(128):
                    optimizer.zero_grad(set_to_none=True)
                    scores, logits = head_forward(name, model, fit, pool_features, indices)
                    loss = expected_context_loss(scores, logits, fit["labels"][indices])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite selector loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(indices)
                scheduler.step()
            metrics, _, _ = evaluate_head(name, model, holdout, pool_features)
            row = {"epoch": epoch, "train_loss": loss_sum / len(fit["query_ids"]) if epoch else None, "holdout": metrics}
            history.append(row)
            if best is None or (metrics["accuracy"], -metrics["nll"]) > (best["holdout"]["accuracy"], -best["holdout"]["nll"]):
                best = row
                torch.save({"model": model.state_dict(), "variant": name, "best": best,
                            "registration_sha256": pilot.digest(output / "registration.json")}, folder / "checkpoint_best.pt")
            if epoch % 10 == 0:
                print(f"FIT {name} {json.dumps(row)}", flush=True)
                pilot.status(output, state="running", stage="fit", variant=name, epoch=epoch)
        pilot.dump(folder / "history.json", history)
        choices.append({"variant": name, "best": best, "checkpoint_sha256": pilot.digest(folder / "checkpoint_best.pt")})
    winner = max(choices, key=lambda c: (c["best"]["holdout"]["accuracy"], -c["best"]["holdout"]["nll"]))
    pilot.dump(output / "selection.json", {"primary": winner, "all_choices": choices, "uses_validation": False})
    return winner


@torch.inference_mode()
def cache_validation(output, val, model, pool_features, pool_ids):
    # Called only AFTER train-only model/epoch selection has been saved and locked.
    assert (output / "selection.json").is_file()
    saved = torch.load(ROOT / ORACLE / "oracle_predictions.pt", map_location="cpu", weights_only=False)
    assert val.dataset.query_ids == saved["query_ids"].tolist()
    lookup = {int(q): i for i, q in enumerate(pool_ids.cpu())}
    positions = torch.tensor([[lookup[int(q)] for q in row] for row in saved["candidate_ids"]])
    features = []
    for batch in val:
        with torch.autocast("cuda", dtype=torch.float16):
            features.append(F.normalize(model.selector.encoder.encode_global(batch["query"].cuda()).float(), dim=-1).cpu())
    qf = torch.cat(features)
    similarities = (qf[:, None] * pool_features.cpu()[positions]).sum(-1)
    # Candidate identities and pair predictions are reused exactly from the frozen oracle.
    torch.save({"query_features": qf, "positions": positions, "similarities": similarities,
                "pair_logits": saved["pair_logits"], "internal_query_logits": saved["query_logits"],
                "baseline_logits": saved["baseline_logits"], "query_ids": saved["query_ids"], "labels": saved["labels"],
                "lock": pilot.digest(output / "registration.json"), "split": "validation"}, output / "validation.pt")


def final_report(output, winner, pool_features, pool_ids):
    data = device_data(output / "validation.pt")
    y = data["labels"].cpu()
    pairs = data["pair_logits"].cpu()
    probabilities = pairs.softmax(-1)
    conf_action = probabilities.amax(-1).argmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    row = torch.arange(len(y))
    predictions = {"independent_no_context": data["baseline_logits"].cpu(), "original_tacs": pairs[:, 0],
                   "internal_query_only": data["internal_query_logits"].cpu(),
                   "two_query_only_ensemble": ((data["baseline_logits"].cpu().softmax(-1) + data["internal_query_logits"].cpu().softmax(-1)) / 2).clamp_min(1e-12).log(),
                   "max_confidence16": pairs[row, conf_action], "min_entropy16": pairs[row, entropy.argmin(-1)],
                   "mean_probability16": probabilities.mean(1).clamp_min(1e-12).log()}
    actions = {}
    for name in VARIANTS:
        model = build_head(name).cuda()
        saved = torch.load(output / name / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        _, predictions[name], actions[name] = evaluate_head(name, model, data, pool_features)
    nc = predictions["independent_no_context"].argmax(-1).eq(y)
    metrics = []
    for name, logits in predictions.items():
        correct = logits.argmax(-1).eq(y)
        metrics.append({"method": name, "accuracy": float(correct.double().mean()), "nll": float(F.cross_entropy(logits, y)),
                        **pilot.paired_stats(nc.numpy(), correct.numpy()), "fixed": int((~nc & correct).sum()),
                        "broken": int((nc & ~correct).sum()), "primary": name == winner["variant"]})
    result = {"scope": "one-seed frozen-model TACS-X selector learning; reused validation, not original joint TACS",
              "count": len(y), "primary_selected_on_train_holdout": winner, "metrics": metrics,
              "validation_used_for_selection": False, "test_access": False,
              "inference_limitations": plan()["compute"], "uncertainty": plan()["uncertainty"]}
    pilot.dump(output / "result.json", result)
    quick.write_csv(output / "comparison.csv", metrics)
    torch.save({"logits": predictions, "actions": actions, "query_ids": data["query_ids"].cpu(), "labels": y}, output / "validation_predictions.pt")
    selected_rows = []
    for name, chosen in actions.items():
        offset = 1 if name == "utility17_fallback" else 0
        for i, a in enumerate(chosen.tolist()):
            cid = -1 if a < offset else int(pool_ids[data["positions"][i, a-offset]])
            selected_rows.append({"variant": name, "query_id": int(data["query_ids"][i]), "candidate_id": cid,
                                  "fallback": cid < 0, "prediction": int(predictions[name][i].argmax()), "label": int(y[i])})
    quick.write_csv(output / "selected_pairs.csv", selected_rows)
    print(f"RESULT {json.dumps(result)}", flush=True)


def run(output):
    record = json.loads((output / "registration.json").read_text())
    for relative, expected in record["source_sha256"].items():
        assert pilot.digest(output / "execution" / relative) == expected
    for relative, expected in record["input_sha256"].items():
        assert pilot.digest(ROOT / relative) == expected
    if (output / "result.json").exists():
        raise ValueError("Do not overwrite completed evaluation")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    pilot.set_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pilot.dump(output / "environment.json", {"python": sys.executable, "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
    original_open, image_audit = quick.protect_test_images()
    started = time.perf_counter()
    try:
        pilot.status(output, state="preparing")
        train, val, pixels, pool_ids, identities = pilot.prepare(output)
        parent_split = json.loads((ROOT / PARENT / "split_audit.json").read_text())
        current_split = json.loads((output / "split_audit.json").read_text())
        for k in ("train_ids", "validation_ids", "pool_ids", "training_image_sha256"):
            assert parent_split[k] == current_split[k]
        splits = make_splits(train.dataset.query_ids, pool_ids.cpu().tolist(), identities.cpu(), train.dataset.candidate_base.targets)
        assert not set(splits["fit"] + splits["holdout"]) & set(val.dataset.query_ids)
        pilot.dump(output / "selector_split.json", {**splits, "fit_views": 2, "pool_disjoint_by_identity": True})
        arms = pilot.models()
        for arm, model in arms.items():
            saved = torch.load(ROOT / PARENT / arm / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            assert saved["history"][-1]["absolute_epoch"] == 9
            model.load_state_dict(saved["model"])
            model.eval().requires_grad_(False)
            del saved
        pool_features = pilot.candidate_features(arms["full_tacs"], pixels)
        torch.save({"features": pool_features.cpu(), "ids": pool_ids.cpu()}, output / "pool.pt")
        clean, augmented = train.dataset.candidate_base, train.dataset.base.dataset
        cache_training(output, "fit", splits["fit"], [clean, augmented], arms["full_tacs"], arms["no_context"], pool_features, pixels, pool_ids, identities)
        cache_training(output, "holdout", splits["holdout"], [clean], arms["full_tacs"], arms["no_context"], pool_features, pixels, pool_ids, identities)
        winner = fit_heads(output, pool_features)
        selection_hash = pilot.digest(output / "selection.json")
        pilot.status(output, state="running", stage="validation", primary=winner["variant"])
        cache_validation(output, val, arms["full_tacs"], pool_features, pool_ids)
        final_report(output, winner, pool_features, pool_ids)
        assert pilot.digest(output / "selection.json") == selection_hash
        for relative, expected in record["input_sha256"].items():
            assert pilot.digest(ROOT / relative) == expected
        assert all(not p.requires_grad and p.grad is None for m in arms.values() for p in m.parameters())
        pilot.dump(output / "integrity_audit.json", {"original_inputs_unchanged": True, "task_weights_frozen": True,
                   "task_optimizer_steps": 0, "selection_locked_before_validation": True, **image_audit, "test_pixels_read": False})
        pilot.status(output, state="complete", elapsed_seconds=time.perf_counter()-started, test_pixels_read=False)
    finally:
        Image.open = original_open
        pilot.dump(output / "image_access_audit.json", image_audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("register", "run"), default="register")
    p.add_argument("--output", default="outputs/cub200_selector_learning_s42_v1")
    args = p.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "register":
        register(output)
    else:
        if Path(__file__).resolve() != output / "execution/scripts/learn_cub_selector.py":
            raise ValueError("Execute the registered source copy with TACS_PROJECT_ROOT set")
        try:
            run(output)
        except Exception as error:
            pilot.status(output, state="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
