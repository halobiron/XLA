"""Registered validation-only, matched query-noisy/context-clean TACS-X pilot.

No test reader, checkpoint search, or automatic external promotion is provided.
Run with .venv-gpu/Scripts/python.exe. The register stage snapshots its sources;
execute that snapshot for a long run immune to subsequent workspace edits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

ROOT = Path(os.environ.get("TACS_QUERY_NOISE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import probe_v3_context
probe_v3_context.ROOT = ROOT  # Also supports the registered execution snapshot.
from probe_v3_context import TrainingImages
from probe_tacs_robustness import corrupt
from search_native_context import atomic_save, json_save
from train_tacs_shortlist import build_task
from tacsx.cli import param_groups
from tacsx.data.cifar100 import _transform
from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap
from tacsx.engine import _set_training_mode, set_seed
from tacsx.models.retrieval_fusion import retrieve

SEEDS = (42, 43, 44)
ARMS = ("no_context", "full_tacs")
SIGMAS = (.08, .16)
EPOCHS = 4
T_CRIT_DF2 = 4.302652729696142


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_seed_summary(correct):
    """[seed, arm, query, severity]; seeds, not queries, are the primary units."""
    x = np.asarray(correct, dtype=np.float64)
    if x.ndim != 4 or x.shape[0:2] != (3, 2) or x.shape[3] != 2 or not x.shape[2]:
        raise ValueError("Expected three paired seeds, two arms, queries and two sigmas")
    if not np.isfinite(x).all() or not np.isin(x, [0, 1]).all():
        raise ValueError("Correctness must be binary")
    per_seed = (x[:, 1] - x[:, 0]).mean(axis=(1, 2)) * 100
    mean = float(per_seed.mean())
    half = T_CRIT_DF2 * float(per_seed.std(ddof=1)) / math.sqrt(3)
    ci = [mean - half, mean + half]
    # Descriptive crossed bootstrap: retain paired arms, the two severities,
    # and shared query identities when resampling both queries and seeds.
    per_query = (x[:, 1] - x[:, 0]).mean(axis=2) * 100
    rng = np.random.default_rng(20261115)
    draws = []
    for _ in range(100):
        qidx = rng.integers(0, x.shape[2], (100, x.shape[2]))
        sidx = rng.integers(0, 3, (100, 3))
        sampled = per_query[:, qidx].mean(axis=2).T
        draws.extend(sampled[np.arange(100)[:, None], sidx].mean(axis=1).tolist())
    return {"seed_differences_pp": per_seed.tolist(), "mean_difference_pp": mean,
            "seed_t_ci95_pp": ci, "crossed_bootstrap_ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
            "all_three_positive": bool((per_seed > 0).all()),
            "promotion_gate_passed": bool((per_seed > 0).all() and ci[0] > 0),
            "inference_scope": "three continuation seeds, conditional on fixed warm start and reused validation; exploratory"}


class NoisyQueries(Dataset):
    def __init__(self, source, ids, seed, epoch=0, sigma_index=None):
        self.source, self.ids, self.seed, self.epoch = source, ids, seed, epoch
        self.sigma_index = sigma_index
        self.training = sigma_index is None
        self.transform = _transform(224, self.training)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        qid = int(self.ids[index])
        augmentation_seed = (self.seed * 10000019 + self.epoch * 100003 + qid) % (2**63 - 1)
        # Per-image deterministic severity/augmentation, independent of arm,
        # model random draws, worker scheduling and resume boundaries.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(augmentation_seed)
            si = int(torch.randint(2, ()).item()) if self.training else self.sigma_index
            image = corrupt(self.source.images[qid], "gaussian_noise", SIGMAS[si], qid)
            query = self.transform(image)
        return {"query": query, "label": self.source.targets[qid], "query_id": qid, "sigma_index": si}


def loader(source, ids, seed, epoch, sigma_index, batch_size, workers):
    return DataLoader(NoisyQueries(source, ids, seed, epoch, sigma_index), batch_size=batch_size,
                      shuffle=sigma_index is None, num_workers=workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed + epoch * 101),
                      prefetch_factor=2 if workers else None)


def epoch_pass(model, batches, arm, pool, identities, optimizer=None, scaler=None, max_batches=None):
    training = optimizer is not None
    _set_training_mode(model) if training else model.eval()
    started = time.perf_counter()
    total = correct = 0
    loss_sum = task_sum = policy_sum = reward_sum = 0.
    probabilities, targets, query_ids, pairs = [], [], [], []
    for step, batch in enumerate(batches):
        q, y = batch["query"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True)
        qids = batch["query_id"].cuda(non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            scores = candidates = positions = action = None
            if arm == "full_tacs":
                with torch.no_grad():
                    qglobal = model.selector.encoder.encode_global(q)
                    _, positions = retrieve(qglobal.float(), pool["features"], pool["ids"], 16,
                                            qids, identities[qids], identities[pool["ids"]])
                qproj = F.normalize(model.selector.projector(qglobal), dim=-1)
                cproj = F.normalize(model.selector.projector(pool["features"][positions]), dim=-1)
                scores = torch.einsum("bd,bnd->bn", qproj, cproj) / .07
                candidates = pool["pixels"][positions]
            if training:
                out = model.run_mode(arm, q, candidates, y, scores)
                logits = out.logits
                loss = out.task_loss + model.lambda_policy * out.policy_loss
                task_sum += float(out.task_loss.detach()) * len(y)
                policy_sum += float(out.policy_loss.detach()) * len(y)
                if out.reward is not None:
                    if out.reward.requires_grad:
                        raise ValueError("Policy reward must be detached")
                    reward_sum += float(out.reward.sum())
            else:
                action = scores.argmax(-1) if scores is not None else None
                chosen = candidates[torch.arange(len(q), device=q.device), action] if candidates is not None else None
                logits = model.task_model(q, chosen)
                loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {arm} loss")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += len(y)
        loss_sum += float(loss.detach()) * len(y)
        correct += int(logits.argmax(-1).eq(y).sum())
        if not training:
            probabilities.append(logits.detach().float().softmax(-1).cpu())
            targets.append(y.cpu())
            query_ids.append(qids.cpu())
            if action is not None:
                cids = pool["ids"][positions[torch.arange(len(q), device=q.device), action]]
                for i in range(len(q)):
                    pairs.append({"query_id": int(qids[i]), "candidate_id": int(cids[i]),
                                  "sigma": SIGMAS[int(batch["sigma_index"][i])],
                                  "score": float(scores[i, action[i]]), "prediction": int(logits[i].argmax()), "label": int(y[i])})
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == len(batches):
            print(f"{arm} {'train' if training else 'validation'} {step+1}/{len(batches)} "
                  f"loss={loss_sum/total:.4f} acc={correct/total:.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if max_batches and step + 1 >= max_batches:
            break
    result = {"accuracy": correct / total, "ce": loss_sum / total, "count": total,
              "elapsed_seconds": time.perf_counter() - started}
    if training:
        result.update(task_loss=task_sum / total, policy_loss=policy_sum / total,
                      mean_reward=reward_sum / total if arm == "full_tacs" else None)
        return result
    return result, torch.cat(probabilities), torch.cat(targets), torch.cat(query_ids), pairs


def spec(args):
    return {"name": "TACS-X query Gaussian noise / clean context pilot v1",
            "seeds": list(SEEDS), "arms": list(ARMS), "epochs": EPOCHS, "batch_size": args.batch_size,
            "workers": args.workers, "lr_backbone": 1e-5, "lr_head": 3e-4, "lr_selector": 1e-4,
            "weight_decay": .05, "scheduler": "cosine T_max=4", "tau": .1, "lambda_policy": 1.,
            "shortlist_k": 16, "selector": "frozen official DINOv3 ViT-S/16; learned identity linear projection; cosine/.07",
            "downstream": "full ViT fine tuning; patch concatenation before transformer via existing adapter",
            "initialization": "both arms and all seeds share archived 5-epoch clean No-Context warm start; seeds vary continuation only",
            "pool": "fixed 9000 training images from seed42 shuffle indices[5000:14000]; no candidate labels",
            "split": "seed42 shuffle; train indices[5000:], validation indices[:5000]; deduplicate validation against train and itself",
            "noise": {"sigma": list(SIGMAS), "space": "RGB [0,1], native32, clip then round to uint8 before resize224",
                      "realization": "existing corrupt(): NumPy default_rng(20261114+query_id); same z across severities/seeds/arms",
                      "train": "uniform severity per query/epoch from paired deterministic RNG; padded crop + horizontal flip",
                      "validation": "both severities, deterministic resize, no augmentation", "contexts": "clean deterministic resize224"},
            "retrieval": "recompute full-pool DINO nearest16 using the actual noisy augmented query; no clean query feature/oracle",
            "identity": "mask original query ID and pixel identity at every retrieval, even after corruption",
            "checkpoint": "fixed final epoch4 for each arm; no validation checkpoint selection or early stopping",
            "primary_endpoint": "equal-weight mean accuracy over sigma .08 and .16, TACS minus No-Context, then mean over 3 seeds",
            "primary_ci": "two-sided 95% paired-seed Student t, df2, tcrit=4.302652729696142; conditional on validation/warm start",
            "secondary_ci": "descriptive 10000 crossed seed/query bootstrap, shared query clusters and paired severities, RNG20261115",
            "gate": "all 3 seed differences strictly >0 AND primary seed-t CI lower bound strictly >0",
            "test_access": False, "automatic_test_or_kaggle": False,
            "precision": "CUDA fp16 autocast, GradScaler; TF32 disabled for both arms",
            "deviations": ["mini reproduction, not exact TACS", "4 continuation epochs, 224px, batch32 by default",
                           "frozen selector backbone, learned projection, DINO nearest16 shortlist",
                           "common clean warm start; not three independent from-scratch fits",
                           "validation and this corruption family explored previously; this is prospective new training, not pristine confirmation",
                           "fixed noise realization per original query; statistical inference does not cover new noise draws"],
            "paths": {"data_root": "data", "archive": "outputs.zip",
                      "pool_features": "outputs/tacs_shortlist_fullft_clean_v3/train_shortlist.pt",
                      "pool_pixels": "outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt"}}


def register(args, output):
    path = output / "registration.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["specification"] != spec(args):
            raise ValueError("Protocol changed; use a new output directory")
        return saved
    if any(output.glob("seed_*")):
        raise ValueError("Cannot preregister after training")
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src", execution / "src", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for source in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(source, execution / "scripts" / source.name)
    source_hashes = {str(p.relative_to(execution)): digest(p) for p in execution.rglob("*.py")}
    paths = ["data/cifar-100-python/train", "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
             *[spec(args)["paths"][k] for k in ("pool_features", "pool_pixels")]]
    # Lock relevant archive entries without reading any archived test results.
    with zipfile.ZipFile(ROOT / "outputs.zip") as archive:
        entry = next(n for n in archive.namelist() if "cifar100_no_context_c4_e5_unf4v3/" in n and n.endswith("checkpoint_best.pt"))
        initialization = {"entry": entry, "sha256": hashlib.sha256(archive.read(entry)).hexdigest()}
    originals = {str(p.relative_to(ROOT)): digest(p) for p in
                 (ROOT / "outputs/tacs_shortlist_fullft_clean_v3").glob("*/checkpoint_best.pt")}
    saved = {"registered_utc": datetime.now(timezone.utc).isoformat(), "specification": spec(args),
             "input_sha256": {p: digest(ROOT / p) for p in paths}, "initialization": initialization,
             "original_checkpoint_sha256": originals, "source_sha256": source_hashes}
    json_save(saved, path)
    print(f"Registered {path}; no test data opened", flush=True)
    return saved


def status(output, **fields):
    json_save({"updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **fields}, output / "status.json")


def setup_data(registration, output):
    source = TrainingImages(ROOT / "data")
    if len(source) != 50000:
        raise ValueError("Unexpected training file")
    ids = list(range(50000))
    random.Random(42).shuffle(ids)
    identities = pixel_identity_ids(source.images)
    validation, excluded = heldout_without_identity_overlap(ids[5000:], ids[:5000], identities)
    cached = torch.load(ROOT / registration["specification"]["paths"]["pool_features"], map_location="cpu", weights_only=False)
    pixel = torch.load(ROOT / registration["specification"]["paths"]["pool_pixels"], map_location="cpu", weights_only=False)
    pool_ids = torch.tensor(ids[5000:14000])
    weight_hash = registration["input_sha256"]["checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    if not (torch.equal(cached["pool_ids"], pool_ids) and torch.equal(pixel["pool_ids"], pool_ids)
            and torch.equal(cached["identity_ids"], identities)
            and cached["plan"]["official_weights_sha256"] == pixel["weights_sha256"] == weight_hash):
        raise ValueError("Pool cache identity/weights mismatch")
    json_save({"train_count": 45000, "validation_count": len(validation), "pool_count": 9000,
               "excluded_validation_ids": excluded, "validation_query_ids": validation,
               "pool_train_only": set(pool_ids.tolist()).issubset(ids[5000:]),
               "test_read": False}, output / "data_audit.json")
    pool = {"ids": pool_ids.cuda(), "features": cached["pool_features"].float().cuda(), "pixels": pixel["pixels"].cuda()}
    return source, ids[5000:], validation, identities.cuda(), pool


def write_csv(rows, path):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report(output, registration):
    correct, rows, query_ids, reference_labels = [], [], None, None
    for seed in SEEDS:
        paired = []
        for arm in ARMS:
            run = output / f"seed_{seed}" / arm
            saved = torch.load(run / "validation_final.pt", map_location="cpu", weights_only=False)
            if saved["registration_sha256"] != digest(output / "registration.json") or saved["epoch"] != EPOCHS:
                raise ValueError("Result protocol mismatch")
            if query_ids is not None and not torch.equal(saved["query_ids"], query_ids):
                raise ValueError("Query order mismatch")
            if reference_labels is not None and not torch.equal(saved["labels"], reference_labels):
                raise ValueError("Query labels mismatch")
            query_ids = saved["query_ids"]
            reference_labels = saved["labels"]
            paired.append(saved["probabilities"].argmax(-1).eq(saved["labels"][:, None]).numpy())
            accuracies = paired[-1].mean(axis=0)
            rows.append({"seed": seed, "arm": arm, "epoch": EPOCHS, "accuracy_sigma_008": float(accuracies[0]),
                         "accuracy_sigma_016": float(accuracies[1]), "primary_mean_accuracy": float(accuracies.mean())})
        correct.append(paired)
    for relative, expected in registration["original_checkpoint_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Original checkpoint changed: {relative}")
    summary = paired_seed_summary(correct)
    summary.update(metrics=rows, validation_count=len(query_ids), test_access=False,
                   original_checkpoints_unchanged=True, registration_sha256=digest(output / "registration.json"))
    json_save(summary, output / "comparison.json")
    write_csv(rows, output / "comparison.csv")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(args, output, registration):
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe; CUDA required")
    for relative, expected in registration["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Registered input changed: {relative}")
    execution = output / "execution"
    for relative, expected in registration["source_sha256"].items():
        if digest(execution / relative) != expected:
            raise ValueError(f"Execution source changed: {relative}")
    # Verify this process actually runs the immutable snapshot, not live files.
    if Path(__file__).resolve() != (execution / "scripts" / Path(__file__).name).resolve():
        raise ValueError("Run the registered execution/scripts/run_query_noise_pilot.py with TACS_QUERY_NOISE_ROOT set")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status(output, state="running", stage="data_audit")
    source, train_ids, val_ids, identities, pool = setup_data(registration, output)
    plan = registration["specification"]
    lock = digest(output / "registration.json")
    json_save({"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0)}, output / "environment.json")
    archive = zipfile.ZipFile(ROOT / "outputs.zip")
    if hashlib.sha256(archive.read(registration["initialization"]["entry"])).hexdigest() != registration["initialization"]["sha256"]:
        raise ValueError("Common initialization changed")
    for seed in SEEDS:
        for arm in ARMS:
            folder = output / f"seed_{seed}" / arm
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "validation_final.pt").exists() and args.stage != "smoke":
                continue
            set_seed(seed)
            model = build_task(archive, arm, "cuda")
            optimizer = torch.optim.AdamW(param_groups(model, plan), weight_decay=plan["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            scaler = torch.amp.GradScaler("cuda")
            history = []
            last = folder / "checkpoint_last.pt"
            if last.exists() and args.stage != "smoke":
                saved = torch.load(last, map_location="cpu", weights_only=False)
                if saved["registration_sha256"] != lock or saved["seed"] != seed or saved["arm"] != arm:
                    raise ValueError("Resume mismatch")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                scaler.load_state_dict(saved["scaler"])
                history = saved["history"]
                del saved
            if args.stage == "smoke":
                # Real CUDA forward/backward audit; never optimizer-step or open
                # validation for a feasibility check before the registered run.
                _set_training_mode(model)
                batches = loader(source, train_ids[:4], seed, 0, None, 4, 0)
                batch = next(iter(batches))
                q, y = batch["query"].cuda(), batch["label"].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    scores = candidates = None
                    if arm == "full_tacs":
                        with torch.no_grad():
                            features = model.selector.encoder.encode_global(q)
                            qids = batch["query_id"].cuda()
                            _, positions = retrieve(features.float(), pool["features"], pool["ids"], 16,
                                                    qids, identities[qids], identities[pool["ids"]])
                        scores = torch.einsum("bd,bnd->bn", F.normalize(model.selector.projector(features), dim=-1),
                                              F.normalize(model.selector.projector(pool["features"][positions]), dim=-1)) / .07
                        candidates = pool["pixels"][positions]
                    out = model.run_mode(arm, q, candidates, y, scores)
                    loss = out.task_loss + out.policy_loss
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise ValueError("Nonfinite gradients in smoke audit")
                if arm == "full_tacs" and (out.reward.requires_grad or model.selector.projector.weight.grad.abs().sum() == 0):
                    raise ValueError("Selector gradient/reward audit failed")
                print(f"SMOKE {arm}: finite CUDA forward/backward, optimizer_steps=0", flush=True)
            else:
                for epoch in range(len(history), EPOCHS):
                    status(output, state="running", stage="training", seed=seed, arm=arm, epoch=epoch + 1)
                    set_seed(seed + epoch * 1009)
                    print(f"START seed={seed} {arm} epoch={epoch+1}/{EPOCHS}", flush=True)
                    metrics = epoch_pass(model, loader(source, train_ids, seed, epoch, None,
                                                       args.batch_size, args.workers), arm, pool, identities, optimizer, scaler)
                    scheduler.step()
                    history.append({"epoch": epoch + 1, "train": metrics})
                    atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                                 "registration_sha256": lock, "seed": seed, "arm": arm, "plan": plan}, last)
                    json_save(history, folder / "history.json")
                    print(f"FINISH seed={seed} {arm} epoch={epoch+1}: {json.dumps(metrics)}", flush=True)
                status(output, state="running", stage="validation", seed=seed, arm=arm, epoch=EPOCHS)
                results, probabilities, all_pairs, labels, qids = [], [], [], None, None
                for si in range(2):
                    result, probs, ys, qs, pairs = epoch_pass(model, loader(source, val_ids, seed, 0, si,
                                                                          args.batch_size, args.workers), arm, pool, identities)
                    if labels is not None and (not torch.equal(labels, ys) or not torch.equal(qids, qs)):
                        raise ValueError("Severity pairing mismatch")
                    results.append({"sigma": SIGMAS[si], **result})
                    probabilities.append(probs)
                    labels, qids = ys, qs
                    all_pairs.extend(pairs)
                write_csv(all_pairs, folder / "validation_selected_pairs.csv")
                json_save({"conditions": results, "primary_mean_accuracy": sum(r["accuracy"] for r in results) / 2,
                           "seed": seed, "arm": arm, "epoch": EPOCHS, "registration_sha256": lock}, folder / "metrics.json")
                atomic_save({"probabilities": torch.stack(probabilities, dim=1), "labels": labels, "query_ids": qids,
                             "seed": seed, "arm": arm, "epoch": EPOCHS, "registration_sha256": lock}, folder / "validation_final.pt")
            del model, optimizer, scheduler, scaler
            torch.cuda.empty_cache()
        if args.stage == "smoke":
            status(output, state="smoke_passed", stage="finite_gradients", optimizer_steps=0, test_access=False)
            return
        print(f"SEED {seed} complete for both arms", flush=True)
    summary = report(output, registration)
    status(output, state="complete", stage="comparison", gate_passed=summary["promotion_gate_passed"], test_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "smoke", "run", "report"), default="register")
    parser.add_argument("--output", default="outputs/query_noise_clean_context_3seed_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size/workers")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    try:
        registration = register(args, output) if args.stage == "register" else json.loads((output / "registration.json").read_text())
        if registration["specification"] != spec(args):
            raise ValueError("Arguments differ from the preregistered plan")
        if args.stage in ("smoke", "run"):
            run(args, output, registration)
        elif args.stage == "report":
            report(output, registration)
    except Exception as error:
        status(output, state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
