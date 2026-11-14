"""Matched query-only/DINO/Full-TACS continuation with a train-only shortlist.

Explicit TACS-X protocol: same query-trained initialization for every task
network; 16 nearest frozen official DINO candidates from the original fixed
20% training pool; trainable identity-initialized selector projection with
cosine logits / .07; frozen selector image encoder. Downstream still combines
patch tokens before ViT blocks. Full retains hard ST-Gumbel and REINFORCE.
No test data are read by prepare/train/validate. Test requires a locked positive
validation comparison. Every epoch saves complete resumable optimizer state.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from probe_v3_context import ROOT, TrainingImages, load_model
from evaluate_tacs_full_pool import encode
from search_native_context import atomic_save, json_save, paired_difference, metrics
from tacsx.cli import param_groups
from tacsx.data.cifar100 import _transform
from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap
from tacsx.engine import _set_training_mode, set_seed
from tacsx.models.dinov3_adapter import DINOv3Adapter
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier
from tacsx.models.retrieval_fusion import retrieve


ARMS = ("no_context", "dino_similarity", "full_tacs")


class ShortlistImages(Dataset):
    def __init__(self, source, ids, pool_source, pool_ids, positions, training, with_context, cached_pixels=False):
        self.source, self.ids, self.pool_source = source, ids, pool_source
        self.pool_ids, self.positions = pool_ids, positions
        self.with_context = with_context
        self.cached_pixels = cached_pixels
        self.transform = _transform(224, training)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        qid = self.ids[index]
        batch = {"query": self.transform(Image.fromarray(self.source.images[qid])),
                 "label": self.source.targets[qid], "local_id": index, "query_id": qid}
        if self.with_context:
            pos = self.positions[index]
            ids = self.pool_ids[pos]
            batch["candidate_positions"] = pos
            batch["candidate_ids"] = ids
            if not getattr(self, "cached_pixels", False):
                batch["candidates"] = torch.stack([self.pool_source[int(i)][0] for i in ids])
        return batch


def fingerprints(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def score(model, query, candidate_features, scale):
    # Used only for the frozen similarity baseline.
    q = F.normalize(model.selector.encoder.encode_global(query), dim=-1)
    return torch.einsum("bd,bnd->bn", q, candidate_features) * scale


def encode_candidate_globals(encoder, candidate_images, chunk_size=128):
    """Refresh current candidate embeddings without retaining encoder activations."""
    b, n, c, h, w = candidate_images.shape
    flat = candidate_images.reshape(b * n, c, h, w)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(flat), chunk_size):
            outputs.append(encoder.encode_global(flat[start:start + chunk_size]))
    return torch.cat(outputs).reshape(b, n, -1)


def learned_scores(model, query, candidate_features, scale, candidate_images=None):
    if candidate_images is None:
        with torch.no_grad():
            q = model.selector.encoder.encode_global(query)
    else:
        # In this arm the selector query path shares the trainable task ViT.
        q = model.selector.encoder.encode_global(query)
        candidate_features = encode_candidate_globals(model.selector.encoder, candidate_images)
    q = F.normalize(model.selector.projector(q), dim=-1)
    c = F.normalize(model.selector.projector(candidate_features), dim=-1)
    return torch.einsum("bd,bnd->bn", q, c) * scale


def run_epoch(model, loader, arm, device, pool_features, optimizer=None, scaler=None, scale=1 / .07, pool_pixels=None):
    training = optimizer is not None
    _set_training_mode(model) if training else model.eval()
    loss_sum = correct = total = 0
    components = torch.zeros(4, device=device, dtype=torch.float64)
    outputs, targets, pairs = [], [], []
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        q, y = batch["query"].to(device), batch["label"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            candidates, scores = None, None
            if arm != "no_context":
                positions = batch["candidate_positions"].to(device)
                candidates = pool_pixels[positions] if pool_pixels is not None else batch["candidates"].to(device)
                features = pool_features[positions]
                shared = arm == "full_tacs" and model.selector.encoder is model.task_model.backbone
                scores = learned_scores(model, q, features, scale, candidates if shared else None) if arm == "full_tacs" else score(model, q, features, scale)
            out = model.run_mode(arm, q, candidates, y, scores)
            loss = out.task_loss + model.lambda_policy * out.policy_loss
        with torch.no_grad():
            policy = out.policy_loss.detach().float() * len(y)
            reward = out.reward.detach().float().sum() if out.reward is not None else policy.new_zeros(())
            entropy = policy.new_zeros(())
            if out.scores.numel():
                probabilities = out.scores.detach().float().softmax(-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            components += torch.stack((out.task_loss.detach().float() * len(y), policy, reward, entropy))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {arm}")
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += len(y)
        loss_sum += float(loss.detach()) * len(y)
        correct += int(out.logits.argmax(-1).eq(y).sum())
        if not training:
            outputs.append(out.logits.float().softmax(-1).cpu())
            targets.append(y.cpu())
            if arm != "no_context":
                action = out.action.cpu()
                ids = batch["candidate_ids"][torch.arange(len(y)), action]
                for i, cid in enumerate(ids):
                    pairs.append({"query_id": int(batch["query_id"][i]), "candidate_id": int(cid),
                                  "selection_score": float(out.scores[i, action[i]]),
                                  "prediction": int(out.logits[i].argmax()), "label": int(y[i])})
        if training and (step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader)):
            print(f"{arm}: {step + 1}/{len(loader)} loss={loss_sum / total:.4f} acc={correct / total:.4f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    result = {"loss": loss_sum / total, "accuracy": correct / total, "elapsed_seconds": time.perf_counter() - started}
    task, policy, reward, entropy = (components / total).cpu().tolist()
    result.update(task_loss=task, policy_loss=policy,
                  mean_reward=reward if arm == "full_tacs" else None,
                  selection_entropy=entropy if arm != "no_context" else None)
    if training:
        return result, None, None, None
    probs, labels = torch.cat(outputs), torch.cat(targets)
    result.update(metrics(probs, labels))
    return result, probs, labels, pairs


def build_task(archive, arm, device, last_n=0, shared_selector_backbone=False):
    model, _, cfg = load_model(archive, "no_context", device)
    # Re-enable every downstream parameter before applying the new protocol.
    # The warm-start checkpoint itself used only the last four blocks.
    model.task_model.backbone.requires_grad_(True)
    model.task_model.head.requires_grad_(True)
    if last_n:
        from tacsx.cli import _freeze
        _freeze(model.task_model.backbone, True, last_n)
    if arm == "no_context":
        return model
    if shared_selector_backbone:
        encoder = model.task_model.backbone
    else:
        encoder = DINOv3Adapter(cfg["backbone"]["repo_dir"], weights=cfg["backbone"]["weights"]).to(device).eval().requires_grad_(False)
    selector = TACSSelector(encoder, projection=False)
    selector.projector = nn.Linear(encoder.embedding_dim, encoder.embedding_dim, bias=False).to(device)
    nn.init.eye_(selector.projector.weight)
    if arm != "full_tacs":
        selector.projector.requires_grad_(False)
    return TACSClassifier(selector, model.task_model, tau=.1, lambda_policy=1., eval_selection="argmax").to(device)


def make_loader(source, selected, pool_source, pool_ids, positions, arm, args, training, epoch=0):
    generator = torch.Generator().manual_seed(args.seed + epoch * 101)
    return DataLoader(ShortlistImages(source, selected, pool_source, pool_ids, positions,
                                     training, arm != "no_context", args.cached_pixels), batch_size=args.batch_size,
                      shuffle=training, num_workers=args.workers, pin_memory=True,
                      generator=generator, prefetch_factor=2 if args.workers else None)


def save_pairs(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "cache_pixels", "train", "validate", "test"), default="prepare")
    p.add_argument("--output", default="outputs/tacs_shortlist_fullft_clean_v3")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-root", default="data", help="directory containing cifar-100-python/train and test")
    p.add_argument("--base-archive", default="outputs.zip", help="zip containing the shared 5-epoch No-Context initialization")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--task-unfreeze-last-blocks", type=int, default=0,
                   help="0: full task fine-tuning, otherwise last N task blocks")
    p.add_argument("--shortlist-k", type=int, default=16,
                   help="number of DINO-proposed images scored per query; smaller values are practical mini-reproduction deviations")
    p.add_argument("--prepared-from", help="reuse compatible frozen-encoder shortlists; never reuse trained models")
    p.add_argument("--cached-pixels", action="store_true", help="equivalent FP16 Gumbel pixel contraction, GPU-resident clean pool")
    p.add_argument("--pixel-cache", help="reuse an immutable compatible pixel cache")
    p.add_argument("--shared-selector-backbone", action="store_true",
                   help="share the trainable task ViT with the selector; refresh candidate embeddings each step")
    args = p.parse_args()
    if args.shortlist_k < 2:
        p.error("--shortlist-k must be at least 2 for categorical selection")
    if not torch.cuda.is_available():
        p.error("CUDA required: .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)
    device = torch.device("cuda")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    archive_path = Path(args.base_archive)
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    archive = zipfile.ZipFile(archive_path)
    entry = next(x for x in archive.namelist() if "cifar100_no_context_c4_e5_unf4v3/" in x and x.endswith("checkpoint_best.pt"))
    plan = {"method": "TACS-X DINO-shortlist, query warm start", "seed": args.seed,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr_backbone": 1e-5,
            "lr_head": 3e-4, "lr_selector": 1e-4, "weight_decay": .05, "tau": .1, "lambda_policy": 1.,
            "shortlist_k": args.shortlist_k, "selector_logit_scale": 1 / .07, "task_unfreeze_last_blocks": args.task_unfreeze_last_blocks,
            "common_task_initialization_crc": archive.getinfo(entry).CRC,
            "official_weights_sha256": fingerprints(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            "candidate_pool": "first 9000 from seed42 shuffled 45000 training split; no labels used",
            "identity_rule": "exclude all pixel-identical candidates, even with different IDs; deduplicate heldout against training and itself",
            "selector": ("shared trainable task DINOv3 ViT, trainable identity-initialized linear projection, scaled cosine; candidate embeddings stop-gradient and refreshed each step"
                         if args.shared_selector_backbone else "frozen official DINO encoder, trainable identity-initialized linear projection, scaled cosine"),
            "downstream": "query/context patch concatenation before transformer, unchanged DINOv3Adapter",
            "training_augmentation": "padded random crop and horizontal flip; deterministic resize for context",
            "arms": list(args.arms), "candidate_labels_used_for_selection": False,
            "deviations": (["All task models continue same 5-epoch No-Context checkpoint", f"DINO nearest-{args.shortlist_k} shortlist", "Explicit cosine logit scale 1/.07", "224px, batch32, 10 continuation epochs, one seed", "Shortlist uses clean query; selector scores actual augmented query during training"] + ([] if args.shared_selector_backbone else ["Frozen selector encoder with trainable linear projection"])),
            "selection_metric": "validation top1 accuracy, CE tie-break", "test_tuning": False}
    plan["shared_selector_backbone"] = bool(args.shared_selector_backbone)
    if args.shared_selector_backbone:
        plan["deviations"].append("Shared selector/task ViT; candidate-image embedding path is stop-gradient, refreshed from current shared weights each step")
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Different plan; preserve run output")
    json_save(plan, plan_path)
    source = TrainingImages(data_root)
    ids = list(range(50000))
    random.Random(42).shuffle(ids)
    selected = {"train": ids[5000:], "validation": ids[:5000]}
    identity_ids = pixel_identity_ids(source.images)
    selected["validation"], excluded = heldout_without_identity_overlap(selected["train"], selected["validation"], identity_ids)
    json_save({"excluded_validation_ids": excluded, "validation_count": len(selected["validation"]),
               "rule": plan["identity_rule"]}, output / "identity_audit.json")
    pool_ids = torch.tensor(ids[5000:14000])
    if set(selected["train"]) & set(selected["validation"]):
        raise ValueError("Split overlap")
    if args.stage == "cache_pixels":
        path = output / "pool_pixels_fp16.pt"
        if path.exists():
            print("Pool pixel cache already exists", flush=True)
            return
        chunks = []
        for start in range(0, len(pool_ids), 256):
            chunks.append(torch.stack([source[int(i)][0] for i in pool_ids[start:start + 256]]).half())
            if start % 1024 == 0:
                print(f"Pool pixels: {min(start + 256, len(pool_ids))}/{len(pool_ids)}", flush=True)
        pixels = torch.cat(chunks)
        del chunks
        # A selected pixel is only consumed by the autocast einsum/patch conv.
        # Test exact contraction equivalence before allowing the cached mode.
        reference = torch.stack([source[int(i)][0] for i in pool_ids[:4]]).to(device)[None]
        masks = torch.tensor([[0., 1., 0., 0.]], device=device, requires_grad=True)
        other = masks.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            original = torch.einsum("bn,bnchw->bchw", masks, reference)
            cached = torch.einsum("bn,bnchw->bchw", other, pixels[:4].to(device)[None])
        original.float().square().mean().backward()
        cached.float().square().mean().backward()
        if not torch.equal(original, cached) or not torch.equal(masks.grad, other.grad):
            raise ValueError("FP16 pool contraction is not equivalent")
        atomic_save({"pixels": pixels, "pool_ids": pool_ids, "weights_sha256": plan["official_weights_sha256"],
                     "input_contraction_max_error": float((original - cached).detach().abs().max()),
                     "mask_gradient_max_error": float((masks.grad - other.grad).abs().max())}, path)
        print(f"Cached {len(pixels)} images; forward and mask-gradient max error 0.0", flush=True)
        return
    if args.stage == "prepare":
        if args.prepared_from:
            for split, query_ids in selected.items():
                cached = torch.load(ROOT / args.prepared_from / f"{split}_shortlist.pt", weights_only=False)
                # Cached global features are independent of shortlist K; rebuild
                # neighbors at the active K from those same label-free vectors.
                required = ("official_weights_sha256", "candidate_pool")
                if any(cached["plan"][key] != plan[key] for key in required) or not set(query_ids).issubset(cached["query_ids"]) or not torch.equal(cached["pool_ids"], pool_ids):
                    raise ValueError("Prepared frozen features/shortlist incompatible")
                raw = torch.load(ROOT / args.prepared_from / f"{split}_features.pt", weights_only=False)
                if raw["ids"].tolist() != cached["query_ids"]:
                    raise ValueError("Frozen query feature identity mismatch")
                lookup = {qid: index for index, qid in enumerate(cached["query_ids"])}
                features = raw["features"][[lookup[qid] for qid in query_ids]].to(device)
                pool = cached["pool_features"].to(device)
                nearest, similarities = [], []
                for start in range(0, len(features), 512):
                    batch_ids = torch.tensor(query_ids[start:start + 512])
                    values, positions = retrieve(features[start:start + 512], pool, pool_ids.to(device), plan["shortlist_k"],
                                                 batch_ids.to(device), identity_ids[batch_ids].to(device), identity_ids[pool_ids].to(device))
                    nearest.append(positions.cpu())
                    similarities.append(values.cpu())
                positions = torch.cat(nearest)
                if (identity_ids[pool_ids[positions]] == identity_ids[torch.tensor(query_ids)][:, None]).any():
                    raise ValueError("Pixel-identical candidate remains")
                atomic_save({"query_ids": query_ids, "positions": positions, "similarities": torch.cat(similarities),
                             "pool_ids": pool_ids, "pool_features": pool.cpu(), "plan": plan,
                             "identity_ids": identity_ids, "prepared_from": args.prepared_from}, output / f"{split}_shortlist.pt")
                print(f"Reused frozen {split} features; rebuilt identity-safe top-{plan['shortlist_k']} shortlist: {len(query_ids)} queries", flush=True)
            return
        encoder = DINOv3Adapter(str(ROOT / "third_party/dinov3"), weights=str(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")).to(device).eval().requires_grad_(False)
        frozen_dir = ROOT / "outputs/tacs_full_pool_dino_control_v1"
        pool_path = frozen_dir / "pool_features.pt"
        if pool_path.exists():
            cached_pool = torch.load(pool_path, weights_only=False)
            if not torch.equal(cached_pool["ids"], pool_ids) or cached_pool["metadata"]["selector"] != "frozen_dino":
                raise ValueError("Frozen pool mismatch")
            pool = cached_pool["features"].to(device)
        else:
            frozen_dir.mkdir(parents=True, exist_ok=True)
            pool = encode(encoder, nn.Identity(), source, pool_ids.tolist(), pool_path,
                          {"selector": "frozen_dino", "official_weights_sha256": plan["official_weights_sha256"]},
                          device, 64, args.workers)
        for split, query_ids in selected.items():
            features = encode(encoder, nn.Identity(), source, query_ids, output / f"{split}_features.pt",
                              plan, device, 64, args.workers)
            neighbors, similarities = [], []
            for start in range(0, len(features), 512):
                values, positions = retrieve(features[start:start + 512], pool, pool_ids.to(device), plan["shortlist_k"],
                                             torch.tensor(query_ids[start:start + 512], device=device),
                                             identity_ids[torch.tensor(query_ids[start:start + 512])].to(device), identity_ids[pool_ids].to(device))
                neighbors.append(positions.cpu())
                similarities.append(values.cpu())
            positions = torch.cat(neighbors)
            if (pool_ids[positions] == torch.tensor(query_ids)[:, None]).any():
                raise ValueError("Self retrieval")
            atomic_save({"query_ids": query_ids, "positions": positions, "similarities": torch.cat(similarities),
                         "pool_ids": pool_ids, "pool_features": pool.cpu(), "identity_ids": identity_ids, "plan": plan}, output / f"{split}_shortlist.pt")
            print(f"Prepared {split}: {len(query_ids)} queries, {plan['shortlist_k']} candidates, all self matches excluded", flush=True)
        return
    cache = {split: torch.load(output / f"{split}_shortlist.pt", weights_only=False) for split in selected}
    if any(cache[x]["plan"] != plan or cache[x]["query_ids"] != selected[x] for x in selected):
        raise ValueError("Shortlist protocol mismatch")
    for split, cached in cache.items():
        if not torch.equal(cached["identity_ids"], identity_ids):
            raise ValueError("Image identities changed since preparation")
        if (identity_ids[pool_ids[cached["positions"]]] == identity_ids[torch.tensor(selected[split])][:, None]).any():
            raise ValueError("Pixel-identical candidate in prepared shortlist")
    pool_features = cache["train"]["pool_features"].to(device)
    pool_pixels = None
    if args.cached_pixels:
        pixel_path = ROOT / args.pixel_cache if args.pixel_cache else output / "pool_pixels_fp16.pt"
        pixels = torch.load(pixel_path, weights_only=False)
        if not torch.equal(pixels["pool_ids"], pool_ids) or pixels["weights_sha256"] != plan["official_weights_sha256"]:
            raise ValueError("Pool pixel cache metadata mismatch")
        pool_pixels = pixels["pixels"].to(device)
        print(f"GPU-resident candidate images: {pool_pixels.numel() * pool_pixels.element_size() / 2**30:.2f} GiB", flush=True)
        del pixels
    if args.shared_selector_backbone and (pool_pixels is None or "full_tacs" not in args.arms):
        raise ValueError("Shared selector backbone requires --cached-pixels and the full_tacs arm")
    if args.stage == "train":
        models, optimizers, scalers, schedulers, histories, bests, pending_metrics = {}, {}, {}, {}, {}, {}, {}
        for arm in args.arms:
            run = output / arm
            run.mkdir(exist_ok=True)
            set_seed(args.seed)
            model = build_task(archive, arm, device, args.task_unfreeze_last_blocks,
                               shared_selector_backbone=args.shared_selector_backbone and arm == "full_tacs")
            optimizer = torch.optim.AdamW(param_groups(model, plan), weight_decay=plan["weight_decay"])
            print(f"{arm}: trainable task={sum(p.numel() for p in model.task_model.parameters() if p.requires_grad):,}; selector={sum(p.numel() for p in model.selector.parameters() if p.requires_grad) if model.selector else 0:,}", flush=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scaler = torch.amp.GradScaler("cuda")
            history, best = [], None
            last = run / "checkpoint_last.pt"
            if last.exists():
                # Keep AdamW's non-capturable scalar step counters on CPU.
                # Mapping the entire checkpoint to CUDA forces a device sync
                # per parameter when AdamW reads these scalars with .item().
                saved = torch.load(last, map_location="cpu", weights_only=False)
                if saved["plan"] != plan:
                    raise ValueError("Checkpoint plan mismatch")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                scaler.load_state_dict(saved["scaler"])
                history, best = saved["history"], saved["best"]
            elif (run / "baseline_provenance.json").exists():
                # A prior identical No-Context arm may be imported with its
                # provenance to avoid spending the GPU budget training it twice.
                history = json.loads((run / "history.json").read_text())
                saved = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
                if saved["plan"] != plan or saved["arm"] != arm or len(history) != args.epochs:
                    raise ValueError("Imported completed baseline does not match this plan")
                model.load_state_dict(saved["model"])
                best = saved["best"]
            pending = run / "checkpoint_pending_validation.pt"
            if pending.exists():
                saved = torch.load(pending, map_location="cpu", weights_only=False)
                if saved["epoch"] == len(history) + 1:
                    if saved["plan"] != plan:
                        raise ValueError("Pending checkpoint plan mismatch")
                    model.load_state_dict(saved["model"])
                    optimizer.load_state_dict(saved["optimizer"])
                    scheduler.load_state_dict(saved["scheduler"])
                    scaler.load_state_dict(saved["scaler"])
                    pending_metrics[arm] = saved["trained"]
            steps = [state["step"] for state in optimizer.state.values() if "step" in state]
            if any(step.device.type != "cpu" for step in steps):
                raise ValueError("Non-capturable AdamW step counters must stay on CPU")
            print(f"{arm}: restored {len(history)} epochs, optimizer scalar counters on CPU={len(steps)}", flush=True)
            models[arm], optimizers[arm], schedulers[arm], scalers[arm] = model, optimizer, scheduler, scaler
            histories[arm], bests[arm] = history, best
        for epoch in range(args.epochs):
            for arm in args.arms:
                if len(histories[arm]) > epoch:
                    continue
                set_seed(args.seed + epoch * 1009)
                print(f"START {arm} epoch {epoch + 1}/{args.epochs}", flush=True)
                if arm in pending_metrics:
                    trained = pending_metrics.pop(arm)
                    print(f"Resume pending validation for {arm}, no repeat training", flush=True)
                else:
                    loader = make_loader(source, selected["train"], source, pool_ids, cache["train"]["positions"], arm, args, True, epoch)
                    trained, _, _, _ = run_epoch(models[arm], loader, arm, device, pool_features, optimizers[arm], scalers[arm], pool_pixels=pool_pixels)
                    atomic_save({"model": models[arm].state_dict(), "optimizer": optimizers[arm].state_dict(),
                                 "scheduler": schedulers[arm].state_dict(), "scaler": scalers[arm].state_dict(),
                                 "epoch": epoch + 1, "trained": trained, "plan": plan},
                                output / arm / "checkpoint_pending_validation.pt")
                loader = make_loader(source, selected["validation"], source, pool_ids, cache["validation"]["positions"], arm, args, False)
                validation_pool_features = pool_features
                if arm == "full_tacs" and models[arm].selector.encoder is models[arm].task_model.backbone:
                    validation_pool_features = encode_candidate_globals(models[arm].selector.encoder, pool_pixels.unsqueeze(0))[0]
                val, probs, labels, pairs = run_epoch(models[arm], loader, arm, device, validation_pool_features, pool_pixels=pool_pixels)
                schedulers[arm].step()
                row = {"epoch": epoch + 1, "train": trained, "validation": val}
                histories[arm].append(row)
                best = bests[arm]
                if best is None or (val["accuracy"], -val["ce"]) > (best["validation"]["accuracy"], -best["validation"]["ce"]):
                    bests[arm] = row
                    atomic_save({"model": models[arm].state_dict(), "plan": plan, "arm": arm, "best": row}, output / arm / "checkpoint_best.pt")
                    atomic_save({"probabilities": probs, "labels": labels, "epoch": epoch + 1}, output / arm / "validation_best.pt")
                    save_pairs(pairs, output / arm / "validation_selected_pairs.csv")
                atomic_save({"model": models[arm].state_dict(), "optimizer": optimizers[arm].state_dict(),
                             "scheduler": schedulers[arm].state_dict(), "scaler": scalers[arm].state_dict(),
                             "history": histories[arm], "best": bests[arm], "plan": plan}, output / arm / "checkpoint_last.pt")
                json_save(histories[arm], output / arm / "history.json")
                print(f"FINISH {arm} {json.dumps(row)}", flush=True)
        print("Matched training complete. Test has not been read.", flush=True)
    if args.stage == "validate":
        saved = {arm: torch.load(output / arm / "validation_best.pt", weights_only=False) for arm in ARMS}
        y = saved["no_context"]["labels"]
        if any(not torch.equal(value["labels"], y) for value in saved.values()):
            raise ValueError("Validation order mismatch")
        result = {"metrics": {arm: {"epoch": value["epoch"], **metrics(value["probabilities"], y)} for arm, value in saved.items()},
                  "full_vs_no_context": paired_difference(saved["full_tacs"]["probabilities"], saved["no_context"]["probabilities"], y),
                  "full_vs_dino": paired_difference(saved["full_tacs"]["probabilities"], saved["dino_similarity"]["probabilities"], y), "plan": plan}
        json_save(result, output / "validation_comparison.json")
        print(json.dumps(result, indent=2), flush=True)
    if args.stage == "test":
        raise NotImplementedError("Final test follows locked positive validation evidence; no test data have been read")


if __name__ == "__main__":
    main()
