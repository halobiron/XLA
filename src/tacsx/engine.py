from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from tacsx.models.baselines import DINOEmbeddingCache, dino_similarity_scores


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scores_for_mode(model, mode, query, candidates):
    if mode == "random_context":
        return torch.rand(query.shape[0], candidates.shape[1], device=query.device)
    if mode == "dino_similarity":
        return dino_similarity_scores(model.task_model.backbone, query, candidates)
    return model.selector(query, candidates)


@torch.no_grad()
def cache_dino_candidate_embeddings(encoder, context_dataset, device, batch_size=128):
    """Encode each fixed-pool candidate once for the frozen DINO baseline."""
    ids = sorted(item.index for item in context_dataset.pool.items)
    source = Subset(context_dataset.candidate_base, ids)
    loader = DataLoader(source, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    was_training = encoder.training
    encoder.eval()
    chunks = []
    for step, (images, _) in enumerate(loader, start=1):
        chunks.append(F.normalize(encoder.encode_global(images.to(device, non_blocking=True)), dim=-1))
        if step == 1 or step == len(loader) or step % max(1, len(loader) // 10) == 0:
            print(f"  caching DINO candidates {step}/{len(loader)}", flush=True)
    encoder.train(was_training)
    return DINOEmbeddingCache(torch.tensor(ids, device=device), torch.cat(chunks, dim=0))


def execute(model, mode, batch, device, dino_cache=None, candidate_dataset=None):
    query = batch["query"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    candidates = None
    if mode != "no_context" and dino_cache is None:
        candidates = batch["candidates"].to(device, non_blocking=True)
    if mode == "no_context":
        scores = None
    elif dino_cache is not None:
        scores = dino_cache.scores(model.task_model.backbone, query, batch["candidate_ids"])
        action = scores.argmax(dim=-1)
        selected_ids = batch["candidate_ids"].to(device)[torch.arange(query.shape[0], device=device), action].cpu()
        context = candidate_dataset.load_candidate_images(selected_ids).to(device, non_blocking=True)
        out = model.run_mode(mode, query, None, labels, scores, selected_context=context, selected_action=action)
        return out, query, context, labels
    else:
        scores = scores_for_mode(model, mode, query, candidates)
    return model.run_mode(mode, query, candidates, labels, scores), query, candidates, labels


def train_epoch(model, loader, optimizer, mode, device, lambda_policy, scaler=None, show_progress=False, dino_cache=None):
    model.train(); total_loss = total_correct = total = 0
    # Cached DINO embeddings require the frozen feature extractor to remain in
    # evaluation mode (and this also avoids stochastic frozen features).
    for module in model.modules():
        if module is not model and not any(p.requires_grad for p in module.parameters(recurse=True)):
            module.eval()
    log_every = max(1, len(loader) // 10)
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out, _, _, labels = execute(model, mode, batch, device, dino_cache=dino_cache, candidate_dataset=loader.dataset)
            loss = out.task_loss + lambda_policy * out.policy_loss
        if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.detach().item() * labels.numel()
        total_correct += (out.logits.argmax(1) == labels).sum().item(); total += labels.numel()
        if show_progress and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"  train batch {step}/{len(loader)}: loss={total_loss / total:.4f} acc={total_correct / total:.4f}", flush=True)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.no_grad()
def evaluate(model, loader, mode, device, collect_pairs=False, dino_cache=None):
    model.eval(); total_loss = total_correct = total = 0; pairs = []; entropies = []; rewards = []
    for batch in loader:
        out, query, candidates, labels = execute(model, mode, batch, device, dino_cache=dino_cache, candidate_dataset=loader.dataset)
        total_loss += out.task_loss.item() * labels.numel(); total_correct += (out.logits.argmax(1) == labels).sum().item(); total += labels.numel()
        if out.scores.numel():
            probs = out.scores.softmax(-1); chosen = out.scores.argmax(-1) if out.action is None else out.action
            entropies.extend((-(probs * probs.clamp_min(1e-12).log()).sum(-1)).cpu().tolist())
            if collect_pairs:
                b = torch.arange(labels.shape[0], device=device)
                ctx = out.context if out.context is not None else candidates[b, chosen]
                cos = F.cosine_similarity(query.flatten(1), ctx.flatten(1)).cpu().tolist()
                chosen_cpu = chosen.detach().cpu()
                for i in range(labels.shape[0]):
                    pairs.append({"query_id": int(batch["query_id"][i]), "query_label": int(labels[i]), "candidate_id": int(batch["candidate_ids"][i, chosen_cpu[i]]), "candidate_label": int(batch["candidate_labels"][i, chosen_cpu[i]]), "similarity": cos[i], "selection_score": float(out.scores[i, chosen[i]]), "reward": float(out.reward[i]) if out.reward is not None else ""})
        if out.reward is not None: rewards.extend(out.reward.cpu().tolist())
    result = {"avg_loss": total_loss / total, "accuracy": total_correct / total,
              "mean_selection_entropy": float(np.mean(entropies)) if entropies else None,
              "mean_reward": float(np.mean(rewards)) if rewards else None}
    if pairs:
        result["cross_class_rate"] = float(np.mean([x["query_label"] != x["candidate_label"] for x in pairs]))
        result["mean_query_context_cosine"] = float(np.mean([x["similarity"] for x in pairs]))
    return result, pairs


def save_run(output_dir: Path, config: dict, history: list[dict], metrics: dict, model, pairs: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if history:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    if pairs:
        with (output_dir / "selected_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pairs[0].keys()); writer.writeheader(); writer.writerows(pairs)
    torch.save({"model": model.state_dict(), "metrics": metrics, "config": config}, output_dir / "checkpoint_best.pt")
