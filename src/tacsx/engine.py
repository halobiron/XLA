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

from tacsx.models.baselines import dino_similarity_scores


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


def execute(model, mode, batch, device):
    query, candidates, labels = (batch[k].to(device, non_blocking=True) for k in ("query", "candidates", "label"))
    scores = None if mode == "no_context" else scores_for_mode(model, mode, query, candidates)
    return model.run_mode(mode, query, candidates, labels, scores), query, candidates, labels


def train_epoch(model, loader, optimizer, mode, device, lambda_policy, scaler=None):
    model.train(); total_loss = total_correct = total = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out, _, _, labels = execute(model, mode, batch, device)
            loss = out.task_loss + lambda_policy * out.policy_loss
        if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.detach().item() * labels.numel()
        total_correct += (out.logits.argmax(1) == labels).sum().item(); total += labels.numel()
    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.no_grad()
def evaluate(model, loader, mode, device, collect_pairs=False):
    model.eval(); total_loss = total_correct = total = 0; pairs = []; entropies = []; rewards = []
    for batch in loader:
        out, query, candidates, labels = execute(model, mode, batch, device)
        total_loss += out.task_loss.item() * labels.numel(); total_correct += (out.logits.argmax(1) == labels).sum().item(); total += labels.numel()
        if out.scores.numel():
            probs = out.scores.softmax(-1); chosen = out.scores.argmax(-1)
            entropies.extend((-(probs * probs.clamp_min(1e-12).log()).sum(-1)).cpu().tolist())
            if collect_pairs:
                b = torch.arange(labels.shape[0], device=device)
                ctx = candidates[b, chosen]
                cos = F.cosine_similarity(query.flatten(1), ctx.flatten(1)).cpu().tolist()
                for i in range(labels.shape[0]):
                    pairs.append({"query_id": int(batch["query_id"][i]), "query_label": int(labels[i]), "candidate_id": int(batch["candidate_ids"][i, chosen[i]]), "candidate_label": int(batch["candidate_labels"][i, chosen[i]]), "similarity": cos[i], "selection_score": float(out.scores[i, chosen[i]]), "reward": float(out.reward[i]) if out.reward is not None else ""})
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
