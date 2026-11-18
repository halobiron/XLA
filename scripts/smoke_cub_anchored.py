"""GPU smoke test for the TACS-X anchored extension on the full-pool CUB protocol.

Checks mechanics and cost only (finite losses/gradients, detached reward, null
option, VRAM, step time). A few dozen optimizer steps are not an accuracy result.
Initializes from the existing query-only warm start (absolute epoch 5), which is
read only; its hash is verified before and after. Validation only, no test access.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from PIL import Image

import run_cub_pilot as pilot
import run_cub_quick_check as quick
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier

ROOT = pilot.ROOT
WARM = ROOT / "outputs/cub200_quick_warmstart_s42_v1/warmup/checkpoint_last.pt"
LR = {"lr_backbone": 1e-5, "lr_head": 3e-4, "lr_selector": 1e-4, "lr_gate": 1e-3}
VARIANTS = {"paper_tacs": {"aux_query_weight": 0.0, "abstain_gate": False},
            "aux_only": {"aux_query_weight": 1.0, "abstain_gate": False},
            "anchored_tacs": {"aux_query_weight": 1.0, "abstain_gate": True}}


def grad_norm(params):
    grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(grads).norm()) if grads else None


def gather_from(pixels):
    return lambda m: (m.to(pixels.dtype) @ pixels.flatten(1)).reshape(len(m), *pixels.shape[1:])


def train_steps(name, model, train, pixels, pool_ids, identities, steps):
    optimizer = torch.optim.AdamW(pilot.param_groups(model, LR), weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    pilot._set_training_mode(model)
    features = pilot.candidate_features(model, pixels)
    torch.cuda.reset_peak_memory_stats()
    rows, times = [], []
    loader = iter(train)
    for step in range(steps):
        batch = next(loader)
        q, y = batch["query"].cuda(), batch["label"].cuda()
        torch.cuda.synchronize(); start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            scores = pilot.utility(model, q, features, batch["query_id"], pool_ids, identities)
            scores.retain_grad()
            if model.anchored:
                out = model.anchored_step(q, y, scores, gather_from(pixels))
                task, policy, reward = out.task_loss, out.policy_loss, out.reward
                aux, gate = out.aux_loss, out.gate_loss
                null_rate = float(out.null_selected.float().mean()) if out.null_selected is not None else None
            else:
                _, task, policy, reward, _ = pilot.forward_train(model, q, y, scores, pixels)
                aux, gate, null_rate = None, None, None
            loss = task + model.lambda_policy * policy
            loss = loss + (aux if aux is not None else 0) + (gate if gate is not None else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name}: nonfinite loss at step {step}")
        assert not reward.requires_grad, "reward must be detached"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        score_grad = scores.grad
        finite_grads = all(torch.isfinite(p.grad).all() for g in optimizer.param_groups
                           for p in g["params"] if p.grad is not None)
        scale_before = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        torch.cuda.synchronize(); times.append(time.perf_counter() - start)
        groups = {g["name"]: g["params"] for g in optimizer.param_groups}
        loss, task, policy = loss.detach(), task.detach(), policy.detach()
        row = {"step": step + 1, "loss": float(loss), "task": float(task), "policy": float(policy),
               "aux": float(aux.detach()) if aux is not None else None,
               "gate_loss": float(gate.detach()) if gate is not None else None,
               "mean_reward": float(reward.mean()), "positive_reward_rate": float((reward > 0).float().mean()),
               "query_only_rate": null_rate, "finite_grads": bool(finite_grads),
               "skipped_by_scaler": scaler.get_scale() < scale_before,
               "score_grad_abs_mean": float(score_grad[torch.isfinite(scores)].abs().mean()) if score_grad is not None else None,
               "backbone_grad_norm": grad_norm(groups.get("backbone", [])),
               "head_grad_norm": grad_norm(groups.get("head", [])),
               "gate_grad_norm": grad_norm(groups.get("gate", []))}
        rows.append(row)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"{name} step {step+1}/{steps} {json.dumps(row)}", flush=True)
    steady = times[3:] or times
    return rows, {"mean_step_seconds": sum(steady) / len(steady),
                  "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}


@torch.no_grad()
def evaluate(model, val, pixels, pool_ids, identities, batches):
    model.eval()
    features = pilot.candidate_features(model, pixels)
    correct = query_only = context_only = either = nulls = total = 0
    gate_right = gate_decisions = 0
    for i, batch in enumerate(val):
        if i == batches:
            break
        q, y = batch["query"].cuda(), batch["label"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            scores = pilot.utility(model, q, features, batch["query_id"], pool_ids, identities)
            if model.anchored:
                out = model.anchored_argmax(q, y, scores, lambda index: pixels[index])
                logits = out.logits
                if out.null_selected is not None:
                    nulls += int(out.null_selected.sum())
            else:
                logits = model.task_model(q, pixels[scores.argmax(-1)])
            ctx = model.task_model(q, pixels[scores.argmax(-1)])  # forced context, ignores gate
            base = model.task_model(q)
        correct += int(logits.argmax(-1).eq(y).sum()); query_only += int(base.argmax(-1).eq(y).sum())
        b_ok, c_ok = base.argmax(-1).eq(y), ctx.argmax(-1).eq(y)
        context_only += int(c_ok.sum()); either += int((b_ok | c_ok).sum())
        if model.abstain_gate:
            # Diagnostic (labels): on queries where exactly one path is right,
            # how often did the gate pick that path?
            differ = b_ok != c_ok
            gate_decisions += int(differ.sum())
            gate_right += int((differ & (out.null_selected == b_ok)).sum())
        total += len(y)
    return {"queries": total, "accuracy": correct / total, "internal_query_only_accuracy": query_only / total,
            "forced_context_accuracy": context_only / total,
            "diagnostic_either_correct_bound": either / total,
            "query_only_rate": nulls / total if model.abstain_gate else None,
            "gate_correct_on_disagreements": f"{gate_right}/{gate_decisions}" if model.abstain_gate else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/cub200_anchored_smoke_s42")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    warm_hash = pilot.digest(WARM)
    original_open, image_audit = quick.protect_test_images()
    started = time.perf_counter()
    try:
        pilot.set_seed(42)
        train, val, pixels, pool_ids, identities = pilot.prepare(output)
        baseline = pilot.models()["no_context"]
        saved = torch.load(WARM, map_location="cpu", weights_only=False)
        baseline.load_state_dict(saved["model"]); del saved
        report = {"scope": "GPU smoke test of mechanics/cost; not an accuracy comparison",
                  "init": str(WARM.relative_to(ROOT)), "steps": args.steps, "batch_size": train.batch_size,
                  "train_batches_per_epoch": len(train), "pool": int(len(pool_ids)), "variants": {}}
        for name in args.variants:
            flags = VARIANTS[name]
            pilot.set_seed(42)
            train.generator.manual_seed(42)
            task = copy.deepcopy(baseline.task_model)
            model = TACSClassifier(TACSSelector(task.backbone, projection=False), task, tau=.1,
                                   lambda_policy=1., **flags).cuda()
            before = evaluate(model, val, pixels, pool_ids, identities, args.eval_batches)
            rows, cost = train_steps(name, model, train, pixels, pool_ids, identities, args.steps)
            after = evaluate(model, val, pixels, pool_ids, identities, args.eval_batches)
            report["variants"][name] = {**flags, **cost,
                "estimated_epoch_minutes": cost["mean_step_seconds"] * len(train) / 60,
                "all_losses_finite": True, "nonfinite_grad_steps": sum(not r["finite_grads"] for r in rows),
                "scaler_skipped_steps": sum(r["skipped_by_scaler"] for r in rows),
                "validation_subset_before": before, "validation_subset_after": after, "steps": rows}
            del model, task
            torch.cuda.empty_cache()
        assert pilot.digest(WARM) == warm_hash, "warm-start checkpoint changed"
        report.update(warm_start_unchanged=True, test_pixels_read=False, image_access=image_audit,
                      elapsed_seconds=time.perf_counter() - started)
        pilot.dump(output / "smoke.json", report)
        summary = {k: {f: v[f] for f in ("mean_step_seconds", "estimated_epoch_minutes", "peak_vram_gib",
                                          "nonfinite_grad_steps", "scaler_skipped_steps",
                                          "validation_subset_before", "validation_subset_after")}
                   for k, v in report["variants"].items()}
        print("SMOKE " + json.dumps(summary, indent=1), flush=True)
    finally:
        Image.open = original_open


if __name__ == "__main__":
    main()
