"""Read-only clean_v3 diagnosis: exhaustive validation context interventions.

Never trains, selects checkpoints, or reads CIFAR's test file. Test analysis
uses the already locked saved logits only. Oracle results are diagnostic bounds,
not deployable retrieval. Writes only to a separate diagnostics directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from tacsx.stepwise import StepwiseEngine, RUN


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(logits, labels):
    return {"accuracy": float(logits.argmax(-1).eq(labels).float().mean()),
            "ce": float(F.cross_entropy(logits, labels))}


def summarize_interventions(output):
    data = torch.load(output / "validation_interventions.pt", map_location="cpu", weights_only=False)
    y, logits, scores = data["labels"], data["candidates"], data["scores"]
    ce = F.cross_entropy(logits.flatten(0,1), y.repeat_interleave(logits.shape[1]), reduction="none").reshape(len(y),-1)
    base = F.cross_entropy(data["tacs_query"], y, reduction="none")
    nc = F.cross_entropy(data["nc"], y, reduction="none")
    reward = base[:,None]-ce
    probs = scores.softmax(-1)
    b = torch.arange(len(y))
    action = scores.argmax(-1)
    correct = logits.argmax(-1).eq(y[:,None])
    nc_correct = data["nc"].argmax(-1).eq(y)
    chosen = correct[b,action]
    pair_a,pair_b = torch.triu_indices(scores.shape[1],scores.shape[1],1)
    ds,du = scores[:,pair_a]-scores[:,pair_b],ce[:,pair_b]-ce[:,pair_a]
    rank_valid = (ds.abs()>1e-8) & (du.abs()>1e-8)
    rank_per_query = ((ds*du>0)&rank_valid).sum(-1)/rank_valid.sum(-1).clamp_min(1)
    values = {
        "learned_minus_dino_accuracy_pp": (chosen.float()-correct[:,0].float())*100,
        "learned_minus_uniform_accuracy_pp": (chosen.float()-correct.float().mean(-1))*100,
        "learned_minus_own_sampling_expectation_accuracy_pp": (chosen.float()-(correct*probs).sum(-1))*100,
        "learned_minus_no_context_accuracy_pp": (chosen.float()-nc_correct.float())*100,
        "pairwise_utility_agreement_per_query": rank_per_query,
    }
    rng = np.random.default_rng(20261114)
    draws = {k:[] for k in values}
    for _ in range(25):
        inds = rng.integers(0,len(y),(200,len(y)))
        for key,v in values.items():
            draws[key].extend(v.numpy()[inds].mean(-1).tolist())
    intervals = {k:{"mean":float(v.mean()),"exploratory_paired_query_ci95":np.quantile(draws[k],[.025,.975]).tolist()} for k,v in values.items()}
    between = reward.mean(-1).var(unbiased=False)
    within = reward.var(-1,unbiased=False).mean()
    result = {"scope":"Post-hoc exploratory validation diagnosis; intervals unadjusted; no selected model or test tuning",
              "count":len(y), "intervals":intervals,
              "reward_variance":{"between_query":float(between),"within_query_context":float(within),
                                 "between_query_fraction":float(between/(between+within)),
                                 "candidate_weighting":"uniform over the 16 contexts for each fixed query"},
              "selection":{"argmax_matches_label_min_ce_fraction_DIAGNOSTIC_ONLY":float(action.eq(ce.argmin(-1)).float().mean()),
                           "mean_probability_of_label_best_context_DIAGNOSTIC_ONLY":float(probs[b,ce.argmin(-1)].mean()),
                           "positive_reward_but_higher_ce_than_independent_nc":float(((reward[b,action]>0)&(ce[b,action]>nc)).float().mean())},
              "oracle_bound_DIAGNOSTIC_ONLY":{"nc_wrong_but_any_context_correct":int((~nc_correct&correct.any(-1)).sum()),
                                               "nc_correct_but_all_contexts_wrong":int((nc_correct&~correct.any(-1)).sum()),
                                               "nc_wrong_and_all_contexts_wrong":int((~nc_correct&~correct.any(-1)).sum())}}
    (output / "supplementary_analysis.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)


def gradient_audit(engine, output, batches=12, batch_size=32):
    """No optimizer steps; sample training batches with actual augmentation."""
    from PIL import Image
    from tacsx.data.cifar100 import _transform
    from tacsx.engine import _set_training_mode, set_seed

    full = engine.models["full_tacs"]
    full.task_model.requires_grad_(True)
    full.selector.projector.requires_grad_(True)
    _set_training_mode(full)
    train = torch.load(RUN / "train_shortlist.pt", map_location="cpu", weights_only=False)
    order = torch.randperm(len(train["query_ids"]))[:batches * batch_size]
    augmentation = _transform(224, True)
    rows = []
    for step in range(batches):
        set_seed(20261114 + step)
        indices = order[step*batch_size:(step+1)*batch_size]
        qids = [train["query_ids"][i] for i in indices]
        pos = train["positions"][indices]
        cids = train["pool_ids"][pos]
        q = torch.stack([augmentation(Image.fromarray(engine.images[i])) for i in qids]).cuda()
        c = torch.stack([torch.stack([engine.tensor(engine.images[i]) for i in row]) for row in cids]).cuda()
        labels = torch.tensor([engine.targets[i] for i in qids], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                features = full.selector.encoder.encode_global(q)
            zq = F.normalize(full.selector.projector(features), dim=-1)
            zc = F.normalize(full.selector.projector(engine.pool_features[pos.cuda()]), dim=-1)
            scores = torch.einsum("bd,bnd->bn", zq, zc) * engine.plan["selector_logit_scale"]
            out = full.run_mode("full_tacs", q, c, labels, scores)
        weight = full.selector.projector.weight
        gt = torch.autograd.grad(out.task_loss, weight, retain_graph=True)[0].float().flatten()
        gp = torch.autograd.grad(out.policy_loss, weight, retain_graph=True)[0].float().flatten()
        ph = torch.autograd.grad(out.policy_loss, full.task_model.head.weight, allow_unused=True)[0]
        rows.append({"seed": 20261114+step, "task_gradient_norm": float(gt.norm()),
                     "policy_gradient_norm": float(gp.norm()),
                     "policy_to_task_gradient_ratio": float(gp.norm()/gt.norm().clamp_min(1e-12)),
                     "gradient_cosine": float(F.cosine_similarity(gt[None],gp[None])),
                     "finite": bool(torch.isfinite(gt).all() and torch.isfinite(gp).all()),
                     "reward_detached": not out.reward.requires_grad, "policy_head_grad_absent": ph is None,
                     "mean_reward": float(out.reward.mean())})
        del out, gt, gp, c, q
        print(f"gradient audit {step+1}/{batches}", flush=True)
    w = weight.detach().float()
    result = {"scope": "Exploratory gradient audit on training images at the locked best checkpoint; no optimization",
              "batch_size": batch_size, "batches": rows, "optimizer_steps": 0,
              "mean_gradient_ratio": sum(r["policy_to_task_gradient_ratio"] for r in rows)/len(rows),
              "mean_gradient_cosine": sum(r["gradient_cosine"] for r in rows)/len(rows),
              "projector_frobenius_distance_from_identity": float((w-torch.eye(w.shape[0],device=w.device)).norm())}
    (output / "gradient_audit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4996)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="outputs/diagnostics/clean_v3_context_cause")
    parser.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        torch.set_num_threads(4)
        summarize_interventions(ROOT / args.output)
        return
    if not torch.cuda.is_available():
        parser.error("Use .venv-gpu/Scripts/python.exe with CUDA")
    torch.set_num_threads(4)
    torch.manual_seed(20261114)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {a: RUN / a / "checkpoint_best.pt" for a in ("no_context", "full_tacs")}
    before = {a: digest(p) for a, p in checkpoint_paths.items()}
    lock = json.loads((RUN / "evaluation/checkpoint_lock.json").read_text())
    assert before == lock["checkpoint_sha256"]
    engine = StepwiseEngine("cuda").load()
    if args.gradient_only:
        gradient_audit(engine, output)
        assert {a: digest(p) for a,p in checkpoint_paths.items()} == before
        return
    full = engine.models["full_tacs"]
    nc = engine.models["no_context"]
    positions = torch.randperm(len(engine.val_ids))[:args.samples]
    rows = positions.tolist()
    qids = torch.tensor([engine.val_ids[i] for i in rows])
    labels = torch.tensor([engine.targets[i] for i in qids])
    neighbor_pos = engine.val_positions[positions]
    candidate_ids = engine.pool_ids[neighbor_pos]
    assert not engine.identity_ids[candidate_ids].eq(engine.identity_ids[qids, None]).any()
    records = {k: [] for k in ("nc", "tacs_query", "self_context", "candidates", "scores")}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            ids = qids[start:start + args.batch_size]
            q = torch.stack([engine.tensor(engine.images[i]) for i in ids]).cuda()
            cids = candidate_ids[start:start + len(ids)]
            qfeature = full.selector.encoder.encode_global(q)
            zq = F.normalize(full.selector.projector(qfeature), dim=-1)
            features = engine.pool_features[neighbor_pos[start:start + len(ids)].cuda()]
            zc = F.normalize(full.selector.projector(features), dim=-1)
            scores = torch.einsum("bd,bnd->bn", zq, zc) * engine.plan["selector_logit_scale"]
            records["scores"].append(scores.cpu())
            records["nc"].append(nc.task_model(q, None).cpu())
            records["tacs_query"].append(full.task_model(q, None).cpu())
            records["self_context"].append(full.task_model(q, q).cpu())
            # Each candidate is forwarded discretely, with the same locked task
            # weights. No labels enter candidate scoring or model inputs.
            all_logits = []
            for rank in range(cids.shape[1]):
                c = torch.stack([engine.tensor(engine.images[i]) for i in cids[:, rank]]).cuda()
                all_logits.append(full.task_model(q, c).cpu())
            records["candidates"].append(torch.stack(all_logits, 1))
            if start % (args.batch_size * 10) == 0 or start + len(ids) == len(rows):
                print(f"validation interventions {start + len(ids)}/{len(rows)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    data = {k: torch.cat(v) for k, v in records.items()}
    data.update(query_ids=qids, labels=labels, candidate_ids=candidate_ids)
    torch.save(data, output / "validation_interventions.pt")
    scores, logits = data["scores"], data["candidates"]
    probs = scores.softmax(-1)
    actions = scores.argmax(-1)
    arange = torch.arange(len(labels))
    learned = logits[arange, actions]
    ce = F.cross_entropy(logits.flatten(0, 1), labels.repeat_interleave(logits.shape[1]), reduction="none").reshape(len(labels), -1)
    correct = logits.argmax(-1).eq(labels[:, None])
    chosen_ce = ce[arange, actions]
    chosen_correct = correct[arange, actions]
    base_ce = F.cross_entropy(data["tacs_query"], labels, reduction="none")
    nc_ce = F.cross_entropy(data["nc"], labels, reduction="none")
    oracle = ce.argmin(-1)
    candidate_labels = torch.tensor([[engine.targets[i] for i in row] for row in candidate_ids])
    same = candidate_labels.eq(labels[:, None])
    pair_a, pair_b = torch.triu_indices(scores.shape[1], scores.shape[1], 1)
    ds = scores[:, pair_a] - scores[:, pair_b]
    du = ce[:, pair_b] - ce[:, pair_a]
    valid = (ds.abs() > 1e-8) & (du.abs() > 1e-8)
    saved_val = torch.load(RUN / "evaluation/validation_predictions.pt", weights_only=False, map_location="cpu")
    saved_logits = {a: saved_val["logits"][a][positions, 0] for a in checkpoint_paths}
    assert torch.equal(saved_val["query_ids"][positions], qids)
    best = {a: torch.load(p, map_location="cpu", weights_only=False)["best"] for a, p in checkpoint_paths.items()}
    result = {
        "scope": "Post-hoc diagnosis, one seed, frozen checkpoints; validation interventions only; no test tuning",
        "count": len(labels), "checkpoint_sha256": before, "pair_rope_layout": engine.pair_rope_layout,
        "best_epoch": {a: b["epoch"] for a, b in best.items()},
        "validation_controls": {"independent_no_context": describe(data["nc"], labels),
                                "full_tacs_selected": describe(learned, labels),
                                "full_tacs_query_only": describe(data["tacs_query"], labels),
                                "full_tacs_dino_nearest1": describe(logits[:, 0], labels),
                                "full_tacs_duplicate_query_ablation": describe(data["self_context"], labels),
                                "full_tacs_uniform16_expectation": {"accuracy": float(correct.float().mean()), "ce": float(ce.mean())},
                                "full_tacs_training_policy_expectation": {"accuracy": float((correct * probs).sum(-1).mean()), "ce": float((ce * probs).sum(-1).mean())},
                                "label_oracle_min_ce_DIAGNOSTIC_ONLY": describe(logits[arange, oracle], labels),
                                "label_oracle_any_correct_DIAGNOSTIC_ONLY": float(correct.any(-1).float().mean())},
        "reconstruction": {"nc_prediction_agreement": float(data["nc"].argmax(-1).eq(saved_logits["no_context"].argmax(-1)).float().mean()),
                           "tacs_prediction_agreement": float(learned.argmax(-1).eq(saved_logits["full_tacs"].argmax(-1)).float().mean()),
                           "nc_max_logit_error": float((data["nc"]-saved_logits["no_context"]).abs().max()),
                           "tacs_max_logit_error": float((learned-saved_logits["full_tacs"]).abs().max())},
        "selector": {"top1_agreement_with_dino": float(actions.eq(0).float().mean()),
                     "mean_selected_dino_rank_1based": float((actions+1).float().mean()),
                     "mean_top1_probability": float(probs.max(-1).values.mean()),
                     "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum(-1).mean()),
                     "effective_candidates_exp_entropy": float((-(probs * probs.clamp_min(1e-12).log()).sum(-1)).exp().mean()),
                     "pairwise_score_utility_agreement": float(((ds*du > 0) & valid).sum() / valid.sum()),
                     "mean_ce_regret_to_oracle": float((chosen_ce-ce.min(-1).values).mean()),
                     "fraction_query_predictions_depend_on_context": float((logits.argmax(-1).max(-1).values != logits.argmax(-1).min(-1).values).float().mean()),
                     "class_diagnostic_ONLY": {"selected_same_class": float(same[arange,actions].float().mean()),
                                               "dino_same_class": float(same[:,0].float().mean()),
                                               "shortlist_any_same_class": float(same.any(-1).float().mean()),
                                               "accuracy_when_selected_same_class": float(chosen_correct[same[arange,actions]].float().mean()),
                                               "accuracy_when_selected_cross_class": float(chosen_correct[~same[arange,actions]].float().mean())}},
        "reward": {"mean_vs_own_query_only": float((base_ce-chosen_ce).mean()),
                   "mean_vs_independent_no_context": float((nc_ce-chosen_ce).mean()),
                   "positive_fraction_vs_own_query_only": float((base_ce>chosen_ce).float().mean()),
                   "positive_fraction_vs_independent_no_context": float((nc_ce>chosen_ce).float().mean()),
                   "warning": "Own-query-only baseline is not trained by an independent query loss in Full TACS. Positive reward does not establish benefit over independently trained No-Context."},
    }
    # The only test data loaded are the existing immutable prediction artifact.
    test = torch.load(RUN / "evaluation/test_standard/test_predictions.pt", weights_only=False, map_location="cpu")
    y = test["labels"]
    nc_test, ft_test = (test["logits"][a] for a in checkpoint_paths)
    a, b = nc_test.argmax(-1).eq(y), ft_test.argmax(-1).eq(y)
    flips = a != b
    discordant = int(flips.sum())
    smaller_count = min(int((~a & b).sum()), int((a & ~b).sum()))
    mcnemar_p = min(1., 2 * sum(math.comb(discordant, k) for k in range(smaller_count + 1)) / 2**discordant)
    result["saved_test_analysis"] = {"no_context": describe(nc_test,y), "full_tacs": describe(ft_test,y),
                                     "fixed": int((~a & b).sum()), "broken": int((a & ~b).sum()),
                                     "prediction_disagreements": int(nc_test.argmax(-1).ne(ft_test.argmax(-1)).sum()),
                                     "mcnemar_exact_two_sided_p": mcnemar_p}
    result["elapsed_seconds"] = time.perf_counter()-started
    after = {a: digest(p) for a,p in checkpoint_paths.items()}
    assert after == before, "Original checkpoints changed during diagnosis"
    result["checkpoints_unchanged"] = True
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
