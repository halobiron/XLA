"""Audit both real TACS gradient paths on a saved shortlist-model checkpoint."""
import hashlib
import io
import zipfile

import torch

from train_tacs_shortlist import ROOT, TrainingImages, build_task, learned_scores
from tacsx.engine import _set_training_mode, set_seed
from search_native_context import json_save


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    run = ROOT / "outputs/tacs_shortlist_fullft_clean_v3"
    data = torch.load(run / "train_shortlist.pt", weights_only=False)
    content = (run / "full_tacs/checkpoint_best.pt").read_bytes()
    saved = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    source = TrainingImages(ROOT / "data")
    qids = data["query_ids"][:8]
    positions = data["positions"][:8]
    cids = data["pool_ids"][positions]
    q = torch.stack([source[i][0] for i in qids]).to(device)
    c = torch.stack([torch.stack([source[int(i)][0] for i in row]) for row in cids]).to(device)
    features = data["pool_features"].to(device)[positions.to(device)]
    model = build_task(zipfile.ZipFile(ROOT / "outputs.zip"), "full_tacs", device)
    model.load_state_dict(saved["model"])
    _set_training_mode(model)
    set_seed(4321)
    with torch.autocast("cuda", dtype=torch.float16):
        scores = learned_scores(model, q, features, saved["plan"]["selector_logit_scale"])
        out = model.run_mode("full_tacs", q, c, torch.tensor([source.targets[i] for i in qids], device=device), scores)
    weight = model.selector.projector.weight
    grad_task = torch.autograd.grad(out.task_loss, weight, retain_graph=True)[0]
    grad_policy = torch.autograd.grad(out.policy_loss, weight, retain_graph=True)[0]
    policy_head_grad = torch.autograd.grad(out.policy_loss, model.task_model.head.weight, allow_unused=True)[0]
    result = {"checkpoint_sha256": hashlib.sha256(content).hexdigest(), "epoch": saved["best"]["epoch"],
              "task_gradient_selector_norm": float(grad_task.float().norm()),
              "policy_gradient_selector_norm": float(grad_policy.float().norm()),
              "gradients_finite": bool(torch.isfinite(grad_task).all() and torch.isfinite(grad_policy).all()),
              "reward_detached": not out.reward.requires_grad,
              "policy_does_not_update_task_head": policy_head_grad is None,
              "selector_projection_distance_from_identity": float((weight.detach() - torch.eye(weight.shape[0], device=device)).norm()),
              "downstream_class": type(model.task_model).__name__, "tau": model.tau, "lambda_policy": model.lambda_policy,
              "optimizer_steps_performed": 0}
    json_save(result, run / "hybrid_real_gradient_audit.json")
    print(result, flush=True)
    if not result["gradients_finite"] or not result["reward_detached"] or not result["policy_does_not_update_task_head"] or min(result["task_gradient_selector_norm"], result["policy_gradient_selector_norm"]) <= 0:
        raise ValueError("Hybrid gradient audit failed")


if __name__ == "__main__":
    main()
