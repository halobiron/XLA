"""Real-model no-update check of cached FP16 candidate-pixel optimization."""
import zipfile

import torch

from train_tacs_shortlist import ROOT, TrainingImages, build_task, learned_scores
from tacsx.engine import _set_training_mode, set_seed
from search_native_context import json_save


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    run = ROOT / "outputs/tacs_shortlist_fullft_v2"
    data = torch.load(run / "train_shortlist.pt", weights_only=False)
    source = TrainingImages(ROOT / "data")
    qids = data["query_ids"][:2]
    positions = data["positions"][:2]
    candidate_ids = data["pool_ids"][positions]
    q = torch.stack([source[i][0] for i in qids]).to(device)
    original = torch.stack([torch.stack([source[int(i)][0] for i in row]) for row in candidate_ids]).to(device)
    features = data["pool_features"].to(device)[positions.to(device)]
    model = build_task(zipfile.ZipFile(ROOT / "outputs.zip"), "full_tacs", device)
    _set_training_mode(model)
    collected = []
    names = ("selector.projector.weight", "task_model.head.weight", "task_model.backbone.model.patch_embed.proj.weight")
    for candidates in (original, original.half()):
        set_seed(4321)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            scores = learned_scores(model, q, features, 1 / .07)
            output = model.run_mode("full_tacs", q, candidates, torch.tensor([source.targets[i] for i in qids], device=device), scores)
            loss = output.task_loss + output.policy_loss
        loss.backward()
        params = dict(model.named_parameters())
        collected.append({"logits": output.logits.detach().cpu(), "loss": loss.detach().cpu(),
                          **{name: params[name].grad.detach().cpu().clone() for name in names}})
    errors = {key: float((collected[0][key] - collected[1][key]).abs().max()) for key in collected[0]}
    finite = all(torch.isfinite(value).all().item() for result in collected for value in result.values())
    result = {"maximum_absolute_errors": errors, "finite": finite,
              "parameters_updated": False, "real_dinov3_forward_and_backward": True,
              "exact_for_checked_batch": finite and all(value == 0 for value in errors.values())}
    json_save(result, run / "pixel_cache_real_gradient_verification.json")
    print(result, flush=True)
    if not finite or any(value > 1e-4 for value in errors.values()):
        raise ValueError("Real-model cache equivalence check failed")


if __name__ == "__main__":
    main()
