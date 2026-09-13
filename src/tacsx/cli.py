from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch import optim

from .data import make_cifar100_loaders
from .engine import evaluate, save_run, set_seed, train_epoch
from .models.dinov3_adapter import DINOv3Adapter
from .models.dual_input_vit import DualInputViT
from .models.selector import TACSSelector
from .models.tacs import TACSClassifier


REQUIRED_MODES = {"no_context", "random_context", "dino_similarity", "gumbel_only", "policy_only", "full_tacs", "topk_tacs", "adaptive_topk_tacs"}


def _freeze(adapter, freeze: bool, last_n: int):
    if not freeze: return
    for p in adapter.parameters(): p.requires_grad = False
    for block in list(adapter.model.blocks)[-last_n:] if last_n else []:
        for p in block.parameters(): p.requires_grad = True


def build_model(cfg, allow_random_backbone: bool, include_context: bool = True):
    back = cfg["backbone"]
    weights = back.get("weights")
    if not weights and not allow_random_backbone:
        raise ValueError("DINOv3 weights are required. Set backbone.weights via official access, or explicitly pass --allow-random-backbone for a smoke test.")
    kwargs = {"weights": weights} if weights else {"pretrained": False}
    task_adapter = DINOv3Adapter(back["repo_dir"], **kwargs)
    _freeze(task_adapter, back.get("freeze", False), int(back.get("unfreeze_last_n_blocks", 0)))
    if not include_context:
        task = DualInputViT(task_adapter, 100)
        return TACSClassifier(None, task, tau=float(cfg["selector"].get("tau", .1)), lambda_policy=float(cfg["policy"].get("lambda_policy", 1.0)))

    selector_adapter = DINOv3Adapter(back["repo_dir"], **kwargs)
    _freeze(selector_adapter, back.get("freeze", False), int(back.get("unfreeze_last_n_blocks", 0)))
    task = DualInputViT(task_adapter, 100)
    selector = TACSSelector(selector_adapter, projection=cfg["selector"].get("projection", True))
    topk = cfg.get("topk", {})
    return TACSClassifier(selector, task, tau=float(cfg["selector"].get("tau", .1)), lambda_policy=float(cfg["policy"].get("lambda_policy", 1.0)), topk_k=int(topk.get("k", 2)), adaptive_threshold=float(topk.get("adaptive_threshold", .9)))


def main(argv=None):
    p = argparse.ArgumentParser(description="TACS-X CIFAR-100 experiment runner")
    p.add_argument("--config", default="configs/cifar100.yaml")
    p.add_argument("--mode", choices=sorted(REQUIRED_MODES))
    p.add_argument("--name")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--allow-random-backbone", action="store_true", help="only for smoke tests; logs a fidelity deviation")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode: cfg["experiment"]["mode"] = args.mode
    if args.name: cfg["experiment"]["name"] = args.name
    mode = cfg["experiment"]["mode"]
    include_context = mode != "no_context"
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    train_loader, val_loader, test_loader, _ = make_cifar100_loaders(
        cfg["data"]["root"], cfg["data"]["image_size"], cfg["data"]["candidate_pool_ratio"],
        cfg["data"]["candidates_per_query"], cfg["training"]["batch_size"], cfg["data"]["num_workers"], cfg["experiment"]["seed"], include_context=include_context,
        validation_ratio=float(cfg["data"].get("validation_ratio", 0.1)))
    model = build_model(cfg, args.allow_random_backbone, include_context=include_context).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=float(cfg["training"]["lr_head"]), weight_decay=float(cfg["training"]["weight_decay"]))
    use_amp = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    history, best, best_state = [], -1.0, None
    for epoch in range(int(cfg["training"]["epochs"])):
        train_loader.dataset.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, mode, device, model.lambda_policy, scaler, show_progress=True)
        val_metrics, _ = evaluate(model, val_loader, mode, device)
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if val_metrics["accuracy"] > best:
            best, best_state = val_metrics["accuracy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{cfg['training']['epochs']}: train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
    model.load_state_dict(best_state)
    metrics, pairs = evaluate(model, test_loader, mode, device, collect_pairs=True)
    metrics.update({"method": mode, "seed": cfg["experiment"]["seed"], "best_val_acc": best, "test_acc": metrics["accuracy"], "random_backbone": args.allow_random_backbone})
    out = Path(cfg["output"]["root"]) / cfg["experiment"]["name"]
    save_run(out, cfg, history, metrics, model, pairs)
    print(yaml.safe_dump(metrics, sort_keys=False))


if __name__ == "__main__": main()
