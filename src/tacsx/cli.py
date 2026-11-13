from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from torch import optim

from .data import make_cifar100_loaders
from .engine import cache_dino_candidate_embeddings, evaluate, save_run, set_seed, train_epoch
from .models.dinov3_adapter import DINOv3Adapter
from .models.dual_input_vit import DualInputViT
from .models.selector import TACSSelector
from .models.tacs import TACSClassifier


REQUIRED_MODES = {"no_context", "random_context", "dino_similarity", "gumbel_only", "policy_only", "full_tacs", "topk_tacs", "adaptive_topk_tacs"}


def _freeze(adapter, freeze: bool, last_n: int):
    """Set backbone trainability.

    * ``unfreeze_last_n_blocks > 0``: only the last N blocks (+ final norm) train,
      whatever ``freeze`` says. Previously ``freeze: false`` returned early and
      silently fine-tuned the *whole* backbone, ignoring N.
    * ``freeze: true`` and N == 0: fully frozen.
    * ``freeze: false`` and N == 0: full fine-tune.
    """
    if not freeze and not last_n:
        return
    for p in adapter.parameters(): p.requires_grad = False
    if last_n:
        for block in list(adapter.model.blocks)[-last_n:]:
            for p in block.parameters(): p.requires_grad = True
        for p in adapter.model.norm.parameters(): p.requires_grad = True


def param_groups(model, training_cfg: dict):
    """Separate learning rates: backbone (lr_backbone), selector head (lr_selector), task head (lr_head).

    Previously every trainable parameter used lr_head, so an unfrozen DINOv3 was
    updated at 3e-4 and lost its pretrained features.
    """
    groups = {"backbone": [], "selector": [], "head": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("task_model.backbone.") or name.startswith("selector.encoder."):
            groups["backbone"].append(p)
        elif name.startswith("selector."):
            groups["selector"].append(p)
        else:
            groups["head"].append(p)
    lrs = {"backbone": float(training_cfg.get("lr_backbone", 1e-5)),
           "selector": float(training_cfg.get("lr_selector", 1e-4)),
           "head": float(training_cfg["lr_head"])}
    return [{"params": ps, "lr": lrs[k], "name": k} for k, ps in groups.items() if ps]


def build_model(cfg, allow_random_backbone: bool, include_selector: bool = True):
    back = cfg["backbone"]
    weights = back.get("weights")
    if not weights and not allow_random_backbone:
        raise ValueError("DINOv3 weights are required. Set backbone.weights via official access, or explicitly pass --allow-random-backbone for a smoke test.")
    kwargs = {"weights": weights} if weights else {"pretrained": False}
    task_adapter = DINOv3Adapter(back["repo_dir"], **kwargs)
    _freeze(task_adapter, back.get("freeze", False), int(back.get("unfreeze_last_n_blocks", 0)))
    if not include_selector:
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
    p.add_argument("--candidates-per-query", type=int, help="override sampled candidates per query; recorded as a practical deviation")
    p.add_argument("--epochs", type=int, help="override training epochs; recorded as a practical deviation")
    p.add_argument("--batch-size", type=int, help="override batch size; recorded as a practical deviation")
    p.add_argument("--data-root", help="directory containing cifar-100-python; useful for a mounted Kaggle Dataset")
    p.add_argument("--download-data", action="store_true", help="explicitly download CIFAR-100 when it is not mounted locally")
    p.add_argument("--allow-random-backbone", action="store_true", help="only for smoke tests; logs a fidelity deviation")
    p.add_argument("--topk-k", type=int, help="override topk.k for topk_tacs")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.mode: cfg["experiment"]["mode"] = args.mode
    if args.name: cfg["experiment"]["name"] = args.name
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.download_data:
        cfg["data"]["download"] = True
        cfg["experiment"].setdefault("deviations", []).append(
            "CIFAR-100 downloaded at runtime instead of using a mounted local dataset")
    if args.candidates_per_query is not None:
        if args.candidates_per_query < 1:
            p.error("--candidates-per-query must be at least 1")
        original = cfg["data"]["candidates_per_query"]
        cfg["data"]["candidates_per_query"] = args.candidates_per_query
        cfg["experiment"].setdefault("deviations", []).append(
            f"candidates_per_query overridden from {original} to {args.candidates_per_query}")
    for key, value, flag in (("epochs", args.epochs, "--epochs"), ("batch_size", args.batch_size, "--batch-size")):
        if value is not None:
            if value < 1:
                p.error(f"{flag} must be at least 1")
            original = cfg["training"][key]
            cfg["training"][key] = value
            cfg["experiment"].setdefault("deviations", []).append(
                f"{key} overridden from {original} to {value}")
    if args.topk_k is not None:
        if args.topk_k < 1:
            p.error("--topk-k must be at least 1")
        cfg.setdefault("topk", {})["k"] = args.topk_k
    mode = cfg["experiment"]["mode"]
    include_context = mode != "no_context"
    include_selector = mode not in {"no_context", "random_context", "dino_similarity"}
    set_seed(int(cfg["experiment"]["seed"]))
    device = torch.device(args.device)
    cache_dino = mode == "dino_similarity"
    train_loader, val_loader, test_loader, _ = make_cifar100_loaders(
        cfg["data"]["root"], cfg["data"]["image_size"], cfg["data"]["candidate_pool_ratio"],
        cfg["data"]["candidates_per_query"], cfg["training"]["batch_size"], cfg["data"]["num_workers"], cfg["experiment"]["seed"], include_context=include_context,
        validation_ratio=float(cfg["data"].get("validation_ratio", 0.1)), return_candidate_images=not cache_dino,
        download=bool(cfg["data"].get("download", False)))
    model = build_model(cfg, args.allow_random_backbone, include_selector=include_selector).to(device)
    dino_cache = None
    if cache_dino:
        if any(p.requires_grad for p in model.task_model.backbone.parameters()):
            raise ValueError("DINO embedding cache requires a fully frozen task backbone")
        print("Caching fixed-pool DINO embeddings once; subsequent epochs reuse them.", flush=True)
        dino_cache = cache_dino_candidate_embeddings(model.task_model.backbone, train_loader.dataset, device)
    groups = param_groups(model, cfg["training"])
    for g in groups:
        print(f"param group {g['name']}: {sum(p.numel() for p in g['params']):,} params, lr={g['lr']:g}", flush=True)
    optimizer = optim.AdamW(groups, weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = int(cfg["training"]["epochs"])
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cfg["training"].get("scheduler") == "cosine" else None)
    use_amp = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    history, best, best_state = [], -1.0, None
    for epoch in range(int(cfg["training"]["epochs"])):
        train_loader.dataset.set_epoch(epoch)
        train_metrics = train_epoch(model, train_loader, optimizer, mode, device, model.lambda_policy, scaler, show_progress=True, dino_cache=dino_cache)
        val_metrics, _ = evaluate(model, val_loader, mode, device, dino_cache=dino_cache)
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        if val_metrics["accuracy"] > best:
            best, best_state = val_metrics["accuracy"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{cfg['training']['epochs']}: train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
    model.load_state_dict(best_state)
    metrics, pairs = evaluate(model, test_loader, mode, device, collect_pairs=True, dino_cache=dino_cache)
    metrics.update({"method": mode, "seed": cfg["experiment"]["seed"], "best_val_acc": best, "test_acc": metrics["accuracy"], "random_backbone": args.allow_random_backbone,
                    # v2: per-group learning rates, cosine schedule, correct partial unfreezing.
                    # Runs without this key used one lr for everything and no scheduler; do not mix them.
                    "trainer_version": 2})
    out = Path(cfg["output"]["root"]) / cfg["experiment"]["name"]
    save_run(out, cfg, history, metrics, model, pairs)
    print(yaml.safe_dump(metrics, sort_keys=False))


if __name__ == "__main__": main()
