"""Matched query-patch/context-patch heads on the unchanged CUB No-Context encoder."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get("TACS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from PIL import Image

import learn_cub_selector as selector
from tacsx.models.patch_context import PatchContextClassifier

quick, pilot = selector.quick, selector.pilot
CACHE = Path("outputs/cub200_selector_learning_s42_v1")
PARENT = Path("outputs/cub200_quick_warmstart_s42_v1")
ARMS = ("query_patches", "context_patches")


def plan():
    return {"scope": "exploratory TACS-X architecture extension; not original TACS classification",
            "seed": 42, "encoder": "unchanged final No-Context, frozen; native per-image patch tokens",
            "initialization": "both heads exactly equal original query logits; zero output correction",
            "arms": list(ARMS), "matching": "same head parameters, query views, order,20 epochs and optimizer",
            "context": "one original TACS argmax context; fixed from previous top16 proposal cache; pool1080 train images",
            "fit": "2400 train query identities, clean+fixed augmentation;600 train identities select epochs",
            "epochs": 20, "batch_size": 128, "lr": .0003, "weight_decay": .01,
            "selection": "each arm best train-holdout accuracy, NLL tie-break; epoch0 allowed, before final validation",
            "validation": "all594 reused validation queries; primary context minus matched query-patch control",
            "uncertainty": "descriptive paired bootstrap95%, one seed, adaptive research/reused validation; no confirmatory claim",
            "test_access": False, "deviations": ["frozen encoder; patch-attention residual rather than pre-block two-image concatenation",
                "fixed selector and no hybrid Gumbel/policy update; small attention head trained on cached features"]}


def register(output):
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    source = ROOT / CACHE / "execution"
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True)
    shutil.copytree(source / "src", execution / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "src/tacsx/models/patch_context.py", execution / "src/tacsx/models/patch_context.py")
    for name in ("learn_cub_selector.py", "run_cub_quick_check.py", "run_cub_pilot.py"):
        shutil.copy2(source / "scripts" / name, execution / "scripts" / name)
    shutil.copy2(__file__, execution / "scripts" / Path(__file__).name)
    inputs = [ROOT / CACHE / n for n in ("registration.json", "selector_split.json", "pool.pt", "fit.pt", "holdout.pt", "validation.pt")]
    inputs += [ROOT / PARENT / "no_context" / n for n in ("checkpoint_last.pt", "validation_epoch_004.csv")]
    inputs += [ROOT / PARENT / "split_audit.json", ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    inputs += [ROOT / "data/CUB_200_2011" / n for n in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]
    pilot.dump(output / "registration.json", {"plan": plan(),
        "input_sha256": {str(p.relative_to(ROOT)): pilot.digest(p) for p in inputs},
        "source_sha256": {str(p.relative_to(execution)): pilot.digest(p) for p in execution.rglob("*.py")}})
    print(f"REGISTERED {output}", flush=True)


@torch.inference_mode()
def encode(output, name, model, source, ids, epoch=0):
    target = output / f"{name}.pt"
    if target.exists():
        return
    cls, patches, logits, labels, query_ids = [], [], [], [], []
    loader = selector.query_loader(source, ids, epoch)
    for step, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = model.backbone.prepare_tokens(batch["query"].cuda())
            features = model.backbone.forward_blocks(tokens.tokens, tokens.grid_size)
            base = model.head(features[:, 0]).float()
        count = tokens.grid_size[0] * tokens.grid_size[1]
        cls.append(features[:, 0].half().cpu())
        patches.append(features[:, -count:].half().cpu())
        logits.append(base.cpu()); labels.append(batch["label"]); query_ids.append(batch["query_id"])
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == len(loader):
            print(f"ENCODE {name} {step+1}/{len(loader)}", flush=True)
            pilot.status(output, state="running", stage="encode", split=name, batch=step+1, total=len(loader))
    torch.save({"cls": torch.cat(cls), "patches": torch.cat(patches), "baseline_logits": torch.cat(logits),
                "labels": torch.cat(labels), "query_ids": torch.cat(query_ids)}, target)


def load_data(output, name):
    return {k: v.cuda() for k, v in torch.load(output / f"{name}.pt", map_location="cpu", weights_only=False).items()}


def forward(model, data, indices, pool):
    context = pool[data["context_positions"][indices]] if model.use_context else None
    return model(data["cls"][indices], data["patches"][indices], data["baseline_logits"][indices], context)


@torch.no_grad()
def evaluate(model, data, pool):
    model.eval()
    logits = torch.cat([forward(model, data, i, pool).cpu() for i in torch.arange(len(data["labels"]), device="cuda").split(128)])
    y = data["labels"].cpu()
    return {"accuracy": float(logits.argmax(-1).eq(y).double().mean()), "nll": float(F.cross_entropy(logits, y))}, logits


def run(output):
    record = json.loads((output / "registration.json").read_text())
    for rel, digest in record["source_sha256"].items():
        assert pilot.digest(output / "execution" / rel) == digest
    for rel, digest in record["input_sha256"].items():
        assert pilot.digest(ROOT / rel) == digest
    if (output / "result.json").exists():
        raise ValueError("Completed experiment already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    pilot.set_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    original_open, image_audit = quick.protect_test_images()
    try:
        train, val, pixels, pool_ids, identities = pilot.prepare(output)
        split = json.loads((ROOT / CACHE / "selector_split.json").read_text())
        original_pool = torch.load(ROOT / CACHE / "pool.pt", map_location="cpu", weights_only=False)
        assert torch.equal(pool_ids.cpu(), original_pool["ids"])
        model = pilot.models()["no_context"]
        saved = torch.load(ROOT / PARENT / "no_context/checkpoint_last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        model.eval().requires_grad_(False)
        task = model.task_model
        encode(output, "pool", task, train.dataset.candidate_base, pool_ids.cpu().tolist())
        encode(output, "fit_clean", task, train.dataset.candidate_base, split["fit"])
        encode(output, "fit_augmented", task, train.dataset.base.dataset, split["fit"], epoch=101)
        encode(output, "holdout", task, train.dataset.candidate_base, split["holdout"])
        pool = load_data(output, "pool")["patches"]
        clean, augmented = load_data(output, "fit_clean"), load_data(output, "fit_augmented")
        fit = {k: torch.cat((clean[k], augmented[k])) for k in clean}
        del clean, augmented
        holdout = load_data(output, "holdout")
        for name, data in (("fit", fit), ("holdout", holdout)):
            cache = torch.load(ROOT / CACHE / f"{name}.pt", map_location="cpu", weights_only=False)
            assert torch.equal(data["query_ids"].cpu(), cache["query_ids"])
            data["context_positions"] = cache["positions"][:, 0].cuda()
            assert not (data["query_ids"] == pool_ids[data["context_positions"]]).any()
        pilot.set_seed(42)
        base = PatchContextClassifier(task.head.weight, use_context=False).cuda()
        heads = {"query_patches": base, "context_patches": copy.deepcopy(base)}
        heads["context_patches"].use_context = True
        for head in heads.values():
            test = forward(head.eval(), holdout, torch.arange(16, device="cuda"), pool)
            assert torch.equal(test, holdout["baseline_logits"][:16].float())
        choices = []
        for arm, head in heads.items():
            optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            folder = output / arm
            folder.mkdir(exist_ok=True)
            history, best = [], None
            for epoch in range(21):
                pilot.set_seed(42 + epoch)
                if epoch:
                    head.train()
                    for indices in torch.randperm(len(fit["labels"]), device="cuda").split(128):
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(forward(head, fit, indices, pool), fit["labels"][indices])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                    scheduler.step()
                metrics, _ = evaluate(head, holdout, pool)
                row = {"epoch": epoch, "holdout": metrics}
                history.append(row)
                if best is None or (metrics["accuracy"], -metrics["nll"]) > (best["holdout"]["accuracy"], -best["holdout"]["nll"]):
                    best = row
                    torch.save({"model": head.state_dict(), "best": best, "arm": arm}, folder / "checkpoint_best.pt")
                if epoch % 5 == 0:
                    print(f"FIT {arm} {json.dumps(row)}", flush=True)
                    pilot.status(output, state="running", stage="fit", arm=arm, epoch=epoch)
            pilot.dump(folder / "history.json", history)
            choices.append({"arm": arm, "best": best, "checkpoint_sha256": pilot.digest(folder / "checkpoint_best.pt")})
        pilot.dump(output / "selection.json", {"choices": choices, "validation_used": False})
        encode(output, "validation", task, val.dataset.candidate_base, val.dataset.query_ids)
        validation = load_data(output, "validation")
        old = torch.load(ROOT / CACHE / "validation.pt", map_location="cpu", weights_only=False)
        assert torch.equal(old["query_ids"], validation["query_ids"].cpu())
        assert torch.equal(old["baseline_logits"].argmax(-1), validation["baseline_logits"].cpu().argmax(-1))
        validation["context_positions"] = old["positions"][:, 0].cuda()
        predictions = {"original_no_context": validation["baseline_logits"].cpu()}
        rows, labels = [], validation["labels"].cpu()
        for arm, head in heads.items():
            state = torch.load(output / arm / "checkpoint_best.pt", map_location="cpu", weights_only=False)
            head.load_state_dict(state["model"])
            _, predictions[arm] = evaluate(head, validation, pool)
        reference = predictions["original_no_context"].argmax(-1).eq(labels)
        for arm, logits in predictions.items():
            correct = logits.argmax(-1).eq(labels)
            rows.append({"arm": arm, "accuracy": float(correct.double().mean()), "nll": float(F.cross_entropy(logits, labels)),
                         **pilot.paired_stats(reference.numpy(), correct.numpy())})
        control = predictions["query_patches"].argmax(-1).eq(labels)
        context = predictions["context_patches"].argmax(-1).eq(labels)
        result = {"scope": plan()["scope"], "count": len(labels), "metrics": rows,
                  "context_minus_query_patch_control": pilot.paired_stats(control.numpy(), context.numpy()),
                  "selections": choices, "test_access": False, "elapsed_seconds": time.perf_counter()-started}
        pilot.dump(output / "result.json", result)
        quick.write_csv(output / "comparison.csv", rows)
        torch.save({"logits": predictions, "labels": labels, "query_ids": validation["query_ids"].cpu(),
                    "candidate_ids": pool_ids[validation["context_positions"]].cpu()}, output / "predictions.pt")
        for rel, digest in record["input_sha256"].items():
            assert pilot.digest(ROOT / rel) == digest
        pilot.dump(output / "integrity_audit.json", {"original_inputs_unchanged": True, "query_encoder_frozen": True,
                   "initialization_matches_baseline_exactly": True, **image_audit, "test_pixels_read": False})
        pilot.status(output, state="complete", elapsed_seconds=result["elapsed_seconds"], test_pixels_read=False)
        print(f"RESULT {json.dumps(result)}", flush=True)
    finally:
        Image.open = original_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), default="register")
    parser.add_argument("--output", default="outputs/cub200_patch_context_s42_v1")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "register":
        register(output)
    else:
        if Path(__file__).resolve() != output / "execution/scripts/check_cub_patch_context.py":
            raise ValueError("Execute registered source with TACS_PROJECT_ROOT set")
        try:
            run(output)
        except Exception as error:
            pilot.status(output, state="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
