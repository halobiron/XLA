"""Fixed-budget, validation-only CUB warm-start check and frozen-context oracle."""
from __future__ import annotations

import argparse
import copy
import csv
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

import run_cub_pilot as pilot
from tacsx.models.selector import TACSSelector
from tacsx.models.tacs import TACSClassifier

PARENT = Path("outputs/cub200_fullpool_pilot_s42_v1")


def plan():
    return {"seed": 42, "parent": str(PARENT), "source_checkpoint_epoch": 3,
            "additional_query_only_epochs": 2, "continuation_epochs": 4,
            "initialization": "identical epoch5 supervised task weights for both continuation arms",
            "optimizer": "fresh AdamW and cosine2 for warm-up; fresh matched AdamW and cosine4 for continuation",
            "training": {k: pilot.plan()[k] for k in ("batch_size", "image_size", "lr_backbone", "lr_head", "lr_selector", "weight_decay", "tau", "lambda_policy")},
            "candidate_pool": "same fixed1080 training images; full-pool Gumbel/REINFORCE, no shortlist in training",
            "candidate_cache": "stop-gradient shared encoder; refresh every50 steps and before validation",
            "paired_queries": "same per-query/epoch deterministic transforms and DataLoader order",
            "oracle": "frozen parent epoch3; top16 by existing selector; labels only for diagnostic bounds",
            "primary_endpoint": "TACS minus independently continued No-Context at fixed continuation epoch4",
            "checkpoint_selection": "fixed final epoch, no early stopping or best-validation selection",
            "uncertainty": "descriptive one-seed paired bootstrap95%; reused validation; not confirmatory",
            "test_access": False,
            "deviations": ["quick check, not proposed20-epoch continuation or paper100 epochs",
                           "warm-up reuses parent epoch3 and resets optimizer; not an uninterrupted5-epoch run",
                           "224px, batch16, one seed; horizontal pair RoPE; cosine scores scaled1/.07",
                           "stop-gradient cached candidate features; lambda1 follows AGENTS.md rather than paper0.5"]}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def oracle_readouts(pair_logits, query_logits, labels):
    """Keep exact accuracy existence bound distinct from minimum-CE selection."""
    b, k, _ = pair_logits.shape
    losses = F.cross_entropy(pair_logits.flatten(0, 1), labels.repeat_interleave(k), reduction="none").reshape(b, k)
    row = torch.arange(b, device=labels.device)
    return {"selector_argmax": pair_logits[:, 0].argmax(-1).eq(labels),
            "internal_query_only": query_logits.argmax(-1).eq(labels),
            "mean_probability16": pair_logits.softmax(-1).mean(1).argmax(-1).eq(labels),
            "oracle_min_ce16": pair_logits[row, losses.argmin(1)].argmax(-1).eq(labels),
            "oracle_any_correct16": pair_logits.argmax(-1).eq(labels[:, None]).any(1)}


def verify_reference(rows, ids, labels, predictions):
    assert [int(r["query_id"]) for r in rows] == ids.tolist(), "Reference query order differs"
    assert [int(r["query_label"]) for r in rows] == labels.tolist(), "Reference labels differ"
    expected = torch.tensor([int(r["prediction"]) for r in rows])
    assert torch.equal(predictions.cpu(), expected), "Reference predictions differ"


def register(output):
    if (output / "registration.json").exists():
        if json.loads((output / "registration.json").read_text())["plan"] != plan():
            raise ValueError("Registered plan changed")
        return
    if any(output.iterdir()):
        raise ValueError("Register into an empty output directory")
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True)
    for name in (Path(__file__).name, "run_cub_pilot.py"):
        shutil.copy2(ROOT / "scripts" / name, execution / "scripts" / name)
    shutil.copytree(ROOT / "src/tacsx", execution / "src/tacsx", ignore=shutil.ignore_patterns("__pycache__"))
    inputs = [ROOT / PARENT / "registration.json", ROOT / PARENT / "split_audit.json"]
    inputs += [ROOT / PARENT / a / "checkpoint_last.pt" for a in pilot.ARMS]
    inputs += [ROOT / PARENT / a / "validation_epoch_003.csv" for a in pilot.ARMS]
    inputs += [ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    inputs += [ROOT / "data/CUB_200_2011" / n for n in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]
    pilot.dump(output / "registration.json", {"plan": plan(),
               "input_sha256": {str(p.relative_to(ROOT)): pilot.digest(p) for p in inputs},
               "source_sha256": {str(p.relative_to(execution)): pilot.digest(p) for p in execution.rglob("*.py")}})
    print(f"REGISTERED {output}", flush=True)


def protect_test_images():
    """Fail on any attempt to open an official-test image, including direct PIL reads."""
    data = ROOT / "data/CUB_200_2011"
    paths = {int(i): p for i, p in (line.split(maxsplit=1) for line in (data / "images.txt").read_text().splitlines())}
    allowed = {(data / "images" / paths[int(i)]).resolve() for i, flag in
               (line.split() for line in (data / "train_test_split.txt").read_text().splitlines()) if int(flag)}
    original = Image.open
    audit = {"training_image_opens": 0, "forbidden_image_attempts": 0}
    image_root = (data / "images").resolve()

    def guarded(fp, *args, **kwargs):
        if isinstance(fp, (str, os.PathLike)):
            path = Path(fp).resolve()
            if path.is_relative_to(image_root):
                if path not in allowed:
                    audit["forbidden_image_attempts"] += 1
                    raise RuntimeError("Official test image access is prohibited")
                audit["training_image_opens"] += 1
        return original(fp, *args, **kwargs)

    Image.open = guarded
    return original, audit


@torch.inference_mode()
def diagnose(output, arms, val, pixels, pool_ids, identities, reference_run=None, reference_epoch=3):
    model = arms["full_tacs"].eval()
    baseline = arms["no_context"].eval()
    features = pilot.candidate_features(model, pixels)
    all_pairs, all_query, all_base, all_labels, all_ids, all_candidates = [], [], [], [], [], []
    for step, batch in enumerate(val):
        q, y = batch["query"].cuda(), batch["label"].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            scores = pilot.utility(model, q, features, batch["query_id"], pool_ids, identities)
            positions = scores.topk(16, dim=-1).indices
            logits = torch.stack([model.task_model(q, pixels[positions[:, j]]).float() for j in range(16)], 1)
            query, base = model.task_model(q).float(), baseline.task_model(q).float()
        all_pairs.append(logits.cpu()); all_query.append(query.cpu()); all_base.append(base.cpu())
        all_labels.append(y.cpu()); all_ids.append(batch["query_id"]); all_candidates.append(pool_ids[positions].cpu())
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == len(val):
            print(f"ORACLE {step+1}/{len(val)}", flush=True)
    pairs, query, base = torch.cat(all_pairs), torch.cat(all_query), torch.cat(all_base)
    labels, ids = torch.cat(all_labels), torch.cat(all_ids)
    correct = oracle_readouts(pairs, query, labels)
    nc = base.argmax(-1).eq(labels)
    reference = ROOT / PARENT if reference_run is None else Path(reference_run)
    for arm, actual in (("no_context", base.argmax(-1)), ("full_tacs", pairs[:, 0].argmax(-1))):
        old = list(csv.DictReader((reference / arm / f"validation_epoch_{reference_epoch:03d}.csv").open()))
        verify_reference(old, ids, labels, actual)
    rows = []
    for name, values in {"independent_no_context": nc, **correct}.items():
        rows.append({"mode": name, "accuracy": values.double().mean().item(),
                     **pilot.paired_stats(nc.numpy(), values.numpy()),
                     "fixed": int((~nc & values).sum()), "broken": int((nc & ~values).sum())})
    result = {"scope": "frozen-checkpoint diagnostic; oracle selections are not deployable", "count": len(labels),
              "reference_run": str(reference), "reference_epoch": reference_epoch,
              "metrics": rows, "parent_predictions_reconstructed_exactly": True,
              "nc_errors_repairable_with_any16": int((~nc & correct["oracle_any_correct16"]).sum()),
              "nc_correct_but_all16_wrong": int((nc & ~correct["oracle_any_correct16"]).sum()), "test_access": False}
    pilot.dump(output / "oracle.json", result)
    write_csv(output / "oracle.csv", rows)
    torch.save({"pair_logits": pairs, "query_logits": query, "baseline_logits": base, "labels": labels,
                "query_ids": ids, "candidate_ids": torch.cat(all_candidates)}, output / "oracle_predictions.pt")
    print(f"ORACLE_RESULT {json.dumps(result)}", flush=True)
    return result


def train_arm(output, model, arm, train, val, pixels, pool_ids, identities, epochs, epoch_offset):
    optimizer = torch.optim.AdamW(pilot.param_groups(model, plan()["training"]), weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    folder = output / arm
    folder.mkdir()
    for epoch in range(1, epochs + 1):
        absolute_epoch = epoch_offset + epoch
        pilot.set_seed(42 + absolute_epoch * 1009)
        train.dataset.set_epoch(absolute_epoch)
        train.generator.manual_seed(42 + absolute_epoch * 101)
        pilot.status(output, state="running", arm=arm, epoch=epoch, absolute_epoch=absolute_epoch)
        mode = "full_tacs" if arm == "full_tacs" else "no_context"
        tm, _, _ = pilot.epoch_run(model, train, mode, pixels, pool_ids, identities, optimizer, scaler)
        vm, correct, pairs = pilot.epoch_run(model, val, mode, pixels, pool_ids, identities)
        scheduler.step()
        history.append({"epoch": epoch, "absolute_epoch": absolute_epoch, "train": tm, "validation": vm})
        pilot.dump(folder / "history.json", history)
        pilot.dump(folder / "config.json", plan())
        write_csv(folder / f"validation_epoch_{epoch:03d}.csv", pairs)
        torch.save({"model": model.state_dict(), "history": history, "config": plan(),
                    "registration_sha256": pilot.digest(output / "registration.json")}, folder / "checkpoint_last.pt")
        print(f"FINISH {arm} {json.dumps(history[-1])}", flush=True)
    return correct, [(p["query_id"], p["query_label"]) for p in pairs]


def run(output):
    record = json.loads((output / "registration.json").read_text())
    for relative, expected in record["source_sha256"].items():
        assert pilot.digest(output / "execution" / relative) == expected
    for relative, expected in record["input_sha256"].items():
        assert pilot.digest(ROOT / relative) == expected
    if any((output / a).exists() for a in ("warmup", *pilot.ARMS)):
        raise ValueError("Do not silently restart a fixed-budget run")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pilot.set_seed(42)
    pilot.dump(output / "environment.json", {"python": sys.executable, "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "precision": "fp16 AMP, TF32 disabled"})
    original_open, image_audit = protect_test_images()
    started = time.perf_counter()
    try:
        pilot.status(output, state="preparing")
        train, val, pixels, pool_ids, identities = pilot.prepare(output)
        old_split = json.loads((ROOT / PARENT / "split_audit.json").read_text())
        new_split = json.loads((output / "split_audit.json").read_text())
        for key in ("train_ids", "validation_ids", "pool_ids", "training_image_sha256"):
            assert old_split[key] == new_split[key], f"Parent split differs: {key}"
        arms = pilot.models()
        for arm, model in arms.items():
            saved = torch.load(ROOT / PARENT / arm / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            assert len(saved["history"]) == 3
            model.load_state_dict(saved["model"])
            del saved
        pilot.status(output, state="running", stage="frozen_oracle")
        diagnose(output, arms, val, pixels, pool_ids, identities)
        del arms["full_tacs"]
        torch.cuda.empty_cache()
        train_arm(output, arms["no_context"], "warmup", train, val, pixels, pool_ids, identities, 2, 3)
        task = copy.deepcopy(arms["no_context"].task_model)
        arms["full_tacs"] = TACSClassifier(TACSSelector(task.backbone, projection=False), task, tau=.1, lambda_policy=1.).cuda()
        assert all(torch.equal(a, b) for a, b in zip(task.state_dict().values(), arms["no_context"].task_model.state_dict().values()))
        assert arms["full_tacs"].selector.encoder is task.backbone
        pilot.dump(output / "initialization_audit.json", {"identical_supervised_task_weights": True,
                   "shared_trainable_selector_task_adapter": True, "warmup_absolute_epoch": 5})
        outcomes, orders = {}, {}
        for arm in pilot.ARMS:
            outcomes[arm], orders[arm] = train_arm(output, arms[arm], arm, train, val, pixels, pool_ids, identities, 4, 5)
        assert orders["no_context"] == orders["full_tacs"]
        stats = pilot.paired_stats(outcomes["no_context"], outcomes["full_tacs"])
        result = {"scope": "one-seed quick warm-start check; reused validation only", "continuation_epoch": 4,
                  "count": len(orders["no_context"]), "accuracy": {a: sum(v) / len(v) for a, v in outcomes.items()},
                  **stats, "fixed": sum(b and not a for a, b in zip(outcomes["no_context"], outcomes["full_tacs"])),
                  "broken": sum(a and not b for a, b in zip(outcomes["no_context"], outcomes["full_tacs"])),
                  "test_pixels_read": False, "elapsed_seconds": time.perf_counter() - started}
        pilot.dump(output / "comparison.json", result)
        write_csv(output / "comparison.csv", [{"arm": a, "validation_accuracy": v, "continuation_epoch": 4}
                                               for a, v in result["accuracy"].items()])
        for relative, expected in record["input_sha256"].items():
            assert pilot.digest(ROOT / relative) == expected, f"Original input changed: {relative}"
        pilot.dump(output / "integrity_audit.json", {"original_inputs_unchanged": True, **image_audit, "test_pixels_read": False})
        pilot.status(output, state="complete", **result)
        print(f"COMPARISON {json.dumps(result)}", flush=True)
    finally:
        Image.open = original_open
        pilot.dump(output / "image_access_audit.json", image_audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("register", "run"), default="register")
    p.add_argument("--output", default="outputs/cub200_quick_warmstart_s42_v1")
    args = p.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "register":
        register(output)
    else:
        if Path(__file__).resolve() != output / "execution/scripts/run_cub_quick_check.py":
            raise ValueError("Execute the registered source copy with TACS_PROJECT_ROOT set")
        try:
            run(output)
        except Exception as error:
            pilot.status(output, state="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
