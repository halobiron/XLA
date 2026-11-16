"""One-factor seed42 warm-up ablation against immutable four-epoch controls.

Use the previous query-noise runner's frozen implementation verbatim. Epoch1
optimizes task loss alone; epochs2-4 optimize task+policy loss. Policy forwards
remain enabled in epoch1 for reward measurement and identical RNG consumption.
No new augmentation, candidate pool, checkpoint selection, test access or seed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(os.environ.get("TACS_QUERY_NOISE_ROOT", Path(__file__).resolve().parents[1]))
PARENT = ROOT / "outputs/query_noise_clean_context_3seed_v1"
WEIGHTS = (0., 1., 1., 1.)
SEED = 42


def policy_weight(epoch):
    if not 1 <= epoch <= len(WEIGHTS):
        raise ValueError("Epoch must be between 1 and 4")
    return WEIGHTS[epoch - 1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_status(output, **fields):
    save_json({"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}, output / "status.json")


def registered_controls():
    return [PARENT / "seed_42" / arm / name for arm in ("no_context", "full_tacs")
            for name in ("checkpoint_last.pt", "validation_final.pt", "metrics.json", "history.json")]


def register(output):
    path = output / "registration.json"
    if path.exists():
        return json.loads(path.read_text())
    if (output / "staged_tacs").exists():
        raise ValueError("Cannot register after starting training")
    parent = json.loads((PARENT / "registration.json").read_text())
    execution = output / "execution" / Path(__file__).name
    execution.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, execution)
    registration = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "query-noise clean-context staged-policy warm-up, one seed, fixed epoch4",
        "seed": SEED, "policy_weight_by_epoch": list(WEIGHTS), "epochs": 4,
        "parent_run": str(PARENT.relative_to(ROOT)), "parent_registration_sha256": digest(PARENT / "registration.json"),
        "base_specification": copy.deepcopy(parent["specification"]),
        "only_training_change": "policy coefficient0 in epoch1, coefficient1 in epochs2-4; task objective always active",
        "warmup_instrumentation": "same full forward/reward measurement and RNG draws as control; epoch1 policy contributes zero gradient",
        "checkpoint_selection": "fixed final epoch4, no early stopping or validation selection",
        "primary_comparison": "staged TACS minus always-on Full TACS, seed42, mean top1 across sigma .08/.16",
        "additional_comparison": "staged TACS minus same No-Context seed42",
        "uncertainty": "descriptive 95% paired query-cluster bootstrap, 10000 draws; one seed, reused validation",
        "success_interpretation": "exploratory signal only; both point differences must be positive to merit further consideration",
        "automatic_followup": False, "test_access": False,
        "source_sha256": digest(execution),
        "control_sha256": {str(p.relative_to(ROOT)): digest(p) for p in registered_controls()},
        "deviations": ["one warm-up epoch with lambda0 intentionally differs from paper's fixed lambda1",
                       "all parent mini-reproduction deviations retained; not an exact reproduction or confirmation"],
    }
    save_json(registration, path)
    print(f"Registered warm-up experiment: {path}", flush=True)
    return registration


def load_runner():
    # Import under the real module name so Windows DataLoader workers can pickle
    # NoisyQueries. Never edit/rewrite the original execution snapshot.
    sys.path.insert(0, str(PARENT / "execution/scripts"))
    import run_query_noise_pilot as runner
    if Path(runner.__file__).resolve() != (PARENT / "execution/scripts/run_query_noise_pilot.py").resolve():
        raise ValueError("Expected original frozen query-noise runner")
    return runner


def verify(output, registration):
    if digest(PARENT / "registration.json") != registration["parent_registration_sha256"]:
        raise ValueError("Parent protocol changed")
    if digest(output / "execution" / Path(__file__).name) != registration["source_sha256"]:
        raise ValueError("Registered warm-up execution changed")
    parent = json.loads((PARENT / "registration.json").read_text())
    for relative, expected in parent["source_sha256"].items():
        if digest(PARENT / "execution" / relative) != expected:
            raise ValueError(f"Frozen parent source changed: {relative}")
    for relative, expected in {**parent["input_sha256"], **registration["control_sha256"]}.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Registered input/control changed: {relative}")
    return parent


def comparison(output, registration, runner):
    import numpy as np
    import torch
    from probe_tacs_robustness import paired_ci
    paths = {arm: PARENT / "seed_42" / arm / "validation_final.pt" for arm in ("no_context", "full_tacs")}
    paths["staged_tacs"] = output / "staged_tacs/validation_final.pt"
    data = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    reference = data["no_context"]
    rows, correct = [], {}
    for arm, saved in data.items():
        expected_lock = digest(output / "registration.json") if arm == "staged_tacs" else registration["parent_registration_sha256"]
        if saved["epoch"] != 4 or saved["seed"] != SEED or saved["registration_sha256"] != expected_lock:
            raise ValueError("Checkpoint seed/epoch/protocol mismatch")
        if not torch.equal(saved["query_ids"], reference["query_ids"]) or not torch.equal(saved["labels"], reference["labels"]):
            raise ValueError("Unpaired validation results")
        correct[arm] = saved["probabilities"].argmax(-1).eq(saved["labels"][:, None]).float().numpy()
        accuracy = correct[arm].mean(0)
        rows.append({"seed": SEED, "arm": arm, "epoch": 4, "accuracy_sigma_008": float(accuracy[0]),
                     "accuracy_sigma_016": float(accuracy[1]), "primary_mean_accuracy": float(accuracy.mean())})
    differences = {arm: paired_ci((correct["staged_tacs"] - correct[arm]).mean(1) * 100)
                   for arm in ("full_tacs", "no_context")}
    for relative, expected in registration["control_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Original control changed during experiment")
    result = {"scope": "exploratory one-seed warm-up ablation, validation only", "metrics": rows,
              "staged_minus_control_pp": differences,
              "positive_point_estimates_vs_both": bool(all(d["difference"] > 0 for d in differences.values())),
              "promotion_gate_passed": False, "automatic_followup": False, "test_access": False,
              "controls_unchanged": True, "registration_sha256": digest(output / "registration.json")}
    save_json(result, output / "comparison.json")
    runner.write_csv(rows, output / "comparison.csv")
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(output, registration, stage):
    import torch
    parent = verify(output, registration)
    if Path(__file__).resolve() != (output / "execution" / Path(__file__).name).resolve():
        raise ValueError("Run the registered execution copy with TACS_QUERY_NOISE_ROOT set")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe; CUDA required")
    runner = load_runner()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_status(output, state="running", stage="data_audit")
    source, train_ids, val_ids, identities, pool = runner.setup_data(parent, output)
    plan = registration["base_specification"]
    lock = digest(output / "registration.json")
    folder = output / "staged_tacs"
    folder.mkdir(exist_ok=True)
    runner.set_seed(SEED)
    with zipfile.ZipFile(ROOT / "outputs.zip") as archive:
        if hashlib.sha256(archive.read(parent["initialization"]["entry"])).hexdigest() != parent["initialization"]["sha256"]:
            raise ValueError("Initialization changed")
        model = runner.build_task(archive, "full_tacs", "cuda")
    optimizer = torch.optim.AdamW(runner.param_groups(model, plan), weight_decay=plan["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    last = folder / "checkpoint_last.pt"
    if last.exists() and stage != "smoke":
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["registration_sha256"] != lock:
            raise ValueError("Resume protocol changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        del saved
    save_json({"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0)}, output / "environment.json")
    if stage == "smoke":
        # Avoid fused attention backward nondeterminism in the equality audit.
        # These switches affect this diagnostic process only, never training.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        from tacsx.models.retrieval_fusion import retrieve
        from train_tacs_shortlist import learned_scores
        runner._set_training_mode(model)
        batch = next(iter(runner.loader(source, train_ids[:4], SEED, 0, None, 4, 0)))
        q, y, qids = batch["query"].cuda(), batch["label"].cuda(), batch["query_id"].cuda()
        # Compare gradient identities in FP32; AMP's extra zero-gradient branch
        # can change FP16 accumulation rounding without changing the objective.
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                features = model.selector.encoder.encode_global(q)
                _, positions = retrieve(features.float(), pool["features"], pool["ids"], 16,
                                        qids, identities[qids], identities[pool["ids"]])
            scores = learned_scores(model, q, pool["features"][positions], 1 / .07)
            out = model.run_mode("full_tacs", q, pool["pixels"][positions].float(), y, scores)
        parameters = [p for p in model.parameters() if p.requires_grad]
        task_grad = tuple(g.clone() if g is not None else None for g in
                          torch.autograd.grad(out.task_loss, parameters, retain_graph=True, allow_unused=True))
        warm_grad = tuple(g.clone() if g is not None else None for g in
                          torch.autograd.grad(out.task_loss + policy_weight(1) * out.policy_loss, parameters,
                                              retain_graph=True, allow_unused=True))
        policy_grad = torch.autograd.grad(out.policy_loss, model.selector.projector.weight)[0]
        max_error = max(float((a - b).abs().max()) for a, b in zip(task_grad, warm_grad) if a is not None and b is not None)
        gradients_match = all(torch.allclose(a, b, rtol=2e-4, atol=2e-6)
                              for a, b in zip(task_grad, warm_grad) if a is not None and b is not None)
        audit = {"optimizer_steps": 0, "warmup_gradient_max_error_vs_task_only": max_error,
                 "gradient_identity_audit_precision": "float32", "warmup_gradients_match_task_only": gradients_match,
                 "policy_gradient_norm_when_enabled": float(policy_grad.float().norm()),
                 "reward_detached": not out.reward.requires_grad,
                 "task_gradients_finite": all(torch.isfinite(g).all().item() for g in task_grad if g is not None)}
        save_json(audit, output / "gradient_audit.json")
        if not gradients_match or not audit["reward_detached"] or not audit["task_gradients_finite"] or not torch.isfinite(policy_grad).all() or not audit["policy_gradient_norm_when_enabled"] > 0:
            raise ValueError(f"Gradient audit failed: {audit}")
        set_status(output, state="smoke_passed", stage="gradient_audit", optimizer_steps=0)
        print(json.dumps(audit), flush=True)
        return
    for epoch in range(len(history), 4):
        weight = policy_weight(epoch + 1)
        model.lambda_policy = weight
        runner.set_seed(SEED + epoch * 1009)
        set_status(output, state="running", stage="training", seed=SEED, epoch=epoch + 1, lambda_policy=weight)
        print(f"START staged_tacs epoch={epoch+1}/4 lambda_policy={weight}", flush=True)
        metrics = runner.epoch_pass(model, runner.loader(source, train_ids, SEED, epoch, None,
                                                        plan["batch_size"], plan["workers"]),
                                    "full_tacs", pool, identities, optimizer, scaler)
        scheduler.step()
        metrics.update(lambda_policy=weight, weighted_policy_loss=weight * metrics["policy_loss"])
        history.append({"epoch": epoch + 1, "train": metrics})
        runner.atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "history": history, "seed": SEED,
                            "registration_sha256": lock, "plan": registration}, last)
        save_json(history, folder / "history.json")
        print(f"FINISH staged_tacs epoch={epoch+1}: {json.dumps(metrics)}", flush=True)
    set_status(output, state="running", stage="validation", seed=SEED, epoch=4)
    probabilities, results, pairs, labels, query_ids = [], [], [], None, None
    for si, sigma in enumerate(runner.SIGMAS):
        result, probs, ys, qs, selected = runner.epoch_pass(model, runner.loader(source, val_ids, SEED, 0, si,
                                                                              plan["batch_size"], plan["workers"]),
                                                          "full_tacs", pool, identities)
        if labels is not None and (not torch.equal(labels, ys) or not torch.equal(query_ids, qs)):
            raise ValueError("Severity order mismatch")
        probabilities.append(probs)
        results.append({"sigma": sigma, **result})
        pairs.extend(selected)
        labels, query_ids = ys, qs
    runner.write_csv(pairs, folder / "validation_selected_pairs.csv")
    save_json({"conditions": results, "primary_mean_accuracy": sum(r["accuracy"] for r in results) / 2,
               "seed": SEED, "epoch": 4, "registration_sha256": lock}, folder / "metrics.json")
    runner.atomic_save({"probabilities": torch.stack(probabilities, dim=1), "labels": labels, "query_ids": query_ids,
                        "seed": SEED, "arm": "staged_tacs", "epoch": 4, "registration_sha256": lock}, folder / "validation_final.pt")
    comparison(output, registration, runner)
    set_status(output, state="complete", stage="comparison", seed=SEED, test_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "smoke", "run", "report"), default="register")
    parser.add_argument("--output", default="outputs/query_noise_staged_warmup_s42_v1")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    try:
        registration = register(output) if args.stage == "register" else json.loads((output / "registration.json").read_text())
        if args.stage in ("run", "smoke"):
            run(output, registration, args.stage)
        elif args.stage == "report":
            comparison(output, registration, load_runner())
    except Exception as error:
        set_status(output, state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
