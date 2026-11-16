"""Frozen current warm-up checkpoint: validation-only oracle diagnostics.

Label-based selections are diagnostic bounds/controls, never deployable policies.
No optimizer, test reader, model selection, or updates to existing artifacts.
"""
from __future__ import annotations

import argparse
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
WARMUP = ROOT / "outputs/query_noise_staged_warmup_s42_v1"


def oracle_views(logits, labels, candidate_labels, scores):
    import torch
    import torch.nn.functional as F
    b, k, classes = logits.shape
    if candidate_labels.shape != (b, k) or scores.shape != (b, k) or labels.shape != (b,):
        raise ValueError("Unpaired candidate logits, labels or scores")
    ce = F.cross_entropy(logits.float().flatten(0, 1), labels.repeat_interleave(k), reduction="none").reshape(b, k)
    same = candidate_labels.eq(labels[:, None])
    first_same = same.long().argmax(1)  # deterministic DINO-nearest fallback0
    actions = {"staged_selected": scores.argmax(1), "dino_nearest1": torch.zeros(b, dtype=torch.long, device=logits.device),
               "oracle_same_class_shortlist": first_same, "oracle_min_ce16": ce.argmin(1)}
    row = torch.arange(b, device=logits.device)
    return {"logits": {name: logits[row, action] for name, action in actions.items()}, "actions": actions,
            "any_correct16": logits.argmax(-1).eq(labels[:, None]).any(1),
            "same_class_available16": same.any(1), "candidate_ce": ce}


def nearest_same_class(query_features, pool_features, query_ids, pool_ids, query_identity, pool_identity,
                       labels, pool_labels):
    import torch
    import torch.nn.functional as F
    similarity = F.normalize(query_features.float(), dim=-1) @ F.normalize(pool_features.float(), dim=-1).T
    valid = (pool_labels[None] == labels[:, None]) & (pool_ids[None] != query_ids[:, None])
    valid &= pool_identity[None] != query_identity[:, None]
    values, position = similarity.masked_fill(~valid, -torch.inf).max(1)
    if not torch.isfinite(values).all():
        raise ValueError("No identity-safe same-class candidate in fixed train pool")
    return position


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(value, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    tmp.replace(path)


def status(output, **fields):
    save_json({"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **fields}, output / "status.json")


def load_runner():
    sys.path.insert(0, str(PARENT / "execution/scripts"))
    import run_query_noise_pilot as runner
    if Path(runner.__file__).resolve() != (PARENT / "execution/scripts/run_query_noise_pilot.py").resolve():
        raise ValueError("Expected frozen original runner")
    return runner


def register(output):
    path = output / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    if (output / "chunks").exists():
        raise ValueError("Cannot register after enumeration")
    execution = output / "execution" / Path(__file__).name
    execution.parent.mkdir(exist_ok=True)
    shutil.copy2(__file__, execution)
    paths = [WARMUP / "registration.json", WARMUP / "staged_tacs/checkpoint_last.pt",
             WARMUP / "staged_tacs/validation_final.pt", PARENT / "registration.json",
             PARENT / "seed_42/no_context/validation_final.pt", PARENT / "seed_42/full_tacs/validation_final.pt"]
    protocol = {"registered_utc": datetime.now(timezone.utc).isoformat(), "seed": 42, "epoch": 4,
                "checkpoint": "current staged-policy warm-up, not historical clean_v3",
                "query": "all 4996 identity-safe validation queries, original Gaussian .08/.16 realizations",
                "candidate_pool": "same 9000 clean training images; original noisy-query frozen-DINO nearest16",
                "modes": {"staged_selected": "original learned argmax within16",
                          "dino_nearest1": "label-free first DINO candidate",
                          "oracle_same_class_shortlist": "first DINO-ranked same-class candidate within16; fallback nearest1 when absent",
                          "oracle_same_class_pool": "DINO-nearest same-class candidate in the entire fixed9000 train pool",
                          "oracle_min_ce16": "minimum true-label CE among16; a loss oracle, not exact accuracy bound",
                          "oracle_any_correct16": "exact accuracy upper bound for this fixed classifier on these16 candidates"},
                "classifier_and_selector": "eval, frozen, inference_mode; zero optimizer steps",
                "precision": "same fp16 autocast and batch32 as original validation, TF32 disabled",
                "labels": "validation query and training-candidate labels used only by explicit oracle diagnostics",
                "primary_readout": "equal-weight mean accuracy across sigma .08/.16 versus saved paired No-Context",
                "uncertainty": "descriptive paired-query bootstrap95%,10000 draws; one checkpoint, reused validation; no multiplicity-controlled claim",
                "interpretation": "existence of useful context does not prove deployable learnability or identify a gradient bug",
                "test_access": False, "automatic_training_or_test_followup": False,
                "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths}, "source_sha256": digest(execution)}
    save_json(protocol, path)
    print(f"Registered {path}", flush=True)
    return protocol


def summarize(output, protocol, runner):
    import torch
    chunks = [[torch.load(p, map_location="cpu", weights_only=False) for p in sorted((output / "chunks").glob(f"sigma_{si}_batch_*.pt"))]
              for si in range(2)]
    lock = digest(output / "protocol.json")
    saved_staged = torch.load(WARMUP / "staged_tacs/validation_final.pt", map_location="cpu", weights_only=False)
    saved_nc = torch.load(PARENT / "seed_42/no_context/validation_final.pt", map_location="cpu", weights_only=False)
    saved_full = torch.load(PARENT / "seed_42/full_tacs/validation_final.pt", map_location="cpu", weights_only=False)
    for reference in (saved_nc, saved_full):
        if not torch.equal(reference["query_ids"], saved_staged["query_ids"]) or not torch.equal(reference["labels"], saved_staged["labels"]):
            raise ValueError("Unpaired reference predictions")
    packed, any_correct, coverage, all_pairs = {}, [], [], []
    reconstruction = []
    for si, rows in enumerate(chunks):
        if not rows or any(row["lock"] != lock for row in rows):
            raise ValueError("Missing or mismatched enumeration")
        query_ids, labels = torch.cat([r["query_ids"] for r in rows]), torch.cat([r["labels"] for r in rows])
        if not torch.equal(query_ids, saved_staged["query_ids"]) or not torch.equal(labels, saved_staged["labels"]):
            raise ValueError("Incomplete or reordered validation enumeration")
        for mode in rows[0]["predictions"]:
            packed.setdefault(mode, []).append(torch.cat([r["predictions"][mode] for r in rows]))
        any_correct.append(torch.cat([r["any_correct16"] for r in rows]))
        coverage.append(torch.cat([r["same_class_available16"] for r in rows]))
        reconstructed = packed["staged_selected"][-1].softmax(-1)
        expected = saved_staged["probabilities"][:, si]
        reconstruction.append({"sigma": runner.SIGMAS[si],
                               "prediction_agreement": float(reconstructed.argmax(-1).eq(expected.argmax(-1)).float().mean()),
                               "max_probability_error": float((reconstructed - expected).abs().max())})
        if reconstruction[-1]["prediction_agreement"] < .999 or reconstruction[-1]["max_probability_error"] > .005:
            raise ValueError(f"Cannot reconstruct selected checkpoint: {reconstruction[-1]}")
        for row in rows:
            for i, qid in enumerate(row["query_ids"]):
                for mode, ids in row["selected_ids"].items():
                    all_pairs.append({"query_id": int(qid), "sigma": runner.SIGMAS[si], "mode": mode,
                                      "candidate_id": int(ids[i]), "label": int(row["labels"][i])})
    labels = saved_staged["labels"]
    packed = {mode: torch.stack(values, dim=1) for mode, values in packed.items()}
    correctness = {mode: logits.argmax(-1).eq(labels[:, None]) for mode, logits in packed.items()}
    correctness["oracle_any_correct16"] = torch.stack(any_correct, dim=1)
    correctness["no_context"] = saved_nc["probabilities"].argmax(-1).eq(labels[:, None])
    correctness["full_tacs_control"] = saved_full["probabilities"].argmax(-1).eq(labels[:, None])
    rows, differences = [], {}
    from probe_tacs_robustness import paired_ci
    for mode, correct in correctness.items():
        accuracy = correct.double().mean(0)
        delta = (correct.double() - correctness["no_context"].double()).mean(1).numpy() * 100
        interval = paired_ci(delta)
        rows.append({"mode": mode, "accuracy_sigma_008": float(accuracy[0]), "accuracy_sigma_016": float(accuracy[1]),
                     "mean_accuracy": float(accuracy.mean()), "difference_vs_no_context_pp": interval["difference"],
                     "ci95_lower_pp": interval["ci"][0], "ci95_upper_pp": interval["ci"][1]})
        differences[mode] = interval
    for relative, expected in protocol["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Original checkpoint/control changed: {relative}")
    coverage = torch.stack(coverage, dim=1)
    same = correctness["oracle_any_correct16"]
    nc = correctness["no_context"]
    result = {"scope": "diagnostic-only oracle, frozen current warm-up epoch4; one seed; validation only",
              "metrics": rows, "reconstruction": reconstruction, "validation_count": len(labels),
              "same_class_shortlist_coverage": coverage.double().mean(0).tolist(),
              "same_class_shortlist_fallback_count": (~coverage).sum(0).tolist(),
              "nc_errors_repairable_with_any_of16": (~nc & same).sum(0).tolist(),
              "nc_correct_but_all16_wrong": (nc & ~same).sum(0).tolist(),
              "oracle_cannot_identify_unique_failure_cause": True,
              "test_access": False, "optimizer_steps": 0, "original_artifacts_unchanged": True, "protocol_sha256": lock}
    save_json(result, output / "result.json")
    runner.write_csv(rows, output / "comparison.csv")
    runner.write_csv(all_pairs, output / "selected_pairs_DIAGNOSTIC_ONLY.csv")
    print(json.dumps(result, indent=2), flush=True)
    return result


def run(output, protocol):
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: .venv-gpu/Scripts/python.exe")
    if Path(__file__).resolve() != (output / "execution" / Path(__file__).name).resolve() or digest(__file__) != protocol["source_sha256"]:
        raise ValueError("Run registered execution copy with TACS_QUERY_NOISE_ROOT set")
    for relative, expected in protocol["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Registered input changed: {relative}")
    parent = json.loads((PARENT / "registration.json").read_text())
    for relative, expected in parent["source_sha256"].items():
        if digest(PARENT / "execution" / relative) != expected:
            raise ValueError("Parent execution changed")
    for relative, expected in parent["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Parent data/cache changed")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner = load_runner()
    runner.set_seed(42)
    status(output, state="running", stage="data_audit")
    source, _, val_ids, identities, pool = runner.setup_data(parent, output)
    with zipfile.ZipFile(ROOT / "outputs.zip") as archive:
        model = runner.build_task(archive, "full_tacs", "cuda")
    saved = torch.load(WARMUP / "staged_tacs/checkpoint_last.pt", map_location="cpu", weights_only=False)
    if len(saved["history"]) != 4 or saved["seed"] != 42 or saved["registration_sha256"] != digest(WARMUP / "registration.json"):
        raise ValueError("Wrong current checkpoint")
    model.load_state_dict(saved["model"])
    del saved
    model.eval().requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Entire model must be frozen")
    pool_labels = torch.tensor(source.targets, device="cuda")[pool["ids"]]
    (output / "chunks").mkdir(exist_ok=True)
    lock = digest(output / "protocol.json")
    import time
    started = time.perf_counter()
    with torch.inference_mode():
        for si, sigma in enumerate(runner.SIGMAS):
            batches = runner.loader(source, val_ids, 42, 0, si, 32, 4)
            for bi, batch in enumerate(batches):
                path = output / "chunks" / f"sigma_{si}_batch_{bi:05d}.pt"
                if path.exists():
                    cached = torch.load(path, map_location="cpu", weights_only=False)
                    if cached["lock"] != lock or not torch.equal(cached["query_ids"], batch["query_id"]):
                        raise ValueError("Resume chunk mismatch")
                    continue
                q, y, qids = batch["query"].cuda(), batch["label"].cuda(), batch["query_id"].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    features = model.selector.encoder.encode_global(q)
                    _, positions = runner.retrieve(features.float(), pool["features"], pool["ids"], 16,
                                                   qids, identities[qids], identities[pool["ids"]])
                    qproj = F.normalize(model.selector.projector(features), dim=-1)
                    cproj = F.normalize(model.selector.projector(pool["features"][positions]), dim=-1)
                    scores = torch.einsum("bd,bnd->bn", qproj, cproj) / .07
                    logits = torch.stack([model.task_model(q, pool["pixels"][positions[:, j]]).float() for j in range(16)], dim=1)
                    same_pos = nearest_same_class(features, pool["features"], qids, pool["ids"], identities[qids],
                                                  identities[pool["ids"]], y, pool_labels)
                    pool_logits = model.task_model(q, pool["pixels"][same_pos]).float()
                views = oracle_views(logits, y, pool_labels[positions], scores)
                row = torch.arange(len(q), device="cuda")
                selected_ids = {mode: pool["ids"][positions[row, action]].cpu() for mode, action in views["actions"].items()}
                selected_ids["oracle_same_class_pool"] = pool["ids"][same_pos].cpu()
                predictions = {mode: value.cpu() for mode, value in views["logits"].items()}
                predictions["oracle_same_class_pool"] = pool_logits.cpu()
                runner.atomic_save({"lock": lock, "query_ids": batch["query_id"], "labels": batch["label"],
                                    "candidate_ids": pool["ids"][positions].cpu(), "scores": scores.float().cpu(),
                                    "all16_logits": logits.cpu(), "predictions": predictions, "selected_ids": selected_ids,
                                    "any_correct16": views["any_correct16"].cpu(),
                                    "same_class_available16": views["same_class_available16"].cpu()}, path)
                if bi == 0 or (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                    elapsed = time.perf_counter() - started
                    status(output, state="running", stage="enumerating", sigma=sigma,
                           queries_done=min((bi + 1) * 32, len(val_ids)), validation_count=len(val_ids), elapsed_seconds=elapsed)
                    print(f"oracle sigma={sigma}: {min((bi+1)*32,len(val_ids))}/{len(val_ids)}, elapsed={elapsed:.1f}s", flush=True)
    summarize(output, protocol, runner)
    status(output, state="complete", stage="comparison", elapsed_seconds=time.perf_counter() - started, test_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "report"), default="register")
    parser.add_argument("--output", default="outputs/query_noise_staged_oracle_s42_v1")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    try:
        protocol = register(output) if args.stage == "register" else json.loads((output / "protocol.json").read_text())
        if args.stage == "run":
            run(output, protocol)
        elif args.stage == "report":
            summarize(output, protocol, load_runner())
    except Exception as error:
        status(output, state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
