"""One independent, test-only follow-up for the validation noise signal.

The endpoint is fixed before opening the CIFAR-100 test labels. It was chosen
because Gaussian-noise mean accuracy was the sole positive validation point
estimate after the registered seven-comparison family; that CI crossed zero.
This is an exploratory, one-seed TACS-X robustness confirmation, not a paper
benchmark or a clean-accuracy claim.
"""
from __future__ import annotations

import argparse
from itertools import chain
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch

from evaluate_native_residual_test import TestImages
from evaluate_shortlist_benchmarks import fingerprints
from probe_tacs_robustness import corrupt, paired_ci
from probe_v3_context import ROOT, TrainingImages, load_model
from search_native_context import atomic_save, json_save
from train_tacs_shortlist import build_task, encode_candidate_globals
from tacsx.data.identity import pixel_identity_ids, heldout_without_identity_overlap
from tacsx.models.dinov3_adapter import DINOv3Adapter
from tacsx.models.retrieval_fusion import retrieve


NOISE = (("gaussian_noise", .08), ("gaussian_noise", .16))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="outputs/tacs_shared_backbone_k4_v1")
    p.add_argument("--data-root", default="data")
    p.add_argument("--base-archive", default="outputs.zip")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    run = ROOT / args.run
    output = run / "evaluation" / "gaussian_noise_followup"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "test_result.json").exists():
        raise ValueError("Follow-up test result already exists; refusing a second look")
    plan = json.loads((run / "plan.json").read_text())
    validation_path = run / "evaluation" / "validation_result.json"
    validation = json.loads(validation_path.read_text())
    if validation["positive_familywise_metrics"]:
        raise ValueError("Use the original registered test gate when it has a positive metric")
    noise = validation["completion_evidence_seven_comparisons"]["gaussian_noise_mean_accuracy_pp"]
    if noise["difference"] <= 0 or noise["ci"][0] > 0:
        raise ValueError("The predeclared follow-up condition is not satisfied")
    if not plan.get("shared_selector_backbone") or plan["shortlist_k"] != 4:
        raise ValueError("Unexpected experiment plan")

    protocol = {
        "primary_endpoint": "mean top-1 accuracy difference (Full TACS minus No-Context) averaged across Gaussian noise sigma .08 and .16",
        "uncertainty": "two-sided 95% paired query-cluster bootstrap, 10000 draws",
        "selection_basis": "sole positive point estimate among seven validation comparisons; validation CI crossed zero; CIFAR-100 test remains unopened",
        "test_conditions": [list(x) for x in NOISE],
        "test_tuning": False,
        "fixed_models": ["full_tacs", "no_context"],
        "fixed_candidate_pool": "first 9000 from seed42 shuffled CIFAR-100 training split; no labels used",
        "pair_self_retrieval": "pixel-identical candidates excluded; held-out test images deduplicated against train and each other",
        "scope": "exploratory one-seed TACS-X noise-robustness confirmation; not clean accuracy, not CIFAR-100-C, not a reported paper benchmark",
        "test_split_reuse_disclosure": "The official CIFAR-100 test split was previously evaluated for an unrelated native-residual extension in this workspace. This follow-up fixes the current checkpoints and chooses its single endpoint from current validation only; the split is not claimed to be historically pristine.",
        "validation_signal": noise,
    }
    protocol_path = output / "prospective_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Follow-up protocol changed")
    json_save(protocol, protocol_path)

    checkpoints = {arm: run / arm / "checkpoint_best.pt" for arm in ("no_context", "full_tacs")}
    checkpoint_bytes = {arm: path.read_bytes() for arm, path in checkpoints.items()}
    lock = {
        "checkpoint_sha256": {arm: hashlib.sha256(value).hexdigest() for arm, value in checkpoint_bytes.items()},
        "training_plan_sha256": fingerprints(run / "plan.json"),
        "validation_result_sha256": fingerprints(validation_path),
        "training_data_sha256": fingerprints(Path(args.data_root) / "cifar-100-python" / "train") if Path(args.data_root).is_absolute() else fingerprints(ROOT / args.data_root / "cifar-100-python" / "train"),
        "test_file_sha256": fingerprints((Path(args.data_root) if Path(args.data_root).is_absolute() else ROOT / args.data_root) / "cifar-100-python" / "test"),
        "protocol_sha256": fingerprints(protocol_path),
    }
    lock_path = output / "checkpoint_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise ValueError("Follow-up checkpoint/data lock changed")
    json_save(lock, lock_path)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    train_source = TrainingImages(data_root)
    order = list(range(50000))
    random.Random(42).shuffle(order)
    pool_ids = torch.tensor(order[5000:14000])
    cache = torch.load(run / "train_shortlist.pt", map_location="cpu", weights_only=False)
    if not torch.equal(cache["pool_ids"], pool_ids):
        raise ValueError("Candidate pool does not match the locked plan")
    train_identity = pixel_identity_ids(train_source.images)
    if not torch.equal(cache["identity_ids"], train_identity):
        raise ValueError("Training image identities changed")
    pool_identity = train_identity[pool_ids].to(device)
    pool_features = cache["pool_features"].to(device)
    pixel_path = run / "pool_pixels_fp16.pt"
    if not pixel_path.exists():
        pixel_path = ROOT / "outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt"
    pixel_cache = torch.load(pixel_path, map_location="cpu", weights_only=False)
    if not torch.equal(pixel_cache["pool_ids"], pool_ids):
        raise ValueError("Pixel candidate cache order mismatch")
    pool_pixels = pixel_cache["pixels"].to(device)

    archive_path = Path(args.base_archive)
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    archive = zipfile.ZipFile(archive_path)
    models = {}
    for arm in ("no_context", "full_tacs"):
        models[arm] = build_task(archive, arm, device, plan["task_unfreeze_last_blocks"],
                                 shared_selector_backbone=(arm == "full_tacs")).eval().requires_grad_(False)
        saved = torch.load(io.BytesIO(checkpoint_bytes[arm]), map_location="cpu", weights_only=False)
        if saved["plan"] != plan:
            raise ValueError("Checkpoint plan mismatch")
        models[arm].load_state_dict(saved["model"])
    full = models["full_tacs"]
    _, _, cfg = load_model(archive, "no_context", device)
    proposal = DINOv3Adapter(cfg["backbone"]["repo_dir"], weights=cfg["backbone"]["weights"]).to(device).eval().requires_grad_(False)
    live_pool_features = encode_candidate_globals(full.selector.encoder, pool_pixels.unsqueeze(0))[0]

    # Only now deserialize test images and labels; the model/endpoint lock exists.
    test_source = TestImages(data_root)
    all_identity = pixel_identity_ids(chain(train_source.images, test_source.images))
    global_test_ids = list(range(50000, 60000))
    query_ids, excluded = heldout_without_identity_overlap(list(range(50000)), global_test_ids, all_identity)
    local_ids = [qid - 50000 for qid in query_ids]

    progress_path = output / "test_progress.pt"
    done, packed, labels, selected = 0, {arm: [[], []] for arm in models}, [], []
    if progress_path.exists():
        state = torch.load(progress_path, map_location="cpu", weights_only=False)
        if state["lock"] != lock:
            raise ValueError("Progress belongs to different models or data")
        done, packed, labels, selected = (state[k] for k in ("done", "packed", "labels", "selected"))
    started = time.perf_counter()
    for start in range(done, len(query_ids), args.batch_size):
        ids = query_ids[start:start + args.batch_size]
        locals_ = local_ids[start:start + args.batch_size]
        qids = torch.tensor(ids, device=device)
        qidentity = all_identity[qids.cpu()].to(device)
        batch_logits, batch_selected = {arm: [] for arm in models}, []
        for ci, (name, severity) in enumerate(NOISE):
            q = torch.stack([test_source.transform(
                corrupt(test_source.images[local], name, severity, qid)) for local, qid in zip(locals_, ids)]).to(device)
            proposal_q = proposal.encode_global(q)
            _, positions = retrieve(proposal_q, pool_features, pool_ids.to(device), plan["shortlist_k"],
                                    qids, qidentity, pool_identity)
            selector_q = full.selector.encoder.encode_global(q)
            selector_c = live_pool_features[positions]
            qproj = torch.nn.functional.normalize(full.selector.projector(selector_q), dim=-1)
            cproj = torch.nn.functional.normalize(full.selector.projector(selector_c), dim=-1)
            scores = torch.einsum("bd,bnd->bn", qproj, cproj) * plan["selector_logit_scale"]
            action = scores.argmax(-1)
            rows = torch.arange(len(q), device=device)
            chosen_positions = positions[rows, action]
            context_ids = pool_ids.to(device)[chosen_positions]
            if (all_identity[context_ids.cpu()] == qidentity.cpu()).any():
                raise ValueError("Self retrieval in follow-up")
            context = pool_pixels[chosen_positions].to(dtype=q.dtype)
            batch_logits["full_tacs"].append(full.task_model(q, context).cpu())
            batch_logits["no_context"].append(models["no_context"].task_model(q, None).cpu())
            if ci == 0:
                batch_selected = context_ids.cpu()
        for arm in models:
            for ci in range(2):
                packed[arm][ci].append(batch_logits[arm][ci])
        labels.extend(test_source.targets[i] for i in locals_)
        selected.append(batch_selected)
        done = start + len(ids)
        atomic_save({"done": done, "packed": packed, "labels": labels,
                     "selected": selected, "lock": lock}, progress_path)
        print(f"test noise follow-up: {done}/{len(query_ids)} x2, {time.perf_counter()-started:.1f}s", flush=True)

    y = torch.tensor(labels)
    probs = {arm: torch.stack([torch.cat(values) for values in packed[arm]], dim=1) for arm in models}
    correct = {arm: probs[arm].argmax(-1).eq(y[:, None]).float().numpy() for arm in models}
    per_query_delta = (correct["full_tacs"] - correct["no_context"]).mean(1) * 100
    primary = paired_ci(per_query_delta, .95)
    result = {
        "split": "CIFAR-100 test, Gaussian-noise follow-up",
        "count_after_identity_filter": len(y),
        "excluded_test_ids": excluded,
        "conditions": [{"name": name, "severity": severity,
                        "no_context_accuracy": float(correct["no_context"][:, i].mean()),
                        "full_tacs_accuracy": float(correct["full_tacs"][:, i].mean()),
                        "difference_pp": float((correct["full_tacs"][:, i]-correct["no_context"][:, i]).mean()*100)}
                       for i, (name, severity) in enumerate(NOISE)],
        "primary_endpoint_mean_noise_accuracy_difference_pp": primary,
        "positive_confirmation": primary["ci"][0] > 0,
        "scope": protocol["scope"],
        "lock": lock,
    }
    json_save(result, output / "test_result.json")
    atomic_save({"probabilities": probs, "labels": y, "selected_context_ids": selected}, output / "test_predictions.pt")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
