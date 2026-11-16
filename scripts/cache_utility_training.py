"""Cache train-only oracle feedback for frozen TACS-X utility selection.

The final validation cache reuses label-free all-candidate classifier forwards;
oracle actions/labels from validation never enter the utility model's inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(os.environ.get("TACS_QUERY_NOISE_ROOT", Path(__file__).resolve().parents[1]))
PARENT = ROOT / "outputs/query_noise_clean_context_3seed_v1"
WARMUP = ROOT / "outputs/query_noise_staged_warmup_s42_v1"
ORACLE = ROOT / "outputs/query_noise_staged_oracle_s42_v1"
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_warmup_oracle import digest, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/tacsx_utility_s42_v1")
    args = p.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(PARENT / "execution/scripts"))
    import run_query_noise_pilot as runner
    if Path(runner.__file__).resolve() != (PARENT / "execution/scripts/run_query_noise_pilot.py").resolve():
        raise ValueError("Expected frozen original runner")
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runner.set_seed(42)
    paths = [WARMUP / "staged_tacs/checkpoint_last.pt", WARMUP / "registration.json",
             PARENT / "seed_42/no_context/checkpoint_last.pt", PARENT / "registration.json",
             ORACLE / "protocol.json"]
    protocol = {"scope": "fixed frozen-classifier TACS-X utility learning cache, validation only",
                "train_queries": 4096, "internal_holdout_queries": 1024, "query_sample_seed": 20261116,
                "query_sampling": "train images outside fixed candidate pool; paired noise realizations remain in same split",
                "candidate_count": 16, "pool_count": 9000, "sigmas": [.08, .16],
                "labels_used_for_training": "train split only, teacher utility from enumerated candidate losses",
                "inference_inputs": "DINO image features and classifier predictions, no ground-truth labels or oracle actions",
                "test_access": False, "source_sha256": digest(__file__),
                "input_sha256": {str(v.relative_to(ROOT)): digest(v) for v in paths}}
    path = output / "cache_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Cache protocol changed")
    save_json(protocol, path)
    lock = digest(path)
    parent = json.loads((PARENT / "registration.json").read_text())
    for relative, expected in {**parent["input_sha256"], **parent["source_sha256"]}.items():
        target = ROOT / relative if relative in parent["input_sha256"] else PARENT / "execution" / relative
        if digest(target) != expected:
            raise ValueError(f"Parent source/data changed: {relative}")
    source, train_ids, validation_ids, identity, pool = runner.setup_data(parent, output)
    pool_set = set(pool["ids"].cpu().tolist())
    available = [qid for qid in train_ids if qid not in pool_set]
    sampled = random.Random(20261116).sample(available, 5120)
    splits = {"train": sampled[:4096], "holdout": sampled[4096:], "validation": validation_ids}
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "holdout"), ("train", "validation"), ("holdout", "validation"))):
        raise ValueError("Query split overlap")
    # Exclude pixel-identical overlap as well as IDs for selector train/holdout.
    id_cpu = identity.cpu()
    if set(id_cpu[splits["train"]].tolist()) & set(id_cpu[splits["holdout"]].tolist()):
        raise ValueError("Pixel identity overlap between selector train and holdout")
    save_json({"ids": splits, "train_holdout_validation_disjoint": True, "test_read": False}, output / "selector_split.json")
    runner.atomic_save({"pool_features": pool["features"].cpu(), "pool_ids": pool["ids"].cpu(), "lock": lock}, output / "pool.pt")
    with zipfile.ZipFile(ROOT / "outputs.zip") as archive:
        model = runner.build_task(archive, "full_tacs", "cuda")
        baseline = runner.build_task(archive, "no_context", "cuda")
    saved = torch.load(WARMUP / "staged_tacs/checkpoint_last.pt", map_location="cpu", weights_only=False)
    if len(saved["history"]) != 4:
        raise ValueError("Unfinished task checkpoint")
    model.load_state_dict(saved["model"])
    del saved
    saved = torch.load(PARENT / "seed_42/no_context/checkpoint_last.pt", map_location="cpu", weights_only=False)
    baseline.load_state_dict(saved["model"])
    del saved
    model.eval().requires_grad_(False)
    baseline.eval().requires_grad_(False)
    started = time.perf_counter()
    with torch.inference_mode():
        for split, ids in splits.items():
            target = output / f"{split}.pt"
            if target.exists():
                prior = torch.load(target, map_location="cpu", weights_only=False)
                if prior["lock"] != lock:
                    raise ValueError("Completed cache belongs to another plan")
                print(f"Reusing completed {split} cache", flush=True)
                del prior
                continue
            folder = output / "cache_chunks" / split
            folder.mkdir(parents=True, exist_ok=True)
            for si, sigma in enumerate(runner.SIGMAS):
                batches = runner.loader(source, ids, 42, 0, si, 32, 4)
                for bi, batch in enumerate(batches):
                    chunk_path = folder / f"sigma_{si}_batch_{bi:05d}.pt"
                    if chunk_path.exists():
                        cached = torch.load(chunk_path, map_location="cpu", weights_only=False)
                        if cached["lock"] != lock or not torch.equal(cached["query_ids"], batch["query_id"]):
                            raise ValueError("Chunk resume mismatch")
                        continue
                    q, qids = batch["query"].cuda(), batch["query_id"].cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = model.selector.encoder.encode_global(q)
                        similarities, positions = runner.retrieve(features.float(), pool["features"], pool["ids"], 16,
                                                                 qids, identity[qids], identity[pool["ids"]])
                        nc_logits = baseline.task_model(q).float()
                        if split == "validation":
                            oracle_chunk = torch.load(ORACLE / "chunks" / f"sigma_{si}_batch_{bi:05d}.pt", map_location="cpu", weights_only=False)
                            if (not torch.equal(oracle_chunk["query_ids"], batch["query_id"])
                                    or not torch.equal(oracle_chunk["candidate_ids"], pool["ids"][positions].cpu())):
                                raise ValueError("Original validation candidate order changed")
                            pair_logits = oracle_chunk["all16_logits"]
                        else:
                            pair_logits = torch.stack([model.task_model(q, pool["pixels"][positions[:, j]]).float().cpu()
                                                       for j in range(16)], dim=1)
                    runner.atomic_save({"lock": lock, "query_ids": batch["query_id"], "labels": batch["label"],
                                        "sigma_index": torch.full_like(batch["label"], si),
                                        "query_features": features.half().cpu(), "positions": positions.cpu(),
                                        "similarities": similarities.float().cpu(), "pair_logits": pair_logits.half(),
                                        "query_logits": nc_logits.half().cpu()}, chunk_path)
                    if bi == 0 or (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                        state = {"pid": os.getpid(), "state": "running", "stage": "cache", "split": split, "sigma": sigma,
                                 "queries_done": min((bi + 1) * 32, len(ids)), "query_count": len(ids),
                                 "elapsed_seconds": time.perf_counter() - started}
                        save_json(state, output / "status.json")
                        print(json.dumps(state), flush=True)
            chunks = [torch.load(v, map_location="cpu", weights_only=False) for v in sorted(folder.glob("*.pt"))]
            keys = [k for k in chunks[0] if k != "lock"]
            packed = {k: torch.cat([v[k] for v in chunks]) for k in keys}
            expected_ids = torch.tensor(ids + ids)
            if not torch.equal(packed["query_ids"], expected_ids):
                raise ValueError("Cache query order mismatch")
            runner.atomic_save({**packed, "split": split, "lock": lock}, target)
            print(f"Cached {split}: {len(expected_ids)} query/severity cases", flush=True)
    for relative, expected in protocol["input_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Original checkpoint changed")
    save_json({"state": "cache_complete", "elapsed_seconds": time.perf_counter() - started, "test_access": False}, output / "status.json")


if __name__ == "__main__":
    main()
