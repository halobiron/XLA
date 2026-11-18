"""Frozen oracle on the final CUB warm-start checkpoints; zero optimizer steps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(os.environ.get("TACS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

import run_cub_quick_check as quick

pilot = quick.pilot
PARENT = Path("outputs/cub200_quick_warmstart_s42_v1")


def register(output):
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory; do not replace a prior diagnostic")
    parent = ROOT / PARENT
    record = json.loads((parent / "registration.json").read_text())
    for relative, expected in record["source_sha256"].items():
        assert pilot.digest(parent / "execution" / relative) == expected
    execution = output / "execution"
    (execution / "scripts").mkdir(parents=True)
    # Use the exact model/data implementation used for the checkpoints.
    shutil.copytree(parent / "execution/src", execution / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(parent / "execution/scripts/run_cub_pilot.py", execution / "scripts/run_cub_pilot.py")
    for name in (Path(__file__).name, "run_cub_quick_check.py"):
        shutil.copy2(ROOT / "scripts" / name, execution / "scripts" / name)
    inputs = [parent / "registration.json", parent / "split_audit.json"]
    inputs += [parent / arm / name for arm in pilot.ARMS for name in ("checkpoint_last.pt", "validation_epoch_004.csv")]
    inputs += [ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
    inputs += [ROOT / "data/CUB_200_2011" / n for n in ("images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt")]
    plan = {"parent": str(PARENT), "checkpoint": "fixed final continuation epoch4, absolute epoch9",
            "seed": 42, "queries": "all594 identity-safe validation images", "candidates_per_query": 16,
            "proposals": "top16 by the frozen current TACS selector from the unchanged1080-image training pool",
            "modes": ["independent_no_context", "selector_argmax", "internal_query_only", "mean_probability16", "oracle_min_ce16", "oracle_any_correct16"],
            "primary_diagnostic": "any-correct accuracy bound within these16 candidates versus matched No-Context84.34%",
            "labels": "validation labels only for readouts/oracles, never candidate scoring",
            "uncertainty": "descriptive paired bootstrap95%; one seed, reused validation, diagnostic only",
            "limitations": "not a full1080-candidate oracle or evidence that a deployable selector can achieve the bound",
            "optimizer_steps": 0, "test_access": False, "automatic_training_followup": False}
    pilot.dump(output / "registration.json", {"plan": plan,
               "input_sha256": {str(p.relative_to(ROOT)): pilot.digest(p) for p in inputs},
               "source_sha256": {str(p.relative_to(execution)): pilot.digest(p) for p in execution.rglob("*.py")}})
    print(f"REGISTERED {output}", flush=True)


def run(output):
    record = json.loads((output / "registration.json").read_text())
    if (output / "oracle.json").exists():
        raise ValueError("Completed diagnostic already exists")
    for relative, expected in record["source_sha256"].items():
        assert pilot.digest(output / "execution" / relative) == expected
    for relative, expected in record["input_sha256"].items():
        assert pilot.digest(ROOT / relative) == expected
    if not torch.cuda.is_available():
        raise RuntimeError("Use .venv-gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    pilot.set_seed(42)
    pilot.dump(output / "environment.json", {"python": sys.executable, "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "precision": "fp16 AMP, TF32 disabled"})
    original_open, image_audit = quick.protect_test_images()
    started = time.perf_counter()
    try:
        pilot.status(output, state="preparing")
        _, val, pixels, pool_ids, identities = pilot.prepare(output)
        old = json.loads((ROOT / PARENT / "split_audit.json").read_text())
        new = json.loads((output / "split_audit.json").read_text())
        for key in ("train_ids", "validation_ids", "pool_ids", "training_image_sha256"):
            assert new[key] == old[key], f"Split differs: {key}"
        arms = pilot.models()
        for arm, model in arms.items():
            checkpoint = torch.load(ROOT / PARENT / arm / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            assert len(checkpoint["history"]) == 4 and checkpoint["history"][-1]["absolute_epoch"] == 9
            model.load_state_dict(checkpoint["model"])
            model.eval().requires_grad_(False)
            assert not any(p.requires_grad for p in model.parameters())
            del checkpoint
        pilot.status(output, state="running", stage="frozen_oracle")
        result = quick.diagnose(output, arms, val, pixels, pool_ids, identities,
                                reference_run=ROOT / PARENT, reference_epoch=4)
        for relative, expected in record["input_sha256"].items():
            assert pilot.digest(ROOT / relative) == expected, f"Input changed: {relative}"
        audit = {"original_inputs_unchanged": True, "all_model_parameters_frozen": True,
                 "all_parameter_gradients_none": all(p.grad is None for model in arms.values() for p in model.parameters()),
                 "optimizer_steps": 0, **image_audit, "test_pixels_read": False}
        assert audit["all_parameter_gradients_none"] and not audit["forbidden_image_attempts"]
        pilot.dump(output / "integrity_audit.json", audit)
        result.update(optimizer_steps=0, test_pixels_read=False, elapsed_seconds=time.perf_counter() - started)
        pilot.dump(output / "oracle.json", result)
        pilot.status(output, state="complete", elapsed_seconds=result["elapsed_seconds"], test_pixels_read=False)
    finally:
        Image.open = original_open
        pilot.dump(output / "image_access_audit.json", image_audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("register", "run"), default="register")
    p.add_argument("--output", default="outputs/cub200_final_oracle_s42_v1")
    args = p.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "register":
        register(output)
    else:
        if Path(__file__).resolve() != output / "execution/scripts/diagnose_cub_context_oracle.py":
            raise ValueError("Execute the registered source copy with TACS_PROJECT_ROOT set")
        try:
            run(output)
        except Exception as error:
            pilot.status(output, state="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
