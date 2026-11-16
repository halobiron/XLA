"""User-requested resource reduction: finish seeds42/43 and never run seed44.

Keep the original registration and execution sources unchanged. Resume the same
training implementation/checkpoints with an explicit amendment; the original
three-seed promotion gate cannot pass in this reduced exploratory pilot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(os.environ.get("TACS_QUERY_NOISE_ROOT", Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduced_summary(correct):
    import numpy as np
    values = np.asarray(correct, dtype=np.float64)
    if values.ndim != 4 or values.shape[:2] != (2, 2) or values.shape[3] != 2 or not values.shape[2]:
        raise ValueError("Expected two retained paired seeds, two arms and two severities")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Correctness must be binary")
    differences = (values[:, 1] - values[:, 0]).mean(axis=(1, 2)) * 100
    return {"effective_seeds": [42, 43], "omitted_seeds": [44],
            "seed_differences_pp": differences.tolist(), "mean_difference_pp": float(differences.mean()),
            "seed_t_ci95_pp": None, "crossed_bootstrap_ci95_pp": None,
            "all_retained_seeds_positive": bool((differences > 0).all()), "all_three_positive": False,
            "promotion_gate_passed": False,
            "gate_reason": "original three-seed requirement not evaluated; seed44 omitted at user request to save time",
            "inference_scope": "descriptive two-seed validation pilot; original three-seed confirmatory gate unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("amend", "run", "report"), default="amend")
    parser.add_argument("--output", default="outputs/query_noise_clean_context_3seed_v1")
    args = parser.parse_args()
    output = ROOT / args.output
    registration_path = output / "registration.json"
    registration = json.loads(registration_path.read_text())
    if registration["specification"]["seeds"] != [42, 43, 44]:
        raise ValueError("Unexpected original seeds")
    amendment_path = output / "resource_amendment.json"
    execution = output / "execution_amendment" / Path(__file__).name
    if args.stage == "amend":
        if amendment_path.exists():
            raise ValueError("Resource amendment already registered")
        execution.parent.mkdir(exist_ok=True)
        shutil.copy2(__file__, execution)
        amendment = {"amended_utc": datetime.now(timezone.utc).isoformat(),
                     "reason": "user requested skipping seed44 because of runtime",
                     "original_registration_sha256": digest(registration_path),
                     "effective_seeds": [42, 43], "omitted_seeds": [44],
                     "model_data_endpoint_unchanged": True, "original_promotion_gate_available": False,
                     "automatic_test_or_kaggle": False, "execution_sha256": digest(execution)}
        amendment_path.write_text(json.dumps(amendment, indent=2), encoding="utf-8")
        print(f"Registered resource amendment: {amendment_path}", flush=True)
        return
    amendment = json.loads(amendment_path.read_text())
    if (amendment["original_registration_sha256"] != digest(registration_path)
            or amendment["execution_sha256"] != digest(execution)
            or Path(__file__).resolve() != execution.resolve()):
        raise ValueError("Run the immutable amended execution with TACS_QUERY_NOISE_ROOT set")
    snapshot = output / "execution/scripts/run_query_noise_pilot.py"
    sys.path.insert(0, str(snapshot.parent))
    # Keep an importable module name for Windows-spawned DataLoader workers.
    runner = importlib.import_module("run_query_noise_pilot")
    if Path(runner.__file__).resolve() != snapshot.resolve():
        raise ValueError("Expected the original registered runner module")
    # Only orchestration and the incomplete-study summary change. The original
    # checkpoint/optimizer/epoch/validation implementation and hashes are reused.
    runner.SEEDS = (42, 43)
    runner.paired_seed_summary = reduced_summary
    plan = registration["specification"]
    run_args = argparse.Namespace(stage=args.stage, batch_size=plan["batch_size"], workers=plan["workers"])
    try:
        if args.stage == "run":
            runner.run(run_args, output, registration)
        else:
            runner.report(output, registration)
    except Exception as error:
        runner.status(output, state="failed", error=repr(error), omitted_seeds=[44])
        raise


if __name__ == "__main__":
    main()
