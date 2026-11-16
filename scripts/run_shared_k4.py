"""Single-command evaluate/retrain of the known shared-K4 TACS-X recipe."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare_frozen_proposals(reference, run):
    """Reuse label-free official-DINO proposals, never trained task weights."""
    run.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "train_shortlist.pt", "validation_shortlist.pt"):
        source, destination = reference / name, run / name
        if destination.exists():
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                raise ValueError(f"Existing prepared input differs: {destination}")
        else:
            shutil.copy2(source, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("evaluate", "train"), default="evaluate")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prepare-only", action="store_true", help="validate/copy frozen proposal caches without model training")
    args = p.parse_args()
    reference = ROOT / "outputs/reference_shared_k4"
    run = reference if args.mode == "evaluate" else ROOT / "outputs/retrained_shared_k4"
    common = [sys.executable, str(ROOT / "scripts/train_tacs_shared_k4_snapshot.py"), "--output", str(run),
              "--epochs", "10", "--seed", "42", "--batch-size", "32", "--workers", str(args.workers),
              "--arms", "no_context", "full_tacs", "--shared-selector-backbone", "--shortlist-k", "4", "--cached-pixels",
              "--data-root", str(ROOT / "data"), "--base-archive", str(ROOT / "outputs.zip")]
    if args.mode == "train":
        prepare_frozen_proposals(reference, run)
        if args.prepare_only:
            print(f"Prepared identity-safe frozen proposals in {run}; no trained models copied", flush=True)
            return
    elif args.prepare_only:
        p.error("--prepare-only requires --mode train")
    if not (run / "pool_pixels_fp16.pt").exists():
        subprocess.run([*common, "--stage", "cache_pixels"], cwd=ROOT, check=True)
    if args.mode == "train":
        subprocess.run([*common, "--stage", "train"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_shared_k4_noise.py"), "--run", str(run),
                    "--split", args.split], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
