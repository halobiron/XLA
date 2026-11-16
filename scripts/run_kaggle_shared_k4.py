"""Paste as one Kaggle cell after attaching tacsx_shared_k4.pyz.

MODE='train' retrains BOTH arms for ten clean epochs from identical warm-start
weights. Use 'evaluate' for the packaged checkpoints. Validation is the default;
the previously measured Gaussian-noise gain is narrow, exploratory and one-seed.
"""
from pathlib import Path
import subprocess
import sys

MODE = "train"
SPLIT = "validation"
bundle = next(Path("/kaggle/input").rglob("tacsx_shared_k4.pyz"))
subprocess.run([sys.executable, str(bundle), "--mode", MODE, "--split", SPLIT,
                "--output", "/kaggle/working/tacsx_shared_k4"], check=True)
