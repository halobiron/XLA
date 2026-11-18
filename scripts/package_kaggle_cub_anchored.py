"""Build a self-contained folder to upload as ONE private Kaggle Dataset.

Contents: run script, tacsx package, DINOv3 backbone code (+licence), official
DINOv3 ViT-S/16 weights, the notebook cell and a SHA-256 manifest. The CUB data
goes in a second private Dataset (the official CUB_200_2011.tgz or its
extracted folder). Nothing is uploaded by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/kaggle/tacsx_cub_anchored"
WEIGHTS = "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    (OUT / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_cub_anchored.py", OUT / "scripts/run_cub_anchored.py")
    shutil.copytree(ROOT / "src/tacsx", OUT / "src/tacsx", ignore=ignore)
    shutil.copytree(ROOT / "third_party/dinov3/dinov3", OUT / "third_party/dinov3/dinov3", ignore=ignore)
    shutil.copy2(ROOT / "third_party/dinov3/LICENSE.md", OUT / "third_party/dinov3/LICENSE.md")
    (OUT / "checkpoints").mkdir()
    shutil.copy2(ROOT / WEIGHTS, OUT / WEIGHTS)
    shutil.copy2(ROOT / "scripts/kaggle_cub_anchored_cell.py", OUT / "kaggle_cell.py")
    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in files}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    size = sum(p.stat().st_size for p in files) / 2**20
    print(f"Wrote {OUT} ({len(files)} files, {size:.1f} MiB)")


if __name__ == "__main__":
    main()
