"""Create a portable, one-command Kaggle zipapp for the utility experiment.

Contains train/holdout/validation caches produced by frozen TACS models, source,
and reference results. Does not claim to retrain the frozen ViT backbones.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]

LAUNCHER = r'''"""Frozen-task TACS-X utility-selector training and validation, no test access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/tacsx_utility_run'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    destination = (args.output / 'inputs').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sys.argv[0]) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        for name, expected in manifest['files_sha256'].items():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Invalid bundle path')
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Corrupt bundle file: '+name)
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('Existing input differs: '+str(target))
            target.write_bytes(content)
    print('TACS-X utility reranking: frozen pretrained task/classifier caches; this command trains the second-stage selector.',flush=True)
    command = [sys.executable, str(destination/'scripts/train_tacsx_utility.py'),
               '--cache',str(destination/'cache'),'--output',str(args.output/'fit'),
               '--device',args.device,'--epochs',str(manifest['epochs']),'--seed',str(manifest['seed']),
               '--models',*manifest['models']]
    subprocess.run(command,check=True)

if __name__ == '__main__':
    main()
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/tacsx_utility_s42_v1")
    p.add_argument("--fit", type=Path, default=ROOT / "outputs/tacsx_utility_s42_v1/fit")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/kaggle/tacsx_utility.pyz")
    args = p.parse_args()
    fit_protocol = json.loads((args.fit / "protocol.json").read_text())
    files = {f"cache/{name}": args.cache / name for name in
             ("train.pt", "holdout.pt", "validation.pt", "pool.pt", "cache_protocol.json", "selector_split.json", "extra_no_context.pt")
             if (args.cache / name).exists()}
    files.update({"scripts/train_tacsx_utility.py": ROOT / "scripts/train_tacsx_utility.py",
                  "src/tacsx/models/utility_selector.py": ROOT / "src/tacsx/models/utility_selector.py",
                  "reference/result.json": args.fit / "result.json", "reference/selection.json": args.fit / "selection.json"})
    manifest = {"method": "TACS-X second-stage outcome-aware utility selector, frozen task caches",
                "training_scope": "utility head only; pretrained ViT training is not repeated by this bundle",
                "test_access": False, "epochs": fit_protocol["epochs"], "seed": fit_protocol["seed"], "models": fit_protocol["variants"],
                "files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.writestr("__main__.py", LAUNCHER)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("src/tacsx/__init__.py", "")
        archive.writestr("src/tacsx/models/__init__.py", "")
        for name, path in files.items():
            archive.write(path, name)
    temp.replace(args.output)
    print(json.dumps({"bundle": str(args.output), "size_mb": args.output.stat().st_size / 1e6,
                      "command": "python /kaggle/input/<dataset>/tacsx_utility.pyz --output /kaggle/working/tacsx_utility_run",
                      "reference_result": json.loads((args.fit / "result.json").read_text())}, indent=2))


if __name__ == "__main__":
    main()
