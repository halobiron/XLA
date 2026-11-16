"""Sequential one-factor seed42 ablations; validation only, resumable children."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/selector_ablation_s42_v1"


def save(value, name):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf8")
    temp.replace(path)


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    variants = [("policy_01", ["--lambda-policy", "0.1"]),
                ("residual_mlp", ["--selector-projector", "residual_mlp"])]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(OUT / "execution/src"), str(ROOT / "scripts")])
    runner = OUT / "execution/train_tacs_shortlist.py"
    originals = {a: ROOT / "outputs/tacs_shortlist_fullft_clean_v3" / a / "checkpoint_best.pt"
                 for a in ("no_context", "full_tacs")}
    hashes = {a: hashlib.sha256(p.read_bytes()).hexdigest() for a,p in originals.items()}
    save({"seed":42,"epochs":10,"variants":variants,"original_checkpoint_sha256":hashes,
          "selection":"validation accuracy, CE tie-break; exploratory one seed",
          "test_access":False,"baseline":"existing matched clean_v3 checkpoints; historical controls, not newly retrained"}, "registration.json")
    for name, flags in variants:
        common = [sys.executable, str(runner), "--output", str(OUT / name), "--epochs", "10",
                  "--arms", "full_tacs", "--cached-pixels", "--pixel-cache",
                  "outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt", "--prepared-from",
                  "outputs/tacs_shortlist_fullft_clean_v3", *flags]
        for stage in ("prepare", "train"):
            save({"state":"running","variant":name,"stage":stage,"pid":os.getpid(),"updated_unix":time.time()}, "status.json")
            with (OUT / f"{name}_{stage}.log").open("a", encoding="utf8") as log:
                result = subprocess.run([*common,"--stage",stage], cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                save({"state":"failed","variant":name,"stage":stage,"exit_code":result.returncode}, "status.json")
                raise RuntimeError(f"{name} {stage} failed; inspect log")
    compare_results(originals, hashes, variants)


def compare_results(originals, hashes, variants):
    import torch
    env_path = str(ROOT / "scripts")
    sys.path.insert(0,env_path)
    from search_native_context import paired_difference, metrics
    controls = ROOT / "outputs/tacs_shortlist_fullft_clean_v3"
    paths = {a:controls / a / "validation_best.pt" for a in originals}
    paths.update({name:OUT / name / "full_tacs/validation_best.pt" for name,_ in variants})
    predictions = {a:torch.load(p,map_location="cpu",weights_only=False) for a,p in paths.items()}
    y = predictions["no_context"]["labels"]
    rows, comparisons = [], {}
    for name,d in predictions.items():
        assert torch.equal(d["labels"],y)
        rows.append({"variant":name,"epoch":d["epoch"],**metrics(d["probabilities"],y)})
        if name not in originals:
            comparisons[name] = {a:paired_difference(d["probabilities"],predictions[a]["probabilities"],y) for a in originals}
    assert {a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in originals.items()} == hashes
    save({"metrics":rows,"paired_comparisons_exploratory":comparisons,"test_access":False}, "comparison.json")
    with (OUT / "comparison.csv").open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save({"state":"complete","updated_unix":time.time()}, "status.json")


if __name__ == "__main__":
    main()
