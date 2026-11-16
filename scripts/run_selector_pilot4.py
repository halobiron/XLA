"""Adopt the running policy trial; stop at four saved epochs, then run MLP.

The original ten-epoch cosine horizon is retained for comparable early curves.
No test access. Partial work after epoch four in the adopted child is discarded.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/selector_ablation_s42_v1"


def save(value, filename):
    p = OUT / filename
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    tmp.replace(p)


def main():
    adopted = int(sys.argv[1])
    save({"effective_epochs":4,"cosine_horizon_epochs":10,"seed":42,
          "reason":"User requested 3-4 epoch feasibility screening before Kaggle",
          "adopted_policy_child_pid":adopted,"selection":"best validation top1 within first four epochs",
          "test_access":False,"ten_epoch_controls":"reference only; first-four-epoch curves also reported"}, "pilot_protocol.json")
    save({"state":"running","variant":"policy_01","stage":"train","effective_epochs":4,"pid":os.getpid()},"status.json")
    history_path = OUT / "policy_01/full_tacs/history.json"
    while True:
        try:
            history = json.loads(history_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            history=[]
        if len(history)>=4:
            if len(history)>4:
                raise RuntimeError("More than four epochs saved; inspect before reporting")
            subprocess.run(["taskkill","/PID",str(adopted),"/T","/F"],check=True,capture_output=True)
            break
        time.sleep(2)
    # Keep the old execution snapshot immutable; cap a separate pilot snapshot.
    source = OUT / "execution/train_tacs_shortlist.py"
    pilot = OUT / "execution/train_tacs_shortlist_pilot4.py"
    code=source.read_text()
    assert "for epoch in range(args.epochs):" in code
    pilot.write_text(code.replace("for epoch in range(args.epochs):","for epoch in range(min(args.epochs, 4)):"))
    save({"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
          "pilot_sha256":hashlib.sha256(pilot.read_bytes()).hexdigest(),"change":"loop cap 4; unchanged plan and cosine horizon 10"},"pilot_source.json")
    env=dict(os.environ)
    env["PYTHONPATH"]=os.pathsep.join([str(OUT/"execution/src"),str(ROOT/"scripts")])
    common=[sys.executable,str(pilot),"--output",str(OUT/"residual_mlp"),"--epochs","10",
            "--arms","full_tacs","--cached-pixels","--pixel-cache","outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt",
            "--prepared-from","outputs/tacs_shortlist_v1","--selector-projector","residual_mlp"]
    for stage in ("prepare","train"):
        save({"state":"running","variant":"residual_mlp","stage":stage,"effective_epochs":4,"pid":os.getpid()},"status.json")
        with (OUT/f"residual_mlp_{stage}.log").open("a",encoding="utf8") as log:
            subprocess.run([*common,"--stage",stage],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    finalize()


def finalize():
    sys.path.insert(0,str(ROOT/"scripts"))
    from run_selector_ablations import compare_results
    originals={a:ROOT/"outputs/tacs_shortlist_fullft_clean_v3"/a/"checkpoint_best.pt" for a in ("no_context","full_tacs")}
    hashes=json.loads((OUT/"registration.json").read_text())["original_checkpoint_sha256"]
    compare_results(originals,hashes,[("policy_01",[]),("residual_mlp",[])])
    curves={}
    for name,folder in [("no_context",ROOT/"outputs/tacs_shortlist_fullft_clean_v3/no_context"),
                        ("full_tacs",ROOT/"outputs/tacs_shortlist_fullft_clean_v3/full_tacs"),
                        ("policy_01",OUT/"policy_01/full_tacs"),("residual_mlp",OUT/"residual_mlp/full_tacs")]:
        rows=json.loads((folder/"history.json").read_text())[:4]
        curves[name]=[{"epoch":r["epoch"],"accuracy":r["validation"]["accuracy"],"ce":r["validation"]["ce"]} for r in rows]
    save({"scope":"four-epoch screening, same ten-epoch cosine prefix, one seed, validation only","curves":curves},"pilot_comparison.json")
    save({"state":"complete","effective_epochs":4,"updated_unix":time.time()},"status.json")


if __name__=="__main__":
    try:
        finalize() if sys.argv[1:] == ["--report-only"] else main()
    except Exception as e:
        save({"state":"failed","error":repr(e)},"status.json")
        raise
