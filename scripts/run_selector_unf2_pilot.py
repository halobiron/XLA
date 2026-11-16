"""Four-epoch separate-selector pilot; validation only, immutable execution code."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/selector_unf2_s42_pilot4"


def save(d,name):
    p=OUT/name;tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(d,indent=2),encoding="utf8");tmp.replace(p)


def runner():
    spec=importlib.util.spec_from_file_location("pilot_runner",OUT/"execution/train_tacs_shortlist.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def audit():
    import torch
    m=runner();torch.set_num_threads(4);m.set_seed(42)
    source=m.TrainingImages(ROOT/"data")
    cache=torch.load(OUT/"train_shortlist.pt",weights_only=False,map_location="cpu")
    qids=cache["query_ids"][:4];pos=cache["positions"][:4]
    q=torch.stack([source[i][0] for i in qids]).cuda()
    c=torch.stack([torch.stack([source[int(i)][0] for i in row]) for row in cache["pool_ids"][pos]]).cuda()
    model=m.build_task(zipfile.ZipFile(ROOT/"outputs.zip"),"full_tacs","cuda",selector_unfreeze_last_blocks=2)
    m._set_training_mode(model)
    with torch.autocast("cuda",dtype=torch.float16):
        scores=m.learned_scores(model,q,cache["pool_features"][pos].cuda(),1/.07,c)
        out=model.run_mode("full_tacs",q,c,torch.tensor([source.targets[i] for i in qids],device="cuda"),scores)
    weight=model.selector.encoder.model.blocks[-1].attn.qkv.weight
    gt=torch.autograd.grad(out.task_loss,weight,retain_graph=True)[0]
    gp=torch.autograd.grad(out.policy_loss,weight)[0]
    d={"task_gradient_norm":float(gt.float().norm()),"policy_gradient_norm":float(gp.float().norm()),
       "finite":bool(torch.isfinite(gt).all() and torch.isfinite(gp).all()),"reward_detached":not out.reward.requires_grad,
       "first_ten_blocks_frozen":not any(p.requires_grad for b in model.selector.encoder.model.blocks[:10] for p in b.parameters()),
       "last_two_blocks_trainable":all(p.requires_grad for b in model.selector.encoder.model.blocks[-2:] for p in b.parameters()),"optimizer_steps":0}
    assert d["finite"] and d["reward_detached"] and d["first_ten_blocks_frozen"] and d["last_two_blocks_trainable"]
    assert min(d["task_gradient_norm"],d["policy_gradient_norm"])>0
    save(d,"gradient_audit.json");print(d,flush=True)


def rank_probe():
    import torch
    import torch.nn.functional as F
    m=runner();torch.set_num_threads(4);torch.manual_seed(20261114)
    cache=torch.load(OUT/"validation_shortlist.pt",weights_only=False,map_location="cpu")
    saved=torch.load(OUT/"full_tacs/checkpoint_best.pt",weights_only=False,map_location="cpu")
    model=m.build_task(zipfile.ZipFile(ROOT/"outputs.zip"),"full_tacs","cuda",selector_unfreeze_last_blocks=2)
    model.load_state_dict(saved["model"]);model.eval().requires_grad_(False)
    source=m.TrainingImages(ROOT/"data")
    indices=torch.randperm(len(cache["query_ids"]))[:1024]
    ranks=[];nc=0;total=0;entropies=[]
    records=[]
    with torch.inference_mode():
        for start in range(0,len(indices),8):
            ix=indices[start:start+8];qids=[cache["query_ids"][i] for i in ix]
            cids=cache["pool_ids"][cache["positions"][ix]]
            q=torch.stack([source[i][0] for i in qids]).cuda()
            c=torch.stack([torch.stack([source[int(i)][0] for i in row]) for row in cids]).cuda()
            y=torch.tensor([source.targets[i] for i in qids],device="cuda")
            scores=m.learned_scores(model,q,None,1/.07,c)
            logits=torch.stack([model.task_model(q,c[:,j]) for j in range(16)],1)
            ce=F.cross_entropy(logits.flatten(0,1),y.repeat_interleave(16),reduction="none").reshape(-1,16)
            a,b=torch.triu_indices(16,16,1,device="cuda")
            ds=scores[:,a]-scores[:,b];du=ce[:,b]-ce[:,a]
            valid=(ds.abs()>1e-8)&(du.abs()>1e-8)
            ranks.extend((((ds*du>0)&valid).sum(-1)/valid.sum(-1).clamp_min(1)).cpu().tolist())
            act=scores.argmax(-1);nc+=int(logits[torch.arange(len(q),device="cuda"),act].argmax(-1).eq(y).sum());total+=len(q)
            records.append({"query_ids":qids,"scores":scores.cpu(),"candidate_ce":ce.cpu()})
            print(f"rank probe {total}/1024",flush=True)
    torch.save(records,OUT/"rank_probe.pt")
    save({"count":total,"seed":20261114,"validation_only":True,"epoch":saved["best"]["epoch"],
          "accuracy":nc/total,"pairwise_score_utility_agreement":sum(ranks)/len(ranks),
          "warning":"1024-query diagnostic; original 50.14% reference used a different checkpoint and full validation, not a matched causal comparison"},"ranking.json")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ);env["PYTHONPATH"]=os.pathsep.join([str(OUT/"execution/src"),str(ROOT/"scripts")])
    train=OUT/"execution/train_tacs_shortlist.py"
    args=[sys.executable,str(train),"--output",str(OUT),"--epochs","10","--stop-after-epochs","4",
          "--arms","full_tacs","--cached-pixels","--pixel-cache","outputs/tacs_shortlist_fullft_v2/pool_pixels_fp16.pt",
          "--prepared-from","outputs/tacs_shortlist_fullft_clean_v3","--selector-unfreeze-last-blocks","2"]
    for stage in ["prepare","audit","train","ranking"]:
        save({"state":"running","stage":stage,"pid":os.getpid(),"effective_epochs":4,"updated_unix":time.time()},"status.json")
        cmd=[*args,"--stage",stage] if stage in ["prepare","train"] else [sys.executable,__file__,"--"+stage]
        with (OUT/(stage+".log")).open("a",encoding="utf8") as log:
            subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    import torch
    history=json.loads((OUT/"full_tacs/history.json").read_text());assert len(history)==4
    curves={"selector_unf2":history}
    for name in ["no_context","full_tacs"]:
        curves[name]=json.loads((ROOT/"outputs/tacs_shortlist_fullft_clean_v3"/name/"history.json").read_text())[:4]
    rows=[]
    for name,h in curves.items():
        best=max(h,key=lambda r:(r["validation"]["accuracy"],-r["validation"]["ce"]))
        rows.append({"variant":name,"best_epoch":best["epoch"],"accuracy":best["validation"]["accuracy"],"ce":best["validation"]["ce"]})
    save({"scope":"one-seed exploratory validation pilot, first four epochs, ten-epoch cosine prefix","metrics":rows},"comparison.json")
    with (OUT/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save({"state":"complete","effective_epochs":4},"status.json")


if __name__=="__main__":
    try:
        if sys.argv[1:]==["--audit"]:audit()
        elif sys.argv[1:]==["--ranking"]:rank_probe()
        else:main()
    except Exception as e:
        save({"state":"failed","error":repr(e)},"status.json");raise
