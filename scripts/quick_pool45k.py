"""Quick probe: Full TACS with a 45k (all-train) frozen-DINO candidate pool.

Same hyperparameters/order as quick_context_fix_experiments.py so its nc_cont
control is reused. Shortlists are label-free (frozen official DINO cosine),
exclude self and pixel-identical images. Validation only; never reads CIFAR test;
never writes checkpoints. Training-split labels guide context only in mix variants.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from PIL import Image

from tacsx.stepwise import StepwiseEngine, RUN
from tacsx.data.cifar100 import _transform
from tacsx.engine import _set_training_mode, set_seed
from tacsx.models.retrieval_fusion import retrieve

FIX = ROOT / "outputs/diagnostics/quick_context_fix"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_shortlist(qfeat, qids, pool_feat, pool_ids, identity, k=16):
    out = []
    for s in range(0, len(qfeat), 512):
        ids = qids[s:s + 512].cuda()
        _, pos = retrieve(qfeat[s:s + 512].cuda(), pool_feat, pool_ids, k, ids,
                          identity[ids.cpu()].cuda(), identity[pool_ids.cpu()].cuda())
        out.append(pos.cpu())
    pos = torch.cat(out)
    assert not (identity[pool_ids.cpu()[pos]] == identity[qids][:, None]).any()
    return pos


def scores(full, q, cfeat, scale):
    with torch.no_grad():
        zq = F.normalize(full.selector.projector(full.selector.encoder.encode_global(q)), dim=-1)
        zc = F.normalize(full.selector.projector(cfeat), dim=-1)
        return torch.einsum("bd,bnd->bn", zq, zc) * scale


def choose(variant, sc, same, gen):
    b, n = sc.shape
    uniform = torch.full((b, n), 1. / n, device=sc.device)
    same_w = torch.where(same.any(1, keepdim=True), same.float(), uniform)
    if variant == "gumbel":
        w = sc.float().softmax(-1)
    elif variant == "nearest1":
        return torch.zeros(b, dtype=torch.long, device=sc.device)
    elif variant.startswith("mix"):
        p = int(variant[3:]) / 100
        w = p * same_w / same_w.sum(1, keepdim=True) + (1 - p) * uniform
    else:
        raise ValueError(variant)
    return torch.multinomial(w, 1, generator=gen).squeeze(1)


@torch.inference_mode()
def exhaustive(task, full, engine, vq, vpos, pool_ids, pool_feat, scale, bs=64):
    task.eval()
    qonly, ctx, sc = [], [], []
    with torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(vq), bs):
            q = vq[s:s + bs].cuda()
            pos = vpos[s:s + bs]
            qonly.append(task(q, None).float().cpu())
            sc.append(scores(full, q, pool_feat[pos.cuda()], scale).float().cpu())
            ids = pool_ids.cpu()[pos]
            per = []
            for r in range(ids.shape[1]):
                c = torch.stack([engine.tensor(engine.images[int(i)]) for i in ids[:, r]]).cuda()
                per.append(task(q, c).float().cpu())
            ctx.append(torch.stack(per, 1))
            if (s // bs) % 20 == 0:
                print(f"  eval {s + len(q)}/{len(vq)}", flush=True)
    return {"query_only": torch.cat(qonly), "contexts": torch.cat(ctx), "scores": torch.cat(sc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["gumbel", "mix50"])
    ap.add_argument("--output", default="outputs/diagnostics/quick_pool45k")
    args = ap.parse_args()
    fix_args = json.loads((FIX / "train_log.json").read_text())["args"]
    steps, bsz = fix_args["steps"], fix_args["batch_size"]
    torch.backends.cuda.matmul.allow_tf32 = False
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    ckpts = {a: RUN / a / "checkpoint_best.pt" for a in ("no_context", "full_tacs")}
    before = {a: digest(p) for a, p in ckpts.items()}
    engine = StepwiseEngine("cuda").load()
    full = engine.models["full_tacs"]
    scale = engine.plan["selector_logit_scale"]
    targets = torch.tensor(engine.targets)
    identity = engine.identity_ids

    tf = torch.load(ROOT / "outputs/tacs_shortlist_v1/train_features.pt", weights_only=False)
    vf = torch.load(ROOT / "outputs/tacs_shortlist_v1/validation_features.pt", weights_only=False)
    pool_ids = tf["ids"].long().cuda()
    pool_feat = F.normalize(tf["features"].float().cuda(), dim=-1)

    meta_fix = torch.load(FIX / "val_meta.pt", weights_only=False)
    vq_ids = meta_fix["query_ids"]
    assert set(vq_ids.tolist()).isdisjoint(set(pool_ids.tolist()))
    vidx = {int(i): k for k, i in enumerate(vf["ids"])}
    vqf = vf["features"][[vidx[int(i)] for i in vq_ids]].float()
    vpos = build_shortlist(vqf, vq_ids, pool_feat, pool_ids, identity)
    same = targets[pool_ids.cpu()[vpos]] == targets[vq_ids][:, None]
    torch.save({"labels": targets[vq_ids], "query_ids": vq_ids, "candidate_same_class_DIAGNOSTIC": same,
                "positions": vpos}, out / "val_meta.pt")
    for name in ("nc_cont.pt", "start_nc.pt"):
        shutil.copy(FIX / name, out / name)
    print(f"val shortlist: nearest1 same-class {same[:, 0].float().mean():.3f}, any {same.any(1).float().mean():.3f}", flush=True)
    vq = torch.stack([engine.tensor(engine.images[int(i)]) for i in vq_ids])
    torch.save(exhaustive(full.task_model, full, engine, vq, vpos, pool_ids, pool_feat, scale), out / "start_full.pt")
    if not args.variants:
        return

    train = torch.load(RUN / "train_shortlist.pt", map_location="cpu", weights_only=False)
    tq_ids = torch.tensor(train["query_ids"])
    tidx = {int(i): k for k, i in enumerate(tf["ids"])}
    tpos = build_shortlist(tf["features"][[tidx[int(i)] for i in tq_ids]].float(), tq_ids, pool_feat, pool_ids, identity)
    aug = _transform(224, True)
    order = torch.randperm(len(tq_ids), generator=torch.Generator().manual_seed(99))[:steps * bsz]
    log = {"steps": steps, "batch_size": bsz, "pool": "all 45000 training-split images"}
    for variant in args.variants:
        task = copy.deepcopy(full.task_model).requires_grad_(True)
        groups = [{"params": [p for n, p in task.named_parameters() if n.startswith("backbone.")], "lr": fix_args["lr_backbone"]},
                  {"params": [p for n, p in task.named_parameters() if not n.startswith("backbone.")], "lr": fix_args["lr_head"]}]
        opt = torch.optim.AdamW(groups, weight_decay=.05)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: .5 * (1 + math.cos(math.pi * t / steps)))
        scaler = torch.amp.GradScaler("cuda")
        gen = torch.Generator(device="cuda").manual_seed(5)
        set_seed(1)
        t0, loss_sum, same_sum = time.perf_counter(), 0., 0.
        for step in range(steps):
            _set_training_mode(task)
            idx = order[step * bsz:(step + 1) * bsz]
            qids = tq_ids[idx]
            pos = tpos[idx]
            y = targets[qids].cuda()
            q = torch.stack([aug(Image.fromarray(engine.images[int(i)])) for i in qids]).cuda()
            cand = pool_ids.cpu()[pos]
            s = (targets[cand] == targets[qids][:, None]).cuda()
            a = choose(variant, scores(full, q, pool_feat[pos.cuda()], scale), s, gen)
            same_sum += float(s[torch.arange(len(qids), device="cuda"), a].float().mean())
            cid = cand[torch.arange(len(qids)), a.cpu()]
            c = torch.stack([engine.tensor(engine.images[int(i)]) for i in cid]).cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(task(q, c), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss.detach())
            if (step + 1) % 300 == 0:
                print(f"{variant} {step + 1}/{steps} loss={loss_sum / (step + 1):.4f} ctx_same={same_sum / (step + 1):.3f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
        log[variant] = {"train_loss": loss_sum / steps, "train_ctx_same": same_sum / steps}
        torch.save(exhaustive(task, full, engine, vq, vpos, pool_ids, pool_feat, scale), out / f"{variant}.pt")
        del task, opt
        torch.cuda.empty_cache()
        (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    assert {a: digest(p) for a, p in ckpts.items()} == before
    log["checkpoints_unchanged"] = True
    (out / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
