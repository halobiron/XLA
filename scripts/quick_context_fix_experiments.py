"""Quick fix probes: break the selector/task-model chicken-and-egg (validation only).

Short in-memory fine-tunes with an identical cosine schedule, then exhaustive
validation logits for all 16 shortlist contexts so label-free inference rules can
be compared offline. Training-split labels may guide which context is shown during
TRAINING (variants mix*/oracle); validation labels are used only for scoring.
Never reads CIFAR test, never writes checkpoints.

Variants:
  nc_cont : No-Context checkpoint, query-only continuation (control for extra steps)
  gumbel  : Full TACS checkpoint, context ~ learned selector softmax (original)
  mix50   : Full TACS checkpoint, same-class context w.p. .5 else uniform shortlist
  oracle  : Full TACS checkpoint, same-class context always
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selector_scores(full, engine, q, positions):
    with torch.no_grad():
        zq = F.normalize(full.selector.projector(full.selector.encoder.encode_global(q)), dim=-1)
        zc = F.normalize(full.selector.projector(engine.pool_features[positions]), dim=-1)
        return torch.einsum("bd,bnd->bn", zq, zc) * engine.plan["selector_logit_scale"]


def choose(variant, scores, same, gen):
    b, n = scores.shape
    uniform = torch.full((b, n), 1. / n, device=scores.device)
    has = same.any(1, keepdim=True)
    same_w = torch.where(has, same.float(), uniform)
    if variant == "gumbel":
        w = scores.float().softmax(-1)
    elif variant == "oracle":
        w = same_w
    elif variant.startswith("mix"):
        p = int(variant[3:]) / 100
        w = p * same_w / same_w.sum(1, keepdim=True) + (1 - p) * uniform
    else:
        raise ValueError(variant)
    return torch.multinomial(w, 1, generator=gen).squeeze(1)


@torch.inference_mode()
def exhaustive(task, full, engine, vq, vpos, with_context, bs=64):
    task.eval()
    qonly, ctx, scores = [], [], []
    with torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(vq), bs):
            q = vq[s:s + bs].cuda()
            pos = vpos[s:s + bs]
            qonly.append(task(q, None).float().cpu())
            if not with_context:
                continue
            scores.append(selector_scores(full, engine, q, pos.cuda()).float().cpu())
            ids = engine.pool_ids[pos]
            per = []
            for r in range(ids.shape[1]):
                c = torch.stack([engine.tensor(engine.images[int(i)]) for i in ids[:, r]]).cuda()
                per.append(task(q, c).float().cpu())
            ctx.append(torch.stack(per, 1))
            if (s // bs) % 20 == 0:
                print(f"  eval {s + len(q)}/{len(vq)}", flush=True)
    out = {"query_only": torch.cat(qonly)}
    if with_context:
        out["contexts"] = torch.cat(ctx)
        out["scores"] = torch.cat(scores)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val", type=int, default=0, help="0 = all validation queries")
    ap.add_argument("--variants", nargs="+", default=["nc_cont", "gumbel", "mix50", "oracle"])
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--output", default="outputs/diagnostics/quick_context_fix")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {a: RUN / a / "checkpoint_best.pt" for a in ("no_context", "full_tacs")}
    before = {a: digest(p) for a, p in ckpts.items()}
    engine = StepwiseEngine("cuda").load()
    full, nc = engine.models["full_tacs"], engine.models["no_context"]
    targets = torch.tensor(engine.targets)

    n_val = args.val or len(engine.val_ids)
    rows = torch.arange(len(engine.val_ids))[:n_val]
    vq_ids = torch.tensor([engine.val_ids[i] for i in rows])
    vpos = engine.val_positions[rows]
    vq = torch.stack([engine.tensor(engine.images[int(i)]) for i in vq_ids])
    meta = {"labels": targets[vq_ids], "query_ids": vq_ids,
            "candidate_same_class_DIAGNOSTIC": targets[engine.pool_ids[vpos]] == targets[vq_ids][:, None]}
    torch.save(meta, out_dir / "val_meta.pt")

    print("start models", flush=True)
    torch.save(exhaustive(nc.task_model, full, engine, vq, vpos, False), out_dir / "start_nc.pt")
    torch.save(exhaustive(full.task_model, full, engine, vq, vpos, True), out_dir / "start_full.pt")

    train = torch.load(RUN / "train_shortlist.pt", map_location="cpu", weights_only=False)
    tq_ids = torch.tensor(train["query_ids"])
    aug = _transform(224, True)
    order = torch.randperm(len(tq_ids), generator=torch.Generator().manual_seed(99))[:args.steps * args.batch_size]
    log = {"args": vars(args)}
    for variant in args.variants:
        src = nc.task_model if variant == "nc_cont" else full.task_model
        task = copy.deepcopy(src).requires_grad_(True)
        groups = [{"params": [p for n, p in task.named_parameters() if n.startswith("backbone.")], "lr": args.lr_backbone},
                  {"params": [p for n, p in task.named_parameters() if not n.startswith("backbone.")], "lr": args.lr_head}]
        opt = torch.optim.AdamW(groups, weight_decay=.05)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: .5 * (1 + math.cos(math.pi * t / args.steps)))
        scaler = torch.amp.GradScaler("cuda")
        gen = torch.Generator(device="cuda").manual_seed(5)
        set_seed(1)
        t0, loss_sum, same_sum = time.perf_counter(), 0., 0.
        for step in range(args.steps):
            _set_training_mode(task)
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            qids = tq_ids[idx]
            pos = train["positions"][idx]
            y = targets[qids].cuda()
            q = torch.stack([aug(Image.fromarray(engine.images[int(i)])) for i in qids]).cuda()
            c = None
            if variant != "nc_cont":
                same = (targets[engine.pool_ids[pos]] == targets[qids][:, None]).cuda()
                a = choose(variant, selector_scores(full, engine, q, pos.cuda()), same, gen)
                same_sum += float(same[torch.arange(len(qids), device="cuda"), a].float().mean())
                cid = engine.pool_ids[pos][torch.arange(len(qids)), a.cpu()]
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
                print(f"{variant} {step + 1}/{args.steps} loss={loss_sum / (step + 1):.4f} "
                      f"ctx_same={same_sum / (step + 1):.3f} {time.perf_counter() - t0:.0f}s", flush=True)
        log[variant] = {"train_loss": loss_sum / args.steps, "train_ctx_same": same_sum / args.steps}
        torch.save(exhaustive(task, full, engine, vq, vpos, variant != "nc_cont"), out_dir / f"{variant}.pt")
        del task, opt
        torch.cuda.empty_cache()
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    assert {a: digest(p) for a, p in ckpts.items()} == before
    log["checkpoints_unchanged"] = True
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
