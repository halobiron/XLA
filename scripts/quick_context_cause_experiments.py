"""Quick causal probes: why does clean_v3 Full TACS not beat No-Context?

Short fine-tunes (in memory) of the locked Full TACS task model, varying only the
context distribution seen in training. Selector projector is fixed. Never reads
CIFAR test; never writes checkpoints. Label-based contexts are DIAGNOSTIC ONLY.

Variants:
  gumbel   : context ~ softmax(learned selector scores)  (matches original training)
  nearest1 : deterministic DINO nearest-1 context          (no selection noise)
  oracle   : random same-class context from the shortlist  (label oracle, capacity test)
Each evaluated on a fixed validation subset under several context conditions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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


def scores_for(full, engine, q, positions):
    with torch.no_grad():
        zq = F.normalize(full.selector.projector(full.selector.encoder.encode_global(q)), dim=-1)
        zc = F.normalize(full.selector.projector(engine.pool_features[positions]), dim=-1)
        return torch.einsum("bd,bnd->bn", zq, zc) * engine.plan["selector_logit_scale"]


def pick(variant, scores, same, gen):
    b = scores.shape[0]
    if variant == "gumbel":
        return torch.multinomial(scores.float().softmax(-1), 1, generator=gen).squeeze(1)
    if variant == "nearest1":
        return torch.zeros(b, dtype=torch.long, device=scores.device)
    if variant == "oracle":
        w = same.float() + 1e-6 * (~same.any(1, keepdim=True)).float()  # fallback: uniform if none
        w[:, 0] += 1e-9
        return torch.multinomial(w, 1, generator=gen).squeeze(1)
    raise ValueError(variant)


def evaluate(task, full, engine, val_q, val_y, val_pos, val_same, nc_logits, bs=64):
    task.eval()
    conds = {k: [] for k in ("query_only", "learned_argmax", "nearest1", "farthest16", "oracle_same", "random_pool")}
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(val_y), bs):
            q = val_q[s:s + bs].cuda()
            pos = val_pos[s:s + bs].cuda()
            same = val_same[s:s + bs].cuda()
            ids = engine.pool_ids[pos.cpu()]
            sc = scores_for(full, engine, q, pos)
            choice = {"learned_argmax": sc.argmax(-1).cpu(),
                      "nearest1": torch.zeros(len(q), dtype=torch.long),
                      "farthest16": torch.full((len(q),), pos.shape[1] - 1, dtype=torch.long),
                      "oracle_same": pick("oracle", sc, same, g).cpu()}
            conds["query_only"].append(task(q, None).float().cpu())
            for k, a in choice.items():
                cid = ids[torch.arange(len(q)), a]
                c = torch.stack([engine.tensor(engine.images[int(i)]) for i in cid]).cuda()
                conds[k].append(task(q, c).float().cpu())
            rnd = engine.pool_ids[torch.randint(len(engine.pool_ids), (len(q),))]
            c = torch.stack([engine.tensor(engine.images[int(i)]) for i in rnd]).cuda()
            conds["random_pool"].append(task(q, c).float().cpu())
    out = {}
    ncw = nc_logits.argmax(1) != val_y
    for k, v in conds.items():
        v = torch.cat(v)
        pred = v.argmax(1)
        out[k] = {"acc": float((pred == val_y).float().mean()), "ce": float(F.cross_entropy(v, val_y)),
                  "acc_on_nc_wrong": float((pred == val_y)[ncw].float().mean()),
                  "broken_on_nc_correct": float((pred != val_y)[~ncw].float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--variants", nargs="+", default=["gumbel", "nearest1", "oracle"])
    ap.add_argument("--output", default="outputs/diagnostics/quick_context_cause")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {a: RUN / a / "checkpoint_best.pt" for a in ("no_context", "full_tacs")}
    before = {a: digest(p) for a, p in ckpts.items()}
    set_seed(0)
    engine = StepwiseEngine("cuda").load()
    full, nc = engine.models["full_tacs"], engine.models["no_context"]
    targets = torch.tensor(engine.targets)

    # fixed validation subset
    g = torch.Generator().manual_seed(123)
    rows = torch.randperm(len(engine.val_ids), generator=g)[:args.val]
    vq_ids = torch.tensor([engine.val_ids[i] for i in rows])
    val_y = targets[vq_ids]
    val_pos = engine.val_positions[rows]
    val_same = targets[engine.pool_ids[val_pos]] == val_y[:, None]
    val_q = torch.stack([engine.tensor(engine.images[int(i)]) for i in vq_ids])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        nc_logits = torch.cat([nc.task_model(val_q[s:s + 64].cuda(), None).float().cpu() for s in range(0, len(val_y), 64)])
    result = {"scope": "quick in-memory probes; validation subset only; labels used only for oracle variant/diagnostics",
              "steps": args.steps, "batch_size": args.batch_size, "val_count": args.val,
              "no_context_val": {"acc": float((nc_logits.argmax(1) == val_y).float().mean())}}
    print("NC val acc", result["no_context_val"], flush=True)
    result["start_full_tacs"] = evaluate(full.task_model, full, engine, val_q, val_y, val_pos, val_same, nc_logits)
    print("start", json.dumps(result["start_full_tacs"]), flush=True)

    train = torch.load(RUN / "train_shortlist.pt", map_location="cpu", weights_only=False)
    tq_ids = torch.tensor(train["query_ids"])
    aug = _transform(224, True)
    order = torch.randperm(len(tq_ids), generator=torch.Generator().manual_seed(99))[:args.steps * args.batch_size]
    result["variants"] = {}
    for variant in args.variants:
        task = copy.deepcopy(full.task_model).requires_grad_(True)
        groups = [{"params": [p for n, p in task.named_parameters() if n.startswith("backbone.")], "lr": 1e-5},
                  {"params": [p for n, p in task.named_parameters() if not n.startswith("backbone.")], "lr": 3e-4}]
        opt = torch.optim.AdamW(groups, weight_decay=.05)
        scaler = torch.amp.GradScaler("cuda")
        gen = torch.Generator(device="cuda").manual_seed(5)
        set_seed(1)
        t0, same_frac, loss_sum = time.perf_counter(), 0., 0.
        for step in range(args.steps):
            _set_training_mode(task)
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            qids = tq_ids[idx]
            pos = train["positions"][idx]
            y = targets[qids].cuda()
            q = torch.stack([aug(Image.fromarray(engine.images[int(i)])) for i in qids]).cuda()
            same = (targets[engine.pool_ids[pos]] == targets[qids][:, None]).cuda()
            sc = scores_for(full, engine, q, pos.cuda())
            a = pick(variant, sc, same, gen).cpu()
            cid = engine.pool_ids[pos][torch.arange(len(qids)), a]
            same_frac += float(same[torch.arange(len(qids)), a.cuda()].float().mean())
            c = torch.stack([engine.tensor(engine.images[int(i)]) for i in cid]).cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(task(q, c), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss)
            if (step + 1) % 200 == 0:
                print(f"{variant} {step + 1}/{args.steps} loss={loss_sum / (step + 1):.4f} "
                      f"ctx_same={same_frac / (step + 1):.3f} {time.perf_counter() - t0:.0f}s", flush=True)
        ev = evaluate(task, full, engine, val_q, val_y, val_pos, val_same, nc_logits)
        result["variants"][variant] = {"train_loss": loss_sum / args.steps,
                                       "train_context_same_class": same_frac / args.steps, "val": ev}
        print(variant, json.dumps(ev), flush=True)
        del task, opt
        torch.cuda.empty_cache()
        (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert {a: digest(p) for a, p in ckpts.items()} == before
    result["checkpoints_unchanged"] = True
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
