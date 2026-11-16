"""Offline analysis of quick_context_fix outputs: label-free context selection rules.

Rules never use validation labels. Oracle rows are DIAGNOSTIC ONLY. To limit
selection bias, rules are ranked on validation half A and reported on half B.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "outputs/diagnostics/quick_context_fix")


def rules(d):
    L = d["contexts"]
    P = L.softmax(-1)
    n = L.shape[0]
    b = torch.arange(n)
    conf = P.max(-1).values
    ent = -(P * P.clamp_min(1e-12).log()).sum(-1)
    r = {"query_only": d["query_only"].softmax(-1),
         "nearest1": P[:, 0],
         "learned_argmax": P[b, d["scores"].argmax(-1)],
         "max_confidence_of16": P[b, conf.argmax(-1)],
         "min_entropy_of16": P[b, ent.argmin(-1)],
         "mean_prob_top4": P[:, :4].mean(1),
         "mean_prob_of16": P.mean(1),
         "conf_weighted_of16": (P * conf[..., None]).sum(1) / conf.sum(1, keepdim=True)}
    return r


def boot(diff, rng, draws=5000):
    idx = rng.integers(0, len(diff), (draws, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    meta = torch.load(OUT / "val_meta.pt", weights_only=False)
    y = meta["labels"]
    same = meta["candidate_same_class_DIAGNOSTIC"]
    n = len(y)
    half = torch.zeros(n, dtype=torch.bool)
    half[torch.randperm(n, generator=torch.Generator().manual_seed(2026))[: n // 2]] = True  # A
    ref = {name: torch.load(OUT / f"{name}.pt", weights_only=False)["query_only"].softmax(-1)
           for name in ("start_nc", "nc_cont") if (OUT / f"{name}.pt").exists()}
    rng = np.random.default_rng(0)
    report = {"count": n, "half_A": int(half.sum()), "half_B": int((~half).sum()), "references": {}, "models": {}}
    for k, p in ref.items():
        report["references"][k] = {"acc": float((p.argmax(1) == y).float().mean())}
    base = ref.get("nc_cont", ref["start_nc"])
    base_ok = (base.argmax(1) == y).float().numpy()
    for name in ("start_full", "gumbel", "nearest1", "mix50", "mix100", "oracle"):
        path = OUT / f"{name}.pt"
        if not path.exists():
            continue
        d = torch.load(path, weights_only=False)
        rs = rules(d)
        L = d["contexts"].softmax(-1)
        first_same = torch.where(same.any(1), same.float().argmax(1), torch.zeros(n, dtype=torch.long))
        rs["ORACLE_same_class_DIAG"] = L[torch.arange(n), first_same]
        rows = {}
        for rule, p in rs.items():
            ok = (p.argmax(1) == y).float().numpy()
            dA = (ok - base_ok)[half.numpy()] * 100
            dB = (ok - base_ok)[~half.numpy()] * 100
            mB, ciB = boot(dB, rng)
            rows[rule] = {"acc_all": float(ok.mean()), "diff_vs_nc_halfA_pp": float(dA.mean()),
                          "diff_vs_nc_halfB_pp": mB, "halfB_ci95": ciB,
                          "diff_vs_nc_all_pp": float((ok - base_ok).mean() * 100)}
        report["models"][name] = rows
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("refs", report["references"])
    for name, rows in report["models"].items():
        print(f"== {name}")
        for rule, v in rows.items():
            print(f"  {rule:24s} acc {100*v['acc_all']:.2f}  vs NC: all {v['diff_vs_nc_all_pp']:+.2f}  "
                  f"A {v['diff_vs_nc_halfA_pp']:+.2f}  B {v['diff_vs_nc_halfB_pp']:+.2f} "
                  f"[{v['halfB_ci95'][0]:+.2f},{v['halfB_ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
