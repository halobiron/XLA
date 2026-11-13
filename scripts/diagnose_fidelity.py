"""Read-only validation diagnostics for saved TACS-X runs (never reads test data).

No training, downloads or checkpoint changes. Layout/head combinations are probes,
not new benchmark results. Run from the repository root with the project Python.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from tacsx.data.candidate_pool import FixedCandidatePool, PoolItem
from tacsx.data.cifar100 import _transform
from tacsx.models.dinov3_adapter import DINOv3Adapter
from tacsx.models.selector import TACSSelector


class TrainingImages:
    """Read only CIFAR's train file, including no integrity scan of test files."""

    def __init__(self, root):
        with (Path(root) / "cifar-100-python" / "train").open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        self.images = np.asarray(data["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = data["fine_labels"]
        self.transform = _transform(224, False)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), self.targets[index]


def checkpoint_audit(path, initial):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    changed = defaultdict(lambda: {"changed": 0, "total": 0})
    for key, ref in initial.items():
        value = state.get("task_model.backbone.model." + key)
        if value is None:
            continue
        group = ".".join(key.split(".")[:2]) if key.startswith("blocks.") else key.split(".")[0]
        changed[group]["total"] += 1
        changed[group]["changed"] += int(not torch.allclose(value, ref, rtol=0, atol=0, equal_nan=True))
    head = torch.nn.Linear(initial["cls_token"].shape[-1], 100)
    head.load_state_dict({k.removeprefix("task_model.head."): v for k, v in state.items()
                          if k.startswith("task_model.head.")})
    selector_state = {k.removeprefix("selector."): v for k, v in state.items() if k.startswith("selector.")}
    return dict(changed), head.eval(), selector_state


@torch.no_grad()
def features(adapter, query, context=None):
    q = adapter.prepare_tokens(query)
    if context is None:
        return adapter.forward_blocks(q.tokens, q.grid_size)[:, 0]
    c = adapter.prepare_tokens(context)
    prefix = q.tokens.shape[1] - q.grid_size[0] * q.grid_size[1]
    tokens = torch.cat((q.tokens, c.tokens[:, prefix:]), dim=1)
    return adapter.forward_blocks(tokens, q.grid_size, num_images=2)[:, 0]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="outputs/diagnostics/fidelity.json")
    args = p.parse_args()
    if args.samples < 1 or args.samples > 5000 or args.batch_size < 1:
        p.error("samples must be 1..5000 and batch size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    adapter = DINOv3Adapter(str(ROOT / "third_party/dinov3"),
                           str(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    initial = adapter.model.state_dict()
    runs = {
        "no_context": "cifar100_no_context_c4_e5",
        "random": "cifar100_random_c4_e5",
        "full": "cifar100_full_tacs_c4_e5_actionfix",
        "topk4": "cifar100_topk_k4_c4_e5",
        "old_unf4": "cifar100_no_context_c4_e5_unf4",
    }
    audit, heads, full_selector_state = {}, {}, None
    for name, run in runs.items():
        audit[name], head, state = checkpoint_audit(ROOT / "outputs" / run / "checkpoint_best.pt", initial)
        if name != "old_unf4":
            heads[name] = head.to(device)
        if name == "full":
            full_selector_state = state
    if any(group["changed"] for group in audit["full"].values()):
        raise ValueError("This shared-encoder probe requires a verified frozen Full TACS checkpoint")
    for key, value in full_selector_state.items():
        if key.startswith("encoder.model.") and not torch.allclose(
                value, initial[key.removeprefix("encoder.model.")], rtol=0, atol=0, equal_nan=True):
            raise ValueError("Full selector backbone differs from the official frozen reference")
    adapter.to(device)
    selector = TACSSelector(adapter).to(device).eval()
    selector.load_state_dict(full_selector_state)
    # The encoder tensors in the Full checkpoint are verified frozen below;
    # loading them therefore does not change the official reference weights.
    base = TrainingImages(args.data_root)
    ids = list(range(len(base)))
    random.Random(42).shuffle(ids)
    val_ids, train_ids = ids[:5000], ids[5000:]
    pool = FixedCandidatePool([PoolItem(i, base.targets[i]) for i in train_ids[:9000]], 42)
    result = {
        "split": "validation_only", "samples": args.samples, "seed": 42,
        "query_ids": val_ids[:args.samples], "checkpoint_changes_vs_official": audit,
        "warning": "Small validation probes, not trained layout comparisons or final benchmark scores.",
        "runs_without_trainer_version": [], "probes": {},
    }
    for metrics_path in sorted((ROOT / "outputs").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "trainer_version" not in metrics:
            result["runs_without_trainer_version"].append(metrics_path.parent.name)
    accum = defaultdict(lambda: defaultdict(float))
    official_max_error = 0.0

    def record(name, feature, reference, labels):
        values = accum[name]
        values["samples"] += len(labels)
        values["cosine_to_query_sum"] += F.cosine_similarity(feature, reference).sum().item()
        for head_name, head in heads.items():
            logits = head(feature)
            values[head_name + "_correct"] += (logits.argmax(-1) == labels).sum().item()
            values[head_name + "_ce_sum"] += F.cross_entropy(logits, labels, reduction="sum").item()

    for start in range(0, args.samples, args.batch_size):
        selected_ids = val_ids[start:min(start + args.batch_size, args.samples)]
        examples = [base[i] for i in selected_ids]
        q = torch.stack([x for x, _ in examples]).to(device)
        labels = torch.tensor([y for _, y in examples], device=device)
        candidate_ids = [pool.sample_indices(i, 4) for i in selected_ids]
        candidates = torch.stack([torch.stack([base[i][0] for i in row]) for row in candidate_ids]).to(device)
        ref = features(adapter, q)
        official_max_error = max(official_max_error, (adapter.encode_global(q) - ref).abs().max().item())
        record("query_only", ref, ref, labels)
        contexts = {"duplicate_query": q, "blank_mean_rgb": torch.zeros_like(q),
                    "random_one": candidates[:, 0], "uniform_mean4": candidates.mean(1)}
        for layout in ("legacy", "horizontal", "per_image"):
            adapter.pair_rope_layout = layout
            for context_name, context in contexts.items():
                record(layout + "/" + context_name, features(adapter, q, context), ref, labels)
        adapter.pair_rope_layout = "legacy"
        scores = selector(q, candidates)
        actions = {"argmax": scores.argmax(-1),
                   "sampled": F.gumbel_softmax(scores, tau=.1, hard=True).argmax(-1)}
        for selection, action in actions.items():
            context = candidates[torch.arange(len(q), device=device), action]
            record("legacy/full_selector_" + selection, features(adapter, q, context), ref, labels)
        print(f"validation probes {min(start + args.batch_size, args.samples)}/{args.samples}", flush=True)
    result["query_adapter_vs_official_max_abs_error"] = official_max_error
    for name, values in accum.items():
        n = values["samples"]
        row = {"mean_cosine_to_query": values["cosine_to_query_sum"] / n}
        for head_name in heads:
            row[head_name + "_head_accuracy"] = values[head_name + "_correct"] / n
            row[head_name + "_head_ce"] = values[head_name + "_ce_sum"] / n
        result["probes"][name] = row
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "official_error": official_max_error,
                      "probes": result["probes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
