"""Step-by-step inference trace for the clean_v3 No-Context and Full-TACS models.

Shared by ``scripts/build_stepwise_demo.py`` (offline, baked validation examples)
and ``tacsx.serve`` (live upload). Both therefore run the exact same forward path.

Reconstruction follows ``outputs/tacs_shortlist_fullft_clean_v3/execution/
train_tacs_shortlist_cpu_state_snapshot.py``:

* task models: v3 No-Context architecture from ``outputs.zip`` (config only), weights
  from ``clean_v3/<arm>/checkpoint_best.pt`` loaded strictly;
* selector: frozen official DINOv3 encoder + identity-initialised linear projector,
  cosine logits scaled by 1/0.07, argmax at inference;
* shortlist: 16 nearest frozen-DINO neighbours from the fixed 9,000-image training
  pool (labels never used), pixel-identical images excluded.

Only training-split images are read (pool and validation). The CIFAR test file is not.
"""
from __future__ import annotations

import io
import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .cli import build_model
from .data.cifar100 import _transform
from .models.dinov3_adapter import DINOv3Adapter
from .models.retrieval_fusion import retrieve
from .models.selector import TACSSelector
from .models.tacs import TACSClassifier

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/tacs_shortlist_fullft_clean_v3"
WEIGHTS = ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
SCALE = 1 / .07
TOPN = 5


class StepwiseEngine:
    def __init__(self, device: str = "cpu", archive: Path | None = None, run: Path = RUN, data_root: Path | None = None):
        self.device = torch.device(device)
        self.archive = Path(archive or ROOT / "outputs.zip")
        self.run = Path(run)
        self.data_root = Path(data_root or ROOT / "data")
        self.transform = _transform(224, False)

    # ---- loading --------------------------------------------------------------------
    def load(self):
        with (self.data_root / "cifar-100-python/train").open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        with (self.data_root / "cifar-100-python/meta").open("rb") as handle:
            self.names = pickle.load(handle, encoding="latin1")["fine_label_names"]
        self.images = np.asarray(data["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = list(data["fine_labels"])
        shortlist = torch.load(self.run / "validation_shortlist.pt", map_location="cpu", weights_only=False)
        self.val_ids = list(shortlist["query_ids"])
        self.val_positions = shortlist["positions"]
        self.val_similarities = shortlist["similarities"]
        self.pool_ids = shortlist["pool_ids"]
        self.pool_features = shortlist["pool_features"].float().to(self.device)
        self.identity_ids = shortlist["identity_ids"]
        self.plan = shortlist["plan"]
        ids = list(range(50000))
        random.Random(42).shuffle(ids)
        if not torch.equal(self.pool_ids, torch.tensor(ids[5000:14000])):
            raise ValueError("Pool ids differ from the clean_v3 protocol")
        self.val_index = {qid: i for i, qid in enumerate(self.val_ids)}
        self.models = {arm: self._load_arm(arm) for arm in ("no_context", "full_tacs")}
        self.dino = self.models["full_tacs"].selector.encoder
        return self

    def _base_config(self):
        with zipfile.ZipFile(self.archive) as archive:
            entry = next(n for n in archive.namelist()
                         if "cifar100_no_context_c4_e5_unf4v3/" in n and n.endswith("checkpoint_best.pt"))
            cfg = torch.load(io.BytesIO(archive.read(entry)), map_location="cpu", weights_only=False)["config"]
        cfg["backbone"]["repo_dir"] = str(ROOT / "third_party/dinov3")
        cfg["backbone"]["weights"] = str(WEIGHTS)
        return cfg

    def _load_arm(self, arm):
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"Local official DINOv3 weights required: {WEIGHTS}")
        cfg = self._base_config()
        self.pair_rope_layout = cfg["backbone"].get("pair_rope_layout", "legacy")
        model = build_model(cfg, False, include_selector=False)
        if arm == "full_tacs":
            encoder = DINOv3Adapter(cfg["backbone"]["repo_dir"], weights=str(WEIGHTS))
            selector = TACSSelector(encoder, projection=False)
            selector.projector = nn.Linear(encoder.embedding_dim, encoder.embedding_dim, bias=False)
            model = TACSClassifier(selector, model.task_model, tau=.1, lambda_policy=1., eval_selection="argmax")
        saved = torch.load(self.run / arm / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        if saved["arm"] != arm:
            raise ValueError(f"checkpoint arm mismatch for {arm}")
        model.load_state_dict(saved["model"], strict=True)
        model.best_epoch = int(saved["best"]["epoch"])
        return model.to(self.device).eval().requires_grad_(False)

    # ---- helpers --------------------------------------------------------------------
    def tensor(self, arr: np.ndarray) -> torch.Tensor:
        return self.transform(Image.fromarray(arr))

    def _top(self, logits):
        p = logits.float().softmax(-1)[0]
        v, i = p.topk(TOPN)
        return [{"label": self.names[int(k)], "class": int(k), "p": round(float(x), 4)} for x, k in zip(v, i)]

    @staticmethod
    def _ce(logits, label):
        return float(F.cross_entropy(logits.float(), torch.tensor([label], device=logits.device)))

    def shortlist_for_upload(self, q: torch.Tensor):
        feature = self.dino.encode_global(q)
        values, positions = retrieve(feature, self.pool_features, self.pool_ids.to(self.device), 16)
        return positions[0].cpu(), values[0].cpu()

    # ---- the trace ------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, query: np.ndarray, label: int | None = None, query_id: int | None = None):
        """Run both models on one 32x32 RGB image and return every intermediate step."""
        q = self.tensor(query).unsqueeze(0).to(self.device)
        if query_id is not None and query_id in self.val_index:
            row = self.val_index[query_id]
            positions, dino_sims = self.val_positions[row], self.val_similarities[row]
            shortlist_source = "shortlist validation đã tính sẵn ở bước prepare của clean_v3"
        else:
            positions, dino_sims = self.shortlist_for_upload(q)
            shortlist_source = "16 láng giềng DINOv3 đóng băng, tìm trực tiếp trên pool train cố định"
        cand_ids = [int(self.pool_ids[p]) for p in positions]
        candidates = torch.stack([self.tensor(self.images[i]) for i in cand_ids]).unsqueeze(0).to(self.device)

        nc_model, tacs = self.models["no_context"], self.models["full_tacs"]
        backbone = tacs.task_model.backbone
        tokens = backbone.prepare_tokens(q)
        grid = tokens.grid_size
        prefix = tokens.tokens.shape[1] - grid[0] * grid[1]

        nc_logits = nc_model.task_model(q, None)

        # Selector: frozen DINO global feature -> learned projector -> scaled cosine.
        zq_raw = tacs.selector.encoder.encode_global(q)
        zq = F.normalize(tacs.selector.projector(zq_raw), dim=-1)
        zc = F.normalize(tacs.selector.projector(self.pool_features[positions.to(self.device)]), dim=-1)
        scores = (zq @ zc.T)[0] * SCALE
        probs = scores.softmax(-1)
        action = int(scores.argmax())
        context = candidates[:, action]
        ctx_logits = tacs.task_model(q, context)
        qonly_logits = tacs.task_model(q, None)

        dino_rank = list(range(16))
        learned_rank = scores.argsort(descending=True).tolist()
        cands = []
        for j, cid in enumerate(cand_ids):
            cands.append({"id": cid, "label": self.names[self.targets[cid]],
                          "dino": round(float(dino_sims[j]), 4), "dinoRank": dino_rank[j] + 1,
                          "score": round(float(scores[j]), 3), "p": round(float(probs[j]), 4),
                          "rank": learned_rank.index(j) + 1,
                          "same": label is not None and self.targets[cid] == label})
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        out = {
            "trueLabel": self.names[label] if label is not None else None,
            "preprocess": {"raw": [32, 32], "input": [224, 224], "patch": backbone.patch_size,
                           "grid": list(grid), "mean": [.485, .456, .406], "std": [.229, .224, .225]},
            "tokens": {"prefix": prefix, "storage": prefix - 1, "patchesPerImage": grid[0] * grid[1],
                       "noContext": prefix + grid[0] * grid[1], "tacs": prefix + 2 * grid[0] * grid[1],
                       "dim": backbone.embedding_dim, "blocks": len(backbone.model.blocks)},
            "noContext": {"top": self._top(nc_logits), "pred": self.names[int(nc_logits.argmax())]},
            "tacs": {
                "shortlistSource": shortlist_source,
                "candidates": cands, "pick": action, "dinoTop1": 0,
                "entropy": round(entropy, 3), "maxEntropy": round(float(np.log(16)), 3),
                "top": self._top(ctx_logits), "pred": self.names[int(ctx_logits.argmax())],
                "queryOnlyTop": self._top(qonly_logits),
            },
        }
        if label is not None:
            out["noContext"]["correct"] = int(nc_logits.argmax()) == label
            out["noContext"]["ce"] = round(self._ce(nc_logits, label), 4)
            ce_ctx, ce_q = self._ce(ctx_logits, label), self._ce(qonly_logits, label)
            out["tacs"].update(correct=int(ctx_logits.argmax()) == label, ce=round(ce_ctx, 4),
                               queryOnlyCe=round(ce_q, 4), reward=round(ce_q - ce_ctx, 4))
        return out, cand_ids
