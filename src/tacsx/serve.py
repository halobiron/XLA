"""Live-inference server for the TACS-X demo page.

    python -m tacsx.serve --port 8765

Serves demo/ (static) plus a small JSON API so the page can classify an uploaded
image with four already-trained runs. No new dependency: stdlib http.server + torch.

Fidelity notes (also shown in the UI):
  * Models are the saved `checkpoint_best.pt` of the main-matrix runs (C=4, frozen DINOv3, 5 epochs).
  * Candidates are drawn from the same fixed, train-derived pool (seed 42, 20% of the 45k train split).
  * Top-1 selectors use argmax of the selector score (deterministic), whereas test-set evaluation
    sampled via Gumbel/Categorical. Top-K and DINO-similarity match their evaluation behaviour.
  * An uploaded photo is first reduced to 32x32 (CIFAR-100 resolution) and then resized to 224.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pickle
import random
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .cli import build_model
from .data.candidate_pool import FixedCandidatePool, PoolItem
from .models.baselines import dino_similarity_scores

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
CIFAR = ROOT / "data" / "cifar-100-python"

RUNS = {  # key -> (display name, run directory, how it picks context)
    "no_context": ("No-Context", "cifar100_no_context_c4_e5", "none"),
    "dino_similarity": ("DINO-similarity", "cifar100_dino_sim_c4_e5", "dino"),
    "full_tacs": ("Full TACS (chọn 1)", "cifar100_full_tacs_c4_e5_actionfix", "argmax"),
    "topk_k4": ("TACS-X Top-K, K=4", "cifar100_topk_k4_c4_e5", "topk"),
}
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
TO_TENSOR = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Engine:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.status = "loading"
        self.error: str | None = None
        self.lock = threading.Lock()
        self.models: dict = {}

    def load(self):
        try:
            import os
            os.chdir(ROOT)  # configs reference ./third_party and ./checkpoints
            train = pickle.load(open(CIFAR / "train", "rb"), encoding="latin1")
            test = pickle.load(open(CIFAR / "test", "rb"), encoding="latin1")
            meta = pickle.load(open(CIFAR / "meta", "rb"), encoding="latin1")
            self.names = meta["fine_label_names"]
            self.train_x = np.array(train["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
            self.train_y = train["fine_labels"]
            self.test_x = np.array(test["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
            self.test_y = test["fine_labels"]
            # Rebuild the fixed pool exactly as make_cifar100_loaders does (seed 42, val 10%, pool 20%).
            ids = list(range(len(self.train_x)))
            random.Random(42).shuffle(ids)
            train_ids = ids[int(len(ids) * 0.1):]
            pool_ids = train_ids[:max(2, int(len(train_ids) * 0.2))]
            self.pool = FixedCandidatePool([PoolItem(i, int(self.train_y[i])) for i in pool_ids], 42)
            for key, (_, run, _) in RUNS.items():
                ckpt = torch.load(ROOT / "outputs" / run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
                cfg = ckpt["config"]
                include_selector = cfg["experiment"]["mode"] not in {"no_context", "random_context", "dino_similarity"}
                model = build_model(cfg, False, include_selector=include_selector)
                model.load_state_dict(ckpt["model"])
                model.topk_k = 4 if key == "topk_k4" else model.topk_k
                self.models[key] = model.to(self.device).eval()
                print(f"loaded {key} <- {run}", flush=True)
            self.status = "ready"
        except Exception as exc:  # surfaced to the page instead of dying silently
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
            print("LOAD FAILED:", self.error, flush=True)

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _thumb(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).resize((96, 96), Image.NEAREST).save(buf, "PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _to_cifar(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        w, h = img.size
        s = min(w, h)  # centre-crop to a square, then shrink to CIFAR resolution
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.array(img.resize((32, 32), Image.LANCZOS))

    def _tensor(self, arr: np.ndarray) -> torch.Tensor:
        return TO_TENSOR(Image.fromarray(arr))

    def _top(self, logits: torch.Tensor, n: int = 5):
        p = logits.softmax(-1)[0]
        v, i = p.topk(n)
        return [{"label": self.names[int(k)], "p": round(float(x), 4)} for x, k in zip(v, i)]

    # ---- inference -----------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, query_arr: np.ndarray, query_id: int, true_label: int | None, c: int, resample: int):
        if self.status != "ready":
            raise RuntimeError("model chưa sẵn sàng")
        ids = self.pool.sample_indices(query_id, c, resample)
        cand = torch.stack([self._tensor(self.train_x[i]) for i in ids]).unsqueeze(0).to(self.device)
        q = self._tensor(query_arr).unsqueeze(0).to(self.device)
        cand_info = [{"img": self._thumb(self.train_x[i]), "label": self.names[self.train_y[i]], "id": int(i)} for i in ids]
        dummy = torch.zeros(1, dtype=torch.long, device=self.device)
        results = []
        t0 = time.time()
        for key, (name, _, how) in RUNS.items():
            model = self.models[key]
            if how == "none":
                logits, picks = model.task_model(q, None), []
            elif how == "dino":
                s = dino_similarity_scores(model.task_model.backbone, q, cand)
                j = int(s.argmax(-1)); logits = model.task_model(q, cand[:, j]); picks = [(j, 1.0)]
            elif how == "argmax":
                s = model.selector(q, cand)
                j = int(s.argmax(-1)); logits = model.task_model(q, cand[:, j]); picks = [(j, 1.0)]
            else:  # topk
                s = model.selector(q, cand)
                out = model.run_mode("topk_tacs", q, cand, dummy, s)
                logits = out.logits
                picks = [(int(i), float(w)) for i, w in zip(out.topk_indices[0], out.topk_weights[0])]
            results.append({"key": key, "name": name, "top": self._top(logits), "picks": [{"i": i, "w": round(w, 4)} for i, w in picks]})
        return {"query": self._thumb(query_arr), "trueLabel": self.names[true_label] if true_label is not None else None,
                "candidates": cand_info, "results": results, "seconds": round(time.time() - t0, 2), "c": c}


ENGINE = Engine()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DEMO), **kw)

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/status":
            return self._json(200, {"status": ENGINE.status, "error": ENGINE.error})
        if u.path == "/api/predict-test":
            qs = parse_qs(u.query)
            try:
                idx = int(qs["idx"][0]); c = int(qs.get("c", ["4"])[0]); rs = int(qs.get("r", ["0"])[0])
                if not (0 <= idx < 10000 and 1 <= c <= 16):
                    raise ValueError
                with ENGINE.lock:
                    return self._json(200, ENGINE.predict(ENGINE.test_x[idx], 50000 + idx, ENGINE.test_y[idx], c, rs))
            except (KeyError, ValueError):
                return self._json(400, {"error": "tham số không hợp lệ"})
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/predict":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > 12 * 1024 * 1024:
            return self._json(413, {"error": "Ảnh trống hoặc lớn hơn 12 MB."})
        raw = self.rfile.read(n)
        qs = parse_qs(u.query)
        try:
            img = Image.open(io.BytesIO(raw)); img.load()
        except Exception:
            return self._json(400, {"error": "Không đọc được file ảnh. Hãy dùng JPG, PNG hoặc WebP."})
        try:
            c = int(qs.get("c", ["4"])[0]); rs = int(qs.get("r", ["0"])[0])
            if not 1 <= c <= 16:
                raise ValueError
            qid = -int(hashlib.sha1(raw).hexdigest()[:8], 16) - 1  # negative: can never collide with a pool id
            with ENGINE.lock:
                return self._json(200, ENGINE.predict(ENGINE._to_cifar(img), qid, None, c, rs))
        except ValueError:
            return self._json(400, {"error": "tham số không hợp lệ"})
        except Exception as exc:
            return self._json(500, {"error": str(exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ENGINE.device = torch.device(a.device)
    threading.Thread(target=ENGINE.load, daemon=True).start()
    print(f"TACS-X demo on http://{a.host}:{a.port}  (models loading in background)", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
