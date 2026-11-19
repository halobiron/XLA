"""Serve the TACS-X demo locally and run both models on images the user uploads.

    .venv-gpu/Scripts/python.exe scripts/serve_demo.py            # then open the printed address

The static page (demo/index.html) still opens offline by double-click; when it is served from here it
also shows an upload button in the walkthrough. Uploaded images are traced through the same seven
steps with the self-trained seed-10 No-Context and TACS-X checkpoints (outputs/cub200_selftrain_v1)
against the 5,400-image training-split pool. Nothing is written to disk, the official test split is
never read, and the server listens on 127.0.0.1 only.

Uploaded images have no true label, so the walkthrough shows predictions without right/wrong marks and
skips the reward (it needs the label). Pool images that are near-duplicates of the upload
(cosine >= 0.995 on the selector embedding) are masked, mirroring the identity masking used in training.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

import build_cub_demo as B
import run_cub_anchored as R
from tacsx.data.cub200 import image_transform
from tacsx.models.tacs import gate_features

MAX_UPLOAD = 15 * 2**20
DUP_COS = 0.995
STATIC = ("/demo/", "/paper/")
STATIC_DATA = ("/outputs/",)  # only small result files are linked from the page


class Engine:
    def __init__(self, run, seed, data):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        self.names = B.class_names(data)
        original_open, _ = R.protect_test_images(data)
        try:
            _, _, _, pools, _, _ = R.prepare(data, 16, 0, train_per_class=2, unlabeled=True)
        finally:
            Image.open = original_open
        self.pixels, self.pool_ids = pools["unlabeled"]
        from tacsx.data import make_cub200_loaders  # same candidate base as the builder
        train, *_ = make_cub200_loaders(data, 224, .2, 1, 16, 0, R.DATA_SEED, include_context=True)
        self.clean = train.dataset.candidate_base
        self.pool_labels = {j: self.clean.targets[int(i)] for j, i in enumerate(self.pool_ids.tolist())}
        warm = R.build_warm_model(ROOT / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                                  ROOT / "third_party/dinov3", seed)
        self.models = {}
        for name, arm in B.STUDENTS.items():
            m = R.make_arm(arm, warm, seed)
            m.load_state_dict(torch.load(run / f"seed_{seed}" / name / "checkpoint_best.pt", map_location="cpu",
                                         weights_only=False)["model"])
            self.models[arm] = m.eval()
        self.features = R.candidate_features(self.models["anchored_tacs"], self.pixels)
        self.transform = image_transform(224, False)
        self.seed, self.lock = seed, threading.Lock()

    def thumb(self, image_id, size):
        with Image.open(self.clean.root / "images" / self.clean.paths[image_id]) as im:
            return B.jpeg(im, size)

    @torch.no_grad()
    def trace(self, image):
        nc, tacs = self.models["no_context"], self.models["anchored_tacs"]
        q = self.transform(image)[None].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            nc_logits = nc.task_model(q).float()
            base = tacs.task_model(q).float()
            qf = F.normalize(tacs.selector.encoder.encode_global(q).float(), dim=-1)
            cos = qf @ self.features.T
            scores = (cos * (1 / .07)).masked_fill(cos >= DUP_COS, -torch.inf).float()
            action = scores.argmax(-1)
            ctx = tacs.task_model(q, self.pixels[action]).float()
            gate_logit = tacs.gate(gate_features(base, ctx, scores)).float()
            tokens_q = tacs.task_model.backbone.prepare_tokens(q).tokens.shape[1]
        probs = scores.softmax(-1)[0]
        finite = torch.isfinite(scores[0])
        top = torch.topk(scores[0], B.TOPK_CANDIDATES)

        def topn(logits, n=5):
            p = logits.softmax(-1)[0]
            v, i = p.topk(n)
            return [{"label": self.names[int(c)], "p": float(x), "cls": int(c)} for x, c in zip(v, i)]

        images = {}

        def ref(image_id, size):
            key = f"up:{image_id}@{size}"
            images.setdefault(key, self.thumb(image_id, size))
            return key

        context_id = int(self.pool_ids[action].item())
        images["up:query@224"] = B.jpeg(image, 224)
        example = {
            "trueLabel": None, "original": B.aspect_thumb(image), "originalSize": list(image.size),
            "query": "up:query@224", "context": ref(context_id, 224),
            "contextLabel": self.names[self.clean.targets[context_id]], "contextSame": None,
            "context_id": context_id, "tokens_query": int(tokens_q),
            "candidates": [{"pos": int(j), "id": int(self.pool_ids[j]), "label": self.names[self.pool_labels[int(j)]],
                            "same": None, "score": float(scores[0, j]), "p": float(probs[j]),
                            "img": ref(int(self.pool_ids[j]), 112)} for j in top.indices.tolist()],
            "pick": 0, "entropy": float(-(probs[finite] * probs[finite].clamp_min(1e-12).log()).sum()),
            "maxEntropy": math.log(int(finite.sum())), "poolSize": int(finite.sum()),
            "masked_near_duplicates": int((~finite).sum()),
            "noContext": {"top": topn(nc_logits), "correct": None, "ce": None},
            "tacs": {"top": topn(ctx), "correct": None, "ce": None,
                     "queryOnly": {"top": topn(base, 3), "correct": None, "ce": None},
                     "gateLogit": float(gate_logit[0]), "gateUsesContext": bool(gate_logit[0] >= 0)},
        }
        return {"example": example, "images": images}


def make_handler(engine):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def log_message(self, fmt, *args):
            sys.stderr.write("  " + fmt % args + "\n")

        def send_json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def api(self, path):
            return path.endswith("/api/health") or path.endswith("/api/trace")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/demo", "/demo/"):
                self.send_response(302)
                self.send_header("Location", "/demo/index.html")
                self.end_headers()
                return
            if path.endswith("/api/health"):
                self.send_json(200, {"ok": True, "seed": engine.seed, "poolSize": int(len(engine.pool_ids))})
                return
            allowed = path.startswith(STATIC) or (path.startswith(STATIC_DATA) and path.endswith((".json", ".csv")))
            if not allowed or ".." in path:
                self.send_error(404)
                return
            super().do_GET()

        def do_POST(self):
            if not self.path.split("?", 1)[0].endswith("/api/trace"):
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_UPLOAD:
                self.send_json(413, {"error": "Ảnh phải nhỏ hơn 15 MB."})
                return
            try:
                with Image.open(io.BytesIO(self.rfile.read(size))) as im:
                    image = im.convert("RGB")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                self.send_json(400, {"error": "Không đọc được tệp này như một ảnh."})
                return
            with engine.lock:
                result = engine.trace(image)
            self.send_json(200, result)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--run", default=str(ROOT / "outputs/cub200_selftrain_v1"))
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--data-root", default=str(ROOT / "data/CUB_200_2011"))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (use .venv-gpu/Scripts/python.exe)")
    data = Path(a.data_root)
    data = data / "CUB_200_2011" if (data / "CUB_200_2011").is_dir() else data
    print("Loading the pool (5,400 training images) and both checkpoints; this takes about a minute...", flush=True)
    engine = Engine(Path(a.run), a.seed, data)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(engine))
    print(f"Ready: http://127.0.0.1:{a.port}/demo/index.html  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
