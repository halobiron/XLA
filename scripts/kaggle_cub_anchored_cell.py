# Kaggle notebook cell: matched multi-seed CUB low-data run (2 training images per class),
# No-Context vs anchored TACS-X. Configuration selected on local validation (dev seed 42);
# this run is the confirmatory one: fresh seeds, official test evaluated once.
# Inputs (private Datasets): this code bundle (tacsx_cub_anchored) and CUB-200-2011
# (official CUB_200_2011.tgz, MD5 97eceeb196236b17998738112f37df78, or its extracted folder).
# Accelerator: "GPU T4 x2" (two seeds in parallel per GPU). Internet not needed.
import glob, hashlib, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

SEEDS = ["0", "1", "2", "3"]            # confirmatory seeds; 42 was the local development seed
TRAIN_PER_CLASS, WARMUP, EPOCHS, WORKERS = "2", "5", "75", "2"
OUT = "/kaggle/working/cub200_lowdata2_v1"

bundle = Path(glob.glob("/kaggle/input/**/scripts/run_cub_anchored.py", recursive=True)[0]).parents[1]
CODE = Path("/kaggle/working/tacsx")
if not CODE.exists():
    shutil.copytree(bundle, CODE)
manifest = json.loads((CODE / "MANIFEST.json").read_text())
for rel, digest in manifest.items():
    h = hashlib.sha256((CODE / rel).read_bytes()).hexdigest()
    assert h == digest, f"bundle file changed: {rel}"

found = glob.glob("/kaggle/input/**/CUB_200_2011/images.txt", recursive=True)
if found:
    DATA = Path(found[0]).parent
else:
    tgz = Path(glob.glob("/kaggle/input/**/CUB_200_2011.tgz", recursive=True)[0])
    md5 = hashlib.md5()
    with tgz.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            md5.update(chunk)
    assert md5.hexdigest() == "97eceeb196236b17998738112f37df78", "not the official CUB archive"
    DATA = Path("/tmp/cub/CUB_200_2011")
    if not DATA.exists():
        with tarfile.open(tgz) as archive:
            archive.extractall("/tmp/cub")
assert sum(1 for _ in open(DATA / "images.txt")) == 11788

common = [sys.executable, "scripts/run_cub_anchored.py", "--output", OUT, "--data-root", str(DATA),
          "--weights", str(CODE / "checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
          "--dinov3-repo", str(CODE / "third_party/dinov3"), "--seeds", *SEEDS,
          "--arms", "no_context", "anchored_tacs", "--train-per-class", TRAIN_PER_CLASS,
          "--warmup-epochs", WARMUP, "--epochs", EPOCHS,
          "--workers", WORKERS]
subprocess.run(common + ["--stage", "register"], cwd=CODE, check=True)

import torch
gpus = torch.cuda.device_count()
groups = [SEEDS[i::gpus] for i in range(gpus)] if gpus > 1 else [SEEDS]
print(f"{gpus} GPU(s): seed groups {groups}", flush=True)
started, procs = time.time(), []
for gpu, group in enumerate(groups):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"/kaggle/working/train_gpu{gpu}.log", "w")
    procs.append((subprocess.Popen(common + ["--stage", "run", "--only-seeds", *group], cwd=CODE, env=env,
                                   stdout=log, stderr=subprocess.STDOUT), log, gpu))
while any(p.poll() is None for p, _, _ in procs):
    time.sleep(300)
    for _, _, gpu in procs:
        lines = open(f"/kaggle/working/train_gpu{gpu}.log").read().splitlines()
        print(f"[{(time.time() - started) / 3600:.2f}h] gpu{gpu}: {lines[-1] if lines else ''}", flush=True)
for p, log, gpu in procs:
    log.close()
    assert p.returncode == 0, f"GPU {gpu} run failed; see /kaggle/working/train_gpu{gpu}.log"
print(f"Training finished in {(time.time() - started) / 3600:.2f}h", flush=True)

# Official test: evaluated ONCE on validation-selected checkpoints (script refuses a second run).
subprocess.run(common + ["--stage", "test"], cwd=CODE, check=True,
               env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
print(json.dumps(json.loads(Path(OUT, "test", "test_result.json").read_text()), indent=1))
