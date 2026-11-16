import json
from pathlib import Path
import subprocess
import sys

import torch


def test_portable_fit_selects_on_train_holdout_and_emits_paired_validation(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "fit"
    cache.mkdir()
    generator = torch.Generator().manual_seed(42)
    pool = torch.randn(24, 6, generator=generator)
    torch.save({"pool_features": pool, "pool_ids": torch.arange(10000, 10024), "lock": "dummy"}, cache / "pool.pt")
    (cache / "cache_protocol.json").write_text(json.dumps({"scope": "dummy encoders, no image files"}))
    for split, offset in (("train", 0), ("holdout", 100), ("validation", 200)):
        values = {"query_ids": torch.arange(offset, offset + 4).repeat(2), "labels": torch.tensor([0, 1, 2, 0]).repeat(2),
                  "sigma_index": torch.arange(2).repeat_interleave(4), "query_features": torch.randn(8, 6, generator=generator),
                  "positions": torch.randint(0, 24, (8, 4), generator=generator), "similarities": torch.randn(8, 4, generator=generator),
                  "pair_logits": torch.randn(8, 4, 3, generator=generator), "query_logits": torch.randn(8, 3, generator=generator),
                  "split": split, "lock": "dummy"}
        torch.save(values, cache / f"{split}.pt")
    script = Path(__file__).resolve().parents[1] / "scripts/train_tacsx_utility.py"
    completed = subprocess.run([sys.executable, str(script), "--cache", str(cache), "--output", str(output),
                                "--epochs", "2", "--models", "statistics17", "--device", "cpu"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    selected = json.loads((output / "selection.json").read_text())
    assert selected["selection_uses_validation"] is False
    result = json.loads((output / "result.json").read_text())
    assert result["test_access"] is False
    assert result["difference_vs_no_context"]["query_clusters"] == 4
    assert (output / "statistics17/checkpoint_best.pt").exists()
    predictions = torch.load(output / "validation_predictions.pt", weights_only=False)
    assert predictions["logits"].shape == (8, 3)
    assert predictions["candidate_ids"].shape == (8,)
