from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_shared_k4 import prepare_frozen_proposals


def test_retraining_preparation_never_copies_trained_task_checkpoints(tmp_path):
    reference, run = tmp_path / "reference", tmp_path / "train"
    reference.mkdir()
    for name in ("plan.json", "train_shortlist.pt", "validation_shortlist.pt"):
        (reference / name).write_bytes(name.encode())
    (reference / "checkpoint_best.pt").write_bytes(b"trained task weights must not be imported")
    prepare_frozen_proposals(reference, run)
    assert sorted(p.name for p in run.iterdir()) == ["plan.json", "train_shortlist.pt", "validation_shortlist.pt"]
    prepare_frozen_proposals(reference, run)
    (run / "plan.json").write_bytes(b"different experiment")
    with pytest.raises(ValueError):
        prepare_frozen_proposals(reference, run)
