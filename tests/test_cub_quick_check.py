import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("cub_quick", scripts / "run_cub_quick_check.py")
quick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick)


def test_minimum_ce_is_not_an_accuracy_upper_bound():
    logits = torch.tensor([[[.34, .33, .33], [.40, .50, .10]]]).log()
    query = torch.tensor([[.1, .8, .1]]).log()
    result = quick.oracle_readouts(logits, query, torch.tensor([0]))
    assert result["oracle_any_correct16"].item()
    assert not result["oracle_min_ce16"].item()
    assert not result["internal_query_only"].item()


def test_reference_check_rejects_different_predictions_even_if_both_are_wrong():
    rows = [{"query_id": "5", "query_label": "0", "prediction": "1"}]
    quick.verify_reference(rows, torch.tensor([5]), torch.tensor([0]), torch.tensor([1]))
    with pytest.raises(AssertionError, match="predictions differ"):
        quick.verify_reference(rows, torch.tensor([5]), torch.tensor([0]), torch.tensor([2]))
    with pytest.raises(AssertionError, match="query order"):
        quick.verify_reference(rows, torch.tensor([6]), torch.tensor([0]), torch.tensor([1]))


def test_image_guard_rejects_official_test_even_through_direct_pil(tmp_path, monkeypatch):
    data = tmp_path / "data/CUB_200_2011"
    (data / "images").mkdir(parents=True)
    (data / "images.txt").write_text("1 train.png\n2 test.png\n")
    (data / "train_test_split.txt").write_text("1 1\n2 0\n")
    for name in ("train.png", "test.png"):
        Image.new("RGB", (2, 2)).save(data / "images" / name)
    monkeypatch.setattr(quick, "ROOT", tmp_path)
    original, audit = quick.protect_test_images()
    try:
        with Image.open(data / "images/train.png") as image:
            assert image.size == (2, 2)
        with pytest.raises(RuntimeError, match="test image"):
            Image.open(data / "images/test.png")
        assert audit == {"training_image_opens": 1, "forbidden_image_attempts": 1}
    finally:
        Image.open = original
