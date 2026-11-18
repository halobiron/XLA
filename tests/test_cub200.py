from pathlib import Path

from PIL import Image
import pytest
import torch

from tacsx.data import CUB200Images, make_cub200_loaders


@pytest.fixture
def cub_root(tmp_path):
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    paths, labels, split = [], [], []
    for i in range(24):
        name = f"bird_{i}.png"
        Image.new("RGB", (10, 12), (i * 10, 0, 0)).save(root / "images" / name)
        paths.append(f"{i+1} {name}")
        labels.append(f"{i+1} {i//8+1}")
        split.append(f"{i+1} {int(i % 8 < 6)}")
    for name, lines in (("images.txt", paths), ("image_class_labels.txt", labels),
                        ("train_test_split.txt", split), ("classes.txt", ["1 bird_a", "2 bird_b", "3 bird_c"])):
        (root / name).write_text("\n".join(lines), encoding="utf8")
    return root


def test_official_metadata_global_ids_and_zero_based_labels(cub_root):
    data = CUB200Images(cub_root.parent)
    assert len(data) == 24
    assert data.targets == [0] * 8 + [1] * 8 + [2] * 8
    assert sum(data.is_train) == 18
    assert data[8][0].mode == "RGB" and data[8][1] == 1


def test_pool_uses_only_fit_split_and_never_self_retrieves(cub_root, monkeypatch):
    original_open = Image.open
    test_names = {f"bird_{i}.png" for i in range(24) if i % 8 >= 6}
    def training_only_open(path, *args, **kwargs):
        assert Path(path).name not in test_names, "Validation must not open official test pixels"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", training_only_open)
    train, val, test, pool = make_cub200_loaders(cub_root, 16, .4, 3, 4, 0, 42)
    train_ids, val_ids, test_ids = [set(loader.dataset.query_ids) for loader in (train, val, test)]
    assert len(train_ids) == 15 and len(val_ids) == 3 and len(test_ids) == 6
    assert not train_ids & val_ids and not (train_ids | val_ids) & test_ids
    assert {item.index for item in pool.items} <= train_ids
    for loader in (train, val):
        for batch in loader:
            assert batch["query"].shape[1:] == (3, 16, 16)
            assert not (batch["candidate_ids"] == batch["query_id"][:, None]).any()


def test_matched_split_order_and_augmentation_across_modes(cub_root):
    baseline = make_cub200_loaders(cub_root, 16, .4, 3, 4, 0, 42, include_context=False)[0]
    torch.randn(100)
    context = make_cub200_loaders(cub_root, 16, .4, 3, 4, 0, 42)[0]
    for a, b in zip(baseline, context):
        assert torch.equal(a["query_id"], b["query_id"])
        assert torch.equal(a["query"], b["query"])


def test_missing_dataset_explains_official_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.caltech.edu"):
        CUB200Images(tmp_path)


def test_rejects_inconsistent_metadata(cub_root):
    (cub_root / "image_class_labels.txt").write_text("1 1", encoding="utf8")
    with pytest.raises(ValueError, match="match across metadata"):
        CUB200Images(cub_root)


def test_rejects_image_path_escape(cub_root):
    path = cub_root / "images.txt"
    path.write_text(path.read_text().replace("bird_0.png", "../outside.png"), encoding="utf8")
    with pytest.raises(ValueError, match="escapes"):
        CUB200Images(cub_root)
