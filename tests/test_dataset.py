from pathlib import Path

from src.data.dataset import build_datasets
from src.data.splits import case_id, make_splits, save_splits


def test_build_datasets_respects_saved_split(tmp_path: Path, synthetic_datalist, monkeypatch):
    splits = make_splits(
        synthetic_datalist, train_frac=1 / 3, val_frac=1 / 3, test_frac=1 / 3, seed=0
    )
    split_path = tmp_path / "split.json"
    save_splits(splits, split_path)

    monkeypatch.setattr(
        "src.data.dataset.build_full_datalist",
        lambda root_dir, download=True: synthetic_datalist,
    )

    train_ds, val_ds, test_ds = build_datasets(
        root_dir="unused",
        splits_path=str(split_path),
        patch_size=(16, 16, 16),
        num_samples=1,
        pos=1,
        neg=1,
        download=False,
    )

    assert [case_id(e) for e in train_ds.data] == splits["train"]
    assert [case_id(e) for e in val_ds.data] == splits["val"]
    assert [case_id(e) for e in test_ds.data] == splits["test"]
