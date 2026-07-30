from pathlib import Path

from src.data.splits import apply_split, case_id, load_splits, make_splits, save_splits


def _fake_datalist(n: int) -> list[dict[str, str]]:
    return [
        {"image": f"BRATS_{i:03d}.nii.gz", "label": f"BRATS_{i:03d}_seg.nii.gz"} for i in range(n)
    ]


def test_make_splits_covers_every_case_with_no_overlap() -> None:
    datalist = _fake_datalist(20)
    splits = make_splits(datalist, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)

    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_ids) == sorted(case_id(e) for e in datalist)
    assert len(set(all_ids)) == len(all_ids)


def test_make_splits_is_deterministic_for_a_fixed_seed() -> None:
    datalist = _fake_datalist(20)
    first = make_splits(datalist, 0.7, 0.15, 0.15, seed=42)
    second = make_splits(datalist, 0.7, 0.15, 0.15, seed=42)
    assert first == second


def test_make_splits_rejects_fractions_that_dont_sum_to_one() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_splits(_fake_datalist(10), 0.5, 0.5, 0.5, seed=0)


def test_apply_split_filters_by_case_id() -> None:
    datalist = _fake_datalist(5)
    subset_ids = [case_id(datalist[0]), case_id(datalist[2])]
    subset = apply_split(datalist, subset_ids)
    assert [case_id(e) for e in subset] == subset_ids


def test_save_and_load_splits_round_trip(tmp_path: Path) -> None:
    splits = {"train": ["a.nii.gz"], "val": ["b.nii.gz"], "test": ["c.nii.gz"]}
    out_path = tmp_path / "nested" / "split.json"
    save_splits(splits, out_path)
    assert load_splits(out_path) == splits
