"""Deterministic, on-disk train/val/test split of the MSD Task01_BrainTumour case list.

MONAI's DecathlonDataset only separates "training" (labeled) from "test" (unlabeled,
challenge-only) per dataset.json. Since this project needs a labeled test set for
evaluation, we build our own split over the labeled cases and persist it so the same
split is reused across every run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CaseList = list[dict[str, str]]


def case_id(entry: dict[str, str]) -> str:
    """Extract a stable case identifier from a datalist entry's image path."""
    return Path(entry["image"]).name


def make_splits(
    datalist: CaseList,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Partition case IDs into train/val/test with no overlap, shuffled by `seed`."""
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    ids = [case_id(entry) for entry in datalist]
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def apply_split(datalist: CaseList, ids: list[str]) -> CaseList:
    wanted = set(ids)
    return [entry for entry in datalist if case_id(entry) in wanted]


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())


def build_full_datalist(root_dir: str, download: bool = True) -> CaseList:
    """Download (if needed) and return the full labeled MSD Task01 case list."""
    from monai.apps import DecathlonDataset
    from monai.transforms import Compose

    Path(root_dir).mkdir(parents=True, exist_ok=True)
    dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=Compose([]),
        download=download,
        val_frac=0.0,
        seed=0,
    )
    return list(dataset.data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True, help="MSD dataset root directory")
    parser.add_argument("--out", required=True, help="Path to write the split JSON")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    datalist = build_full_datalist(args.root_dir, download=not args.no_download)
    splits = make_splits(
        datalist, args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    save_splits(splits, args.out)
    print(
        f"wrote split to {args.out}: "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
