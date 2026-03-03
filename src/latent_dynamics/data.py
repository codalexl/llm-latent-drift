from __future__ import annotations

from typing import Any, Callable

import numpy as np
from datasets import Dataset, load_dataset

from latent_dynamics.config import DATASET_REGISTRY, TOY_CONTRASTIVE, DatasetSpec


def load_examples(
    dataset_key: str,
    split: str = "train",
    max_samples: int | None = None,
) -> tuple[Dataset, DatasetSpec]:
    spec = DATASET_REGISTRY[dataset_key]

    if spec.loader == "toy":
        ds = Dataset.from_list(TOY_CONTRASTIVE)
    else:
        ds_all = load_dataset(spec.path, spec.name)
        if split not in ds_all:
            split = next(iter(ds_all.keys()))
            print(f"Requested split not present. Using split='{split}'.")
        ds = ds_all[split]

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    return ds, spec


def make_70_15_15_split(
    ds: Dataset,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return train, calib, test splits with 70/15/15 proportion."""
    if len(ds) < 3:
        raise ValueError("Dataset too small for 70/15/15 split.")

    # First: 70% train, 30% temp.
    train_test = ds.train_test_split(test_size=0.30, seed=seed)
    train_ds = train_test["train"]
    temp_ds = train_test["test"]

    # Second: split temp into 15% calib, 15% test (equal halves of remaining 30%).
    calib_test = temp_ds.train_test_split(test_size=0.5, seed=seed)
    calib_ds = calib_test["train"]
    test_ds = calib_test["test"]

    return train_ds, calib_ds, test_ds


def prepare_text_and_labels(
    ds: Dataset,
    text_field: str,
    label_field: str | None = None,
    label_fn: Callable[[dict[str, Any]], int] | None = None,
) -> tuple[list[str], np.ndarray | None]:
    texts: list[str] = []
    labels: list[int] = []

    for row in ds:
        text = row[text_field]
        if text is None:
            continue
        texts.append(str(text))

        if label_field is not None:
            labels.append(int(row[label_field]))
        elif label_fn is not None:
            labels.append(int(label_fn(row)))

    if len(labels) == 0:
        return texts, None
    return texts, np.array(labels, dtype=np.int64)


def inspect_schema(ds: Dataset, n: int = 3) -> None:
    print("Rows:", len(ds))
    print("Columns:", ds.column_names)
    for i in range(min(n, len(ds))):
        print(f"[{i}]", {k: ds[i][k] for k in ds.column_names})
