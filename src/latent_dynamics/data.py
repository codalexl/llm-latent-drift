from __future__ import annotations

from typing import Any

import numpy as np
from datasets import (
    Dataset,
    load_dataset,
)

from latent_dynamics.config import DATASET_REGISTRY, TOY_CONTRASTIVE, DatasetSpec

STRATIFY_SEED = 42

_HF_FETCH_HINT = (
    "Hugging Face dataset download failed.\n"
    "- Proxy 403 / ProxyError: unset HTTP_PROXY and HTTPS_PROXY for this run, or add "
    "huggingface.co to NO_PROXY, or set LATENT_DYNAMICS_BYPASS_PROXY_FOR_HF=1 in .env "
    "(loaded by latent_dynamics._env).\n"
    "- Offline: prefetch the dataset, then run with HF_HUB_OFFLINE=1."
)


def _load_dataset_remote(*args: object, **kwargs: object) -> Any:
    try:
        return load_dataset(*args, **kwargs)  # type: ignore[arg-type]
    except Exception as e:
        raise RuntimeError(f"{_HF_FETCH_HINT}\nCaused by: {e!r}") from e


def _label_from_row(row: dict[str, Any], spec: DatasetSpec) -> int | None:
    if spec.label_field is not None:
        return int(row[spec.label_field])
    if spec.label_fn is not None:
        return int(spec.label_fn(row))
    return None


def _balanced_sample_by_label(
    ds: Dataset,
    spec: DatasetSpec,
    max_samples: int,
    seed: int,
) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds

    buckets: dict[int, list[int]] = {}
    for idx, row in enumerate(ds):
        label = _label_from_row(row, spec)
        if label is None:
            continue
        buckets.setdefault(label, []).append(idx)

    if len(buckets) < 2:
        print(
            "Stratified sampling requested but this split has <2 labels; "
            "falling back to first max_samples rows."
        )
        return ds.select(range(max_samples))

    rng = np.random.default_rng(seed)
    labels = sorted(buckets.keys())
    for lab in labels:
        rng.shuffle(buckets[lab])

    per_label = max_samples // len(labels)
    remainder = max_samples % len(labels)
    selected: list[int] = []
    leftovers: list[int] = []
    for i, lab in enumerate(labels):
        target = per_label + (1 if i < remainder else 0)
        take = min(target, len(buckets[lab]))
        selected.extend(buckets[lab][:take])
        leftovers.extend(buckets[lab][take:])

    needed = max_samples - len(selected)
    if needed > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:needed])

    if len(selected) < max_samples:
        print(
            f"Only {len(selected)} labelled rows available for stratified sampling; "
            f"requested {max_samples}."
        )

    rng.shuffle(selected)
    return ds.select(selected)


def load_examples(
    dataset_key: str,
    max_samples: int | None = None,
    stratify_labels: bool = False,
    seed: int = STRATIFY_SEED,
) -> tuple[Dataset, DatasetSpec]:
    spec = DATASET_REGISTRY[dataset_key]

    if spec.path == "toy_contrastive":
        ds = Dataset.from_list(TOY_CONTRASTIVE)
    elif spec.path == "allenai/wildjailbreak":
        ds = _load_dataset_remote(
            spec.path,
            name=spec.split,
            split="train",
            delimiter="\t",
            keep_default_na=False,
        )
    else:
        ds = _load_dataset_remote(spec.path, split=spec.split)

    if max_samples and len(ds) > max_samples:
        if stratify_labels:
            ds = _balanced_sample_by_label(
                ds,
                spec,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(ds), size=max_samples, replace=False)
            ds = ds.select(indices.tolist())

    return ds, spec


def _prepare_wildjailbreak_text_and_labels(
    ds: Dataset,
    spec: DatasetSpec,
    return_metadata: bool = False,
) -> tuple[list[str], np.ndarray | None, list[dict[str, Any]] | None]:
    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for row in ds:
        input_label = str(row["data_type"]).strip().lower()
        if input_label == "vanilla_benign":
            label = 0
        elif input_label == "vanilla_harmful":
            label = 1
        elif input_label == "adversarial_benign":
            label = 0
        elif input_label == "adversarial_harmful":
            label = 1
        else:
            raise ValueError(f"Unknown label: {input_label}")

        if "vanilla" in input_label:
            text = row["vanilla"]
        elif "adversarial" in input_label:
            text = row["adversarial"]
        else:
            raise ValueError(f"Unknown label: {input_label}")

        texts.append(str(text))
        labels.append(label)
        metadata.append(
            {
                "data_type": row["data_type"],
                "vanilla_text": row["vanilla"],
            }
        )
    if return_metadata:
        return texts, np.array(labels, dtype=np.int64), metadata
    return texts, np.array(labels, dtype=np.int64)


def prepare_text_and_labels(
    ds: Dataset,
    spec: DatasetSpec,
    return_metadata: bool = False,
) -> tuple[list[str], np.ndarray | None, list[dict[str, Any]] | None]:
    if spec.path == "allenai/wildjailbreak":
        return _prepare_wildjailbreak_text_and_labels(ds, spec, return_metadata)

    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []

    for row in ds:
        text = row[spec.text_field]
        if text is None:
            raise ValueError(f"Text is None for row {row}")
        texts.append(str(text))

        if spec.label_field is not None:
            labels.append(int(row[spec.label_field]))
        elif spec.label_fn is not None:
            labels.append(int(spec.label_fn(row)))

    if len(labels) == 0:
        return texts, None

    if return_metadata:
        return texts, np.array(labels, dtype=np.int64), metadata

    return texts, np.array(labels, dtype=np.int64)


if __name__ == "__main__":
    ds, spec = load_examples("wildjailbreak", max_samples=1000, stratify_labels=True)
    texts, labels, metadata = prepare_text_and_labels(ds, spec, return_metadata=True)
    breakpoint()
