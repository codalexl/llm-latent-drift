from __future__ import annotations

from typing import Any

import numpy as np
from datasets import (
    Dataset,
    load_dataset,
)

from latent_dynamics.config import DATASET_REGISTRY, DatasetSpec

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


def _load_wildjailbreak_split(split: str = "train") -> Any:
    """Load WildJailbreak robustly across hub dataset layout changes."""
    try:
        return _load_dataset_remote(
            "allenai/wildjailbreak",
            data_files={split: f"{split}/{split}.tsv"},
            split=split,
            delimiter="\t",
            keep_default_na=False,
        )
    except RuntimeError as e:
        # Backward-compat fallback for older datasets API behavior.
        if "data_files=None" not in str(e):
            raise
        return _load_dataset_remote("allenai/wildjailbreak", split=split)


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

    if spec.path == "allenai/wildjailbreak":
        ds = _load_wildjailbreak_split(split="train")
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


_SYNTHETIC_MULTITURN_PREFIXES: tuple[str, ...] = (
    "User: Hi — can you help me think through something?\n"
    "Assistant: Sure, what are you working on?\n",
    "User: Thanks for being here. I need a bit of advice.\n"
    "Assistant: Happy to help — what’s on your mind?\n",
    "User: Quick question before we dive in.\n"
    "Assistant: Of course, go ahead.\n",
    "User: I’m brainstorming and could use a sounding board.\n"
    "Assistant: Great — what topic should we focus on?\n",
)


def _wrap_wildjailbreak_adversarial_as_multiturn(
    adversarial_prompt: str,
    rng: np.random.Generator,
) -> str:
    """Prefix a jailbreak user utterance with a short benign two-turn chat."""
    prefix = str(rng.choice(list(_SYNTHETIC_MULTITURN_PREFIXES)))
    body = adversarial_prompt.strip()
    return prefix + f"User: {body}"


def _load_wildjailbreak_adversarial_harmful(
    rng: np.random.Generator,
    n: int,
) -> list[str]:
    spec = DATASET_REGISTRY["wildjailbreak"]
    wj = _load_wildjailbreak_split(split="train")

    def _row_is_adversarial_harmful(example: dict[str, Any]) -> bool:
        return (
            str(example.get("data_type", "")).strip().lower() == "adversarial_harmful"
        )

    wj_h = wj.filter(_row_is_adversarial_harmful)
    if len(wj_h) < n:
        raise ValueError(
            f"Need {n} adversarial_harmful WildJailbreak rows; only {len(wj_h)} available."
        )
    pick = rng.choice(len(wj_h), size=n, replace=False)
    out: list[str] = []
    for i in pick.tolist():
        row = wj_h[int(i)]
        adv = row.get("adversarial")
        if adv is None or not str(adv).strip():
            raise ValueError("WildJailbreak adversarial_harmful row has empty adversarial text.")
        out.append(_wrap_wildjailbreak_adversarial_as_multiturn(str(adv), rng))
    return out


def _sample_wildchat_non_toxic_sessions(
    rng: np.random.Generator,
    n: int,
    max_scan: int,
) -> list[str]:
    """Real multi-turn sessions from public WildChat-1M (non-toxic release)."""
    wc = _load_dataset_remote("allenai/WildChat-1M", split="train")
    if len(wc) == 0:
        raise ValueError("WildChat-1M split is empty.")
    scan = min(len(wc), max_scan)
    idx = rng.choice(len(wc), size=scan, replace=False)
    out: list[str] = []
    for i in idx.tolist():
        row = wc[int(i)]
        if row.get("toxic") is True:
            continue
        text = _format_wildchat_1m_session(row.get("conversation") or [])
        if not text.strip():
            continue
        out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(
            f"Only collected {len(out)} non-toxic WildChat sessions "
            f"after scanning {scan} rows (need {n})."
        )
    return out


def load_wildjailbreak_adversarial_harmful(n: int, seed: int = 42) -> list[str]:
    """WildJailbreak ``adversarial_harmful`` rows as synthetic multi-turn prompts (see hybrid benchmark)."""
    if n <= 0:
        return []
    return _load_wildjailbreak_adversarial_harmful(np.random.default_rng(seed), n)


def load_wildchat_benign_multi_turn(n: int, seed: int = 42) -> list[str]:
    """Non-toxic multi-turn sessions sampled from WildChat-1M (same protocol as hybrid benchmark)."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    pool_scan = max(n * 64, 4096)
    return _sample_wildchat_non_toxic_sessions(rng, n, max_scan=pool_scan)


def load_wildchat_multi_turn_preset(
    max_samples: int = 400,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Half unsafe (WildJailbreak adversarial harmful as multi-turn) and half benign (WildChat-1M).

    Replaces the old silent dummy fallback; prompts and labels are jointly shuffled with ``seed``.
    """
    rng = np.random.default_rng(seed)
    half = max(2, max_samples // 2)
    pool_scan = max(half * 64, 4096)
    unsafe = _load_wildjailbreak_adversarial_harmful(rng, half)
    benign = _sample_wildchat_non_toxic_sessions(rng, half, max_scan=pool_scan)
    prompts = unsafe + benign
    labels = [1] * len(unsafe) + [0] * len(benign)
    idx = rng.permutation(len(prompts))
    return [prompts[int(i)] for i in idx.tolist()], [labels[int(i)] for i in idx.tolist()]


def load_hybrid_wildchat_jailbreak_benchmark_sessions(
    max_cases: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """Benchmark mix: benign = WildChat-1M multi-turn; unsafe = WildJailbreak adversarial harmful in synthetic chat.

    Class 0: real `allenai/WildChat-1M` sessions (non-toxic hub release; rows with ``toxic`` skipped).
    Class 1: `allenai/wildjailbreak` rows with ``data_type == adversarial_harmful``, adversarial
    column wrapped with a short benign two-turn preamble. Sessions and labels are shuffled together.
    """
    texts, labels = load_wildchat_multi_turn_preset(max_cases, seed)
    return texts, np.array(labels, dtype=np.int64)


def _format_wildchat_1m_session(conversation: list[Any]) -> str:
    parts: list[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role_raw = turn.get("role", "user")
        role = str(role_raw).strip().lower()
        content = turn.get("content")
        text = "" if content is None else str(content).strip()
        if role == "user":
            parts.append(f"User: {text}")
        elif role == "assistant":
            parts.append(f"Assistant: {text}")
        else:
            parts.append(f"{role_raw}: {text}")
    return "\n".join(parts).strip()


def _prepare_wildchat_1m_text_and_labels(
    ds: Dataset,
    spec: DatasetSpec,
    return_metadata: bool = False,
) -> tuple[list[str], np.ndarray | None, list[dict[str, Any]] | None]:
    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for row in ds:
        conv = row.get("conversation") or []
        text = _format_wildchat_1m_session(conv)
        if not text:
            text = "[empty conversation]"
        label = _label_from_row(dict(row), spec)
        if label is None:
            continue
        texts.append(text)
        labels.append(int(label))
        if return_metadata:
            metadata.append(
                {
                    "conversation_hash": row.get("conversation_hash"),
                    "toxic": row.get("toxic"),
                    "redacted": row.get("redacted"),
                    "turn": row.get("turn"),
                }
            )
    if return_metadata:
        return texts, np.array(labels, dtype=np.int64), metadata
    return texts, np.array(labels, dtype=np.int64)


def _prepare_wildjailbreak_text_and_labels(
    ds: Dataset,
    spec: DatasetSpec,
    return_metadata: bool = False,
) -> tuple[list[str], np.ndarray | None, list[dict[str, Any]] | None]:
    texts: list[str] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for row in ds:
        dt = str(row.get("data_type", "")).strip().lower()
        if "vanilla" in dt:
            raw = row.get("vanilla")
        elif "adversarial" in dt:
            raw = row.get("adversarial")
        else:
            raise ValueError(f"Unknown WildJailbreak data_type: {dt!r}")
        if raw is None:
            raise ValueError(f"Empty text for data_type={dt!r}")
        label = _label_from_row(dict(row), spec)
        if label is None:
            continue
        texts.append(str(raw))
        labels.append(int(label))
        if return_metadata:
            metadata.append({"data_type": row.get("data_type")})
    if return_metadata:
        return texts, np.array(labels, dtype=np.int64), metadata
    return texts, np.array(labels, dtype=np.int64)


def prepare_text_and_labels(
    ds: Dataset,
    spec: DatasetSpec,
    return_metadata: bool = False,
) -> tuple[list[str], np.ndarray | None, list[dict[str, Any]] | None]:
    if spec.path == "allenai/WildChat-1M":
        return _prepare_wildchat_1m_text_and_labels(ds, spec, return_metadata)

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
    ds, spec = load_examples("wildchat", max_samples=1000, stratify_labels=True)
    texts, labels, metadata = prepare_text_and_labels(ds, spec, return_metadata=True)
    breakpoint()
