from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.online_runtime import run_driftguard_session


def _threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.clip(scores, 0.0, 1.5))
    if candidates.size == 0:
        return 0.5
    best_t = float(candidates[0])
    best_j = -1.0
    for t in candidates:
        preds = (scores >= t).astype(np.int64)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def calibrate_risk_score(
    cfg: DriftGuardConfig,
    model: AutoModelForCausalLM | object,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    labels: list[int],
    device: str,
    output_path: Path | str = "calibration_results.json",
) -> dict[str, Any]:
    """Run labeled prompts and fit threshold diagnostics for online risk scores."""
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must have equal length.")
    if not prompts:
        raise ValueError("calibration requires at least one prompt.")

    per_prompt_max_risk: list[float] = []
    for prompt in prompts:
        session = run_driftguard_session(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            safe_reference=None,
        )
        max_risk = max((step.risk_score for step in session.steps), default=0.0)
        per_prompt_max_risk.append(float(max_risk))

    result = summarize_calibration_from_scores(
        scores=per_prompt_max_risk,
        labels=labels,
        cfg=cfg,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def summarize_calibration_from_scores(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    cfg: DriftGuardConfig,
) -> dict[str, Any]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must have the same length.")
    if x.shape[0] == 0:
        raise ValueError("scores must be non-empty.")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration requires at least two classes in labels.")

    roc_auc = float(roc_auc_score(y, x))
    pr_auc = float(average_precision_score(y, x))
    threshold = _best_threshold(x, y)
    precision, recall = _threshold_metrics(x, y, threshold)

    return {
        "n_samples": int(len(x)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": float(threshold),
        "precision_at_best_threshold": float(precision),
        "recall_at_best_threshold": float(recall),
        "score_summary": {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        },
        "weights": {
            "continuity": float(cfg.continuity_weight),
            "lipschitz": float(cfg.lipschitz_weight),
            "topology": float(cfg.topology_weight),
        },
    }
