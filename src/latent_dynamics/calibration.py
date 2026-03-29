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


def _weight_candidates(step: float = 0.1) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    grid = np.arange(0.0, 1.0 + step / 2.0, step)
    for c in grid:
        for l in grid:
            t = 1.0 - c - l
            if t < -1e-9:
                continue
            out.append((float(c), float(l), float(max(t, 0.0))))
    return out


def _decompose_risk(step: Any, cfg: DriftGuardConfig) -> tuple[float, float, float]:
    continuity = 0.0
    if step.cosine_continuity is not None:
        continuity = max(0.0, cfg.cosine_floor - float(step.cosine_continuity)) / max(
            1.0 - cfg.cosine_floor,
            1e-6,
        )
    lipschitz = 0.0
    if step.lipschitz_proxy is not None:
        lipschitz = max(0.0, float(step.lipschitz_proxy) - cfg.lipschitz_ceiling) / max(
            cfg.lipschitz_ceiling,
            1e-6,
        )
    topology = 0.0
    if step.topology_diameter is not None:
        topology += 0.40 * (max(float(step.topology_diameter), 0.0) / max(cfg.topology_diameter_scale, 1e-6))
    if step.topology_persistence_l1 is not None:
        topology += 0.30 * (
            max(float(step.topology_persistence_l1), 0.0) / max(cfg.persistence_l1_scale, 1e-6)
        )
    if step.topology_beta0 is not None:
        topology += 0.15 * (max(float(step.topology_beta0), 0.0) / max(cfg.beta0_scale, 1e-6))
    if step.topology_beta1 is not None:
        topology += 0.15 * (max(float(step.topology_beta1), 0.0) / max(cfg.beta1_scale, 1e-6))
    return continuity, lipschitz, min(topology, 2.0)


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
    per_prompt_components: list[tuple[float, float, float]] = []
    for prompt in prompts:
        session = run_driftguard_session(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            safe_reference=None,
        )
        comps = [_decompose_risk(step, cfg) for step in session.steps]
        if comps:
            c_vals = [c for c, _, _ in comps]
            l_vals = [l for _, l, _ in comps]
            t_vals = [t for _, _, t in comps]
            per_prompt_components.append((max(c_vals), max(l_vals), max(t_vals)))
        else:
            per_prompt_components.append((0.0, 0.0, 0.0))
        max_risk = max((step.risk_score for step in session.steps), default=0.0)
        per_prompt_max_risk.append(float(max_risk))

    y = np.asarray(labels, dtype=np.int64)
    comp = np.asarray(per_prompt_components, dtype=np.float64)
    best_auc = -1.0
    best_weights = (
        float(cfg.continuity_weight),
        float(cfg.lipschitz_weight),
        float(cfg.topology_weight),
    )
    best_scores = np.asarray(per_prompt_max_risk, dtype=np.float64)
    if len(np.unique(y)) >= 2:
        for c_w, l_w, t_w in _weight_candidates(step=0.1):
            candidate = np.clip(c_w * comp[:, 0] + l_w * comp[:, 1] + t_w * comp[:, 2], 0.0, 1.5)
            auc = float(roc_auc_score(y, candidate))
            if auc > best_auc:
                best_auc = auc
                best_weights = (c_w, l_w, t_w)
                best_scores = candidate

    result = summarize_calibration_from_scores(
        scores=best_scores,
        labels=labels,
        cfg=cfg,
    )
    result["optimal_weights"] = {
        "continuity": best_weights[0],
        "lipschitz": best_weights[1],
        "topology": best_weights[2],
    }
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
