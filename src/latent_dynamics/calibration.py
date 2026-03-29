from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
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


def _quantile_scales(values: np.ndarray) -> list[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 1e-8]
    if vals.size == 0:
        return [1.0]
    quants = np.quantile(vals, [0.5, 0.75, 0.9])
    out = sorted({float(max(q, 1e-4)) for q in quants})
    return out


def _decompose_risk_raw(step: Any, cfg: DriftGuardConfig) -> tuple[float, float, np.ndarray]:
    """Return unscaled per-step risk components for calibration search."""
    # Raw continuity/lipschitz terms before scale normalization.
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
    topo_terms = np.zeros(4, dtype=np.float64)
    if step.topology_diameter is not None:
        topo_terms[0] = max(float(step.topology_diameter), 0.0)
    if step.topology_persistence_l1 is not None:
        topo_terms[1] = max(float(step.topology_persistence_l1), 0.0)
    if step.topology_beta0 is not None:
        topo_terms[2] = max(float(step.topology_beta0), 0.0)
    if step.topology_beta1 is not None:
        topo_terms[3] = max(float(step.topology_beta1), 0.0)
    return continuity, lipschitz, topo_terms


def _apply_scales(
    raw: np.ndarray,
    scales: tuple[float, float, float, float],
) -> np.ndarray:
    diam_s, p1_s, b0_s, b1_s = scales
    # Keep the same topology decomposition weights, but learn the scales from labels.
    topo = (
        0.40 * (raw[:, 0] / max(diam_s, 1e-6))
        + 0.30 * (raw[:, 1] / max(p1_s, 1e-6))
        + 0.15 * (raw[:, 2] / max(b0_s, 1e-6))
        + 0.15 * (raw[:, 3] / max(b1_s, 1e-6))
    )
    return np.clip(topo, 0.0, 2.0)


def calibrate_risk_score(
    cfg: DriftGuardConfig,
    model: AutoModelForCausalLM | object,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    labels: list[int],
    device: str,
    output_path: Path | str = "calibration_results.json",
) -> dict[str, Any]:
    """Run labeled prompts and fit weights/scales/threshold from labels."""
    if len(prompts) != len(labels):
        raise ValueError("prompts and labels must have equal length.")
    if not prompts:
        raise ValueError("calibration requires at least one prompt.")

    prompt_components: list[tuple[float, float, np.ndarray]] = []
    for prompt in prompts:
        session = run_driftguard_session(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            safe_reference=None,
        )
        comps = [_decompose_risk_raw(step, cfg) for step in session.steps]
        if not comps:
            prompt_components.append((0.0, 0.0, np.zeros(4, dtype=np.float64)))
            continue
        c_vals = [c for c, _, _ in comps]
        l_vals = [l for _, l, _ in comps]
        topo = np.stack([t for _, _, t in comps], axis=0)
        prompt_components.append(
            (
                float(np.max(c_vals)),
                float(np.max(l_vals)),
                np.max(topo, axis=0).astype(np.float64),
            )
        )

    y = np.asarray(labels, dtype=np.int64)
    c_raw = np.asarray([x[0] for x in prompt_components], dtype=np.float64)
    l_raw = np.asarray([x[1] for x in prompt_components], dtype=np.float64)
    t_raw = np.stack([x[2] for x in prompt_components], axis=0)
    best_auc = -1.0
    best_weights = (
        float(cfg.continuity_weight),
        float(cfg.lipschitz_weight),
        float(cfg.topology_weight),
    )
    best_scales = (
        float(cfg.topology_diameter_scale),
        float(cfg.persistence_l1_scale),
        float(cfg.beta0_scale),
        float(cfg.beta1_scale),
    )
    default_topology = _apply_scales(t_raw, best_scales)
    best_scores = np.clip(
        best_weights[0] * c_raw + best_weights[1] * l_raw + best_weights[2] * default_topology,
        0.0,
        1.5,
    )
    if len(np.unique(y)) >= 2:
        diam_grid = _quantile_scales(t_raw[:, 0])
        p1_grid = _quantile_scales(t_raw[:, 1])
        b0_grid = _quantile_scales(t_raw[:, 2])
        b1_grid = _quantile_scales(t_raw[:, 3])

        for diam_s in diam_grid:
            for p1_s in p1_grid:
                for b0_s in b0_grid:
                    for b1_s in b1_grid:
                        topo = _apply_scales(t_raw, (diam_s, p1_s, b0_s, b1_s))
                        for c_w, l_w, t_w in _weight_candidates(step=0.1):
                            candidate = np.clip(
                                c_w * c_raw + l_w * l_raw + t_w * topo,
                                0.0,
                                1.5,
                            )
                            auc = float(roc_auc_score(y, candidate))
                            if auc > best_auc:
                                best_auc = auc
                                best_weights = (c_w, l_w, t_w)
                                best_scales = (diam_s, p1_s, b0_s, b1_s)
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
    result["optimal_scales"] = {
        "topology_diameter_scale": best_scales[0],
        "persistence_l1_scale": best_scales[1],
        "beta0_scale": best_scales[2],
        "beta1_scale": best_scales[3],
    }
    if len(np.unique(y)) >= 2:
        fpr, tpr, roc_th = roc_curve(y, best_scores)
        pr_prec, pr_rec, pr_th = precision_recall_curve(y, best_scores)
        result["roc_curve"] = {
            "fpr": fpr.astype(float).tolist(),
            "tpr": tpr.astype(float).tolist(),
            "thresholds": roc_th.astype(float).tolist(),
        }
        result["pr_curve"] = {
            "precision": pr_prec.astype(float).tolist(),
            "recall": pr_rec.astype(float).tolist(),
            "thresholds": pr_th.astype(float).tolist(),
        }
    result["recommended_config_update"] = {
        "risk_threshold": float(result["best_threshold"]),
        "continuity_weight": best_weights[0],
        "lipschitz_weight": best_weights[1],
        "topology_weight": best_weights[2],
        "topology_diameter_scale": best_scales[0],
        "persistence_l1_scale": best_scales[1],
        "beta0_scale": best_scales[2],
        "beta1_scale": best_scales[3],
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
