from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.contrastive_vectors import compute_contrastive_vector
from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.online_runtime import run_driftguard_session

logger = logging.getLogger(__name__)


def _is_better_candidate(
    *,
    auc: float,
    pr_auc: float,
    score_std: float,
    best_auc: float,
    best_pr_auc: float,
    best_score_std: float,
    eps: float = 1e-8,
) -> bool:
    """Stable tie-breaking for calibration search to avoid degenerate all-zero picks."""
    if auc > best_auc + eps:
        return True
    if abs(auc - best_auc) <= eps and pr_auc > best_pr_auc + eps:
        return True
    if (
        abs(auc - best_auc) <= eps
        and abs(pr_auc - best_pr_auc) <= eps
        and score_std > best_score_std + eps
    ):
        return True
    return False


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
    topo_weights: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15),
) -> np.ndarray:
    diam_s, p1_s, b0_s, b1_s = scales
    diam_w, p1_w, b0_w, b1_w = topo_weights
    topo = (
        diam_w * (raw[:, 0] / max(diam_s, 1e-6))
        + p1_w * (raw[:, 1] / max(p1_s, 1e-6))
        + b0_w * (raw[:, 2] / max(b0_s, 1e-6))
        + b1_w * (raw[:, 3] / max(b1_s, 1e-6))
    )
    return np.clip(topo, 0.0, 2.0)


def _compose_scaled_score(
    c_raw: np.ndarray,
    l_raw: np.ndarray,
    topo_scaled: np.ndarray,
    *,
    weights: tuple[float, float, float],
    cl_scales: tuple[float, float],
) -> np.ndarray:
    c_scale, l_scale = cl_scales
    continuity = c_raw / max(float(c_scale), 1e-6)
    lipschitz = l_raw / max(float(l_scale), 1e-6)
    return (
        weights[0] * continuity
        + weights[1] * lipschitz
        + weights[2] * topo_scaled
    )


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
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Persist safe trajectories for inference-only manifold reducers (e.g., UMAP).
    safe_prompts_for_reducer = [p for p, y_i in zip(prompts, labels) if int(y_i) == 0]
    if safe_prompts_for_reducer:
        safe_states: list[np.ndarray] = []
        for prompt in safe_prompts_for_reducer:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_input_tokens)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.no_grad():
                out_hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            h = out_hidden.hidden_states[cfg.layer_idx]
            if hasattr(h, "detach"):
                arr = h.detach().squeeze(0).to("cpu").float().numpy()
            else:
                arr = np.asarray(h).squeeze(0)
            if arr.ndim == 2 and arr.shape[0] > 0:
                safe_states.append(np.asarray(arr, dtype=np.float32))
        if safe_states:
            safe_matrix = np.concatenate(safe_states, axis=0).astype(np.float32)
            safe_path = out.parent / "calibration_safe_trajectories.npy"
            np.save(safe_path, safe_matrix)

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
    best_pr_auc = -1.0
    best_score_std = -1.0
    best_weights = (
        float(cfg.continuity_weight),
        float(cfg.lipschitz_weight),
        float(cfg.topology_weight),
    )
    best_scales = (
        float(cfg.continuity_scale),
        float(cfg.lipschitz_scale),
        float(cfg.topology_diameter_scale),
        float(cfg.persistence_l1_scale),
        float(cfg.beta0_scale),
        float(cfg.beta1_scale),
    )
    topo_sub_weights = (
        cfg.topology_diameter_weight,
        cfg.topology_persistence_l1_weight,
        cfg.topology_beta0_weight,
        cfg.topology_beta1_weight,
    )
    default_topology = _apply_scales(t_raw, best_scales[2:], topo_weights=topo_sub_weights)
    best_scores = np.clip(
        _compose_scaled_score(
            c_raw,
            l_raw,
            default_topology,
            weights=best_weights,
            cl_scales=best_scales[:2],
        ),
        0.0,
        1.5,
    )
    cv_auc_mean: float | None = None
    overfit_gap: float | None = None
    if len(np.unique(y)) >= 2:
        # Initialize from defaults so tie-cases don't collapse to first grid point.
        best_auc = float(roc_auc_score(y, best_scores))
        best_pr_auc = float(average_precision_score(y, best_scores))
        best_score_std = float(np.std(best_scores))

        c_grid = _quantile_scales(c_raw)
        l_grid = _quantile_scales(l_raw)
        diam_grid = _quantile_scales(t_raw[:, 0])
        p1_grid = _quantile_scales(t_raw[:, 1])
        b0_grid = _quantile_scales(t_raw[:, 2])
        b1_grid = _quantile_scales(t_raw[:, 3])

        # k-fold CV: evaluate candidates on held-out folds to avoid overfitting.
        n_folds = min(5, min(int(np.sum(y == 0)), int(np.sum(y == 1))))
        n_folds = max(n_folds, 2)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.random_state)
        folds = list(skf.split(c_raw, y))

        for c_s in c_grid:
            for l_s in l_grid:
                for diam_s in diam_grid:
                    for p1_s in p1_grid:
                        for b0_s in b0_grid:
                            for b1_s in b1_grid:
                                topo = _apply_scales(t_raw, (diam_s, p1_s, b0_s, b1_s), topo_weights=topo_sub_weights)
                                for c_w, l_w, t_w in _weight_candidates(step=0.1):
                                    candidate = np.clip(
                                        _compose_scaled_score(
                                            c_raw,
                                            l_raw,
                                            topo,
                                            weights=(c_w, l_w, t_w),
                                            cl_scales=(c_s, l_s),
                                        ),
                                        0.0,
                                        1.5,
                                    )
                                    # Use mean CV AUC as primary selection metric.
                                    fold_aucs = []
                                    for train_idx, val_idx in folds:
                                        y_val = y[val_idx]
                                        if len(np.unique(y_val)) < 2:
                                            continue
                                        fold_aucs.append(float(roc_auc_score(y_val, candidate[val_idx])))
                                    if not fold_aucs:
                                        continue
                                    auc = float(np.mean(fold_aucs))
                                    pr_auc = float(average_precision_score(y, candidate))
                                    cand_std = float(np.std(candidate))
                                    if _is_better_candidate(
                                        auc=auc,
                                        pr_auc=pr_auc,
                                        score_std=cand_std,
                                        best_auc=best_auc,
                                        best_pr_auc=best_pr_auc,
                                        best_score_std=best_score_std,
                                    ):
                                        best_auc = auc
                                        best_pr_auc = pr_auc
                                        best_score_std = cand_std
                                        best_weights = (c_w, l_w, t_w)
                                        best_scales = (c_s, l_s, diam_s, p1_s, b0_s, b1_s)
                                        best_scores = candidate

        # Compute overfitting gap: train AUC vs mean CV (validation) AUC.
        train_auc = float(roc_auc_score(y, best_scores))
        cv_fold_aucs = []
        for train_idx, val_idx in folds:
            y_val = y[val_idx]
            if len(np.unique(y_val)) < 2:
                continue
            cv_fold_aucs.append(float(roc_auc_score(y_val, best_scores[val_idx])))
        cv_auc_mean = float(np.mean(cv_fold_aucs)) if cv_fold_aucs else train_auc
        overfit_gap = train_auc - cv_auc_mean

    result = summarize_calibration_from_scores(
        scores=best_scores,
        labels=labels,
        cfg=cfg,
    )
    if cv_auc_mean is not None:
        result["cv_auc_mean"] = float(cv_auc_mean)
    if overfit_gap is not None:
        result["overfit_gap"] = float(overfit_gap)
    result["optimal_weights"] = {
        "continuity": best_weights[0],
        "lipschitz": best_weights[1],
        "topology": best_weights[2],
    }
    result["optimal_scales"] = {
        "continuity_scale": best_scales[0],
        "lipschitz_scale": best_scales[1],
        "topology_diameter_scale": best_scales[2],
        "persistence_l1_scale": best_scales[3],
        "beta0_scale": best_scales[4],
        "beta1_scale": best_scales[5],
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
        "continuity_scale": best_scales[0],
        "lipschitz_scale": best_scales[1],
        "topology_diameter_scale": best_scales[2],
        "persistence_l1_scale": best_scales[3],
        "beta0_scale": best_scales[4],
        "beta1_scale": best_scales[5],
    }
    best_topology = _apply_scales(t_raw, best_scales[2:], topo_weights=topo_sub_weights)
    c_scaled = c_raw / max(best_scales[0], 1e-6)
    l_scaled = l_raw / max(best_scales[1], 1e-6)
    result["risk_component_maxima"] = {
        "continuity_raw_max_per_prompt": c_raw.astype(float).tolist(),
        "lipschitz_raw_max_per_prompt": l_raw.astype(float).tolist(),
        "continuity_scaled_max_per_prompt": c_scaled.astype(float).tolist(),
        "lipschitz_scaled_max_per_prompt": l_scaled.astype(float).tolist(),
        "topology_scaled_max_per_prompt": best_topology.astype(float).tolist(),
    }
    result["risk_component_contributions_at_optimal_weights"] = {
        "continuity": (best_weights[0] * c_scaled).astype(float).tolist(),
        "lipschitz": (best_weights[1] * l_scaled).astype(float).tolist(),
        "topology": (best_weights[2] * best_topology).astype(float).tolist(),
    }
    safe_prompts = cfg.safe_prompts or [p for p, y_i in zip(prompts, labels) if int(y_i) == 0]
    harmful_prompts = cfg.harmful_prompts or [p for p, y_i in zip(prompts, labels) if int(y_i) != 0]
    layer_key = f"layer_{cfg.layer_idx}"
    if safe_prompts and harmful_prompts:
        contrastive_vectors: dict[str, list[float] | None] = {}
        try:
            vec = compute_contrastive_vector(
                model=model,
                tokenizer=tokenizer,
                safe_prompts=safe_prompts,
                harmful_prompts=harmful_prompts,
                layer_idx=cfg.layer_idx,
                cfg=cfg,
                device=device,
            )
            contrastive_vectors[layer_key] = vec.astype(float).tolist()
        except Exception as e:
            print(f"⚠️  Contrastive vector extraction failed for layer {cfg.layer_idx}: {e}")
            contrastive_vectors[layer_key] = None
        result["contrastive_vectors"] = contrastive_vectors
    else:
        result["contrastive_vectors"] = {}
    out.write_text(json.dumps(result, indent=2))
    if safe_prompts_for_reducer:
        safe_path = out.parent / "calibration_safe_trajectories.npy"
        if safe_path.exists():
            result["safe_trajectories_path"] = str(safe_path)
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
