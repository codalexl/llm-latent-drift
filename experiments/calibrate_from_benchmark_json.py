#!/usr/bin/env python3
"""Run DriftGuard calibration grid search on pre-computed benchmark JSON.

Unlike calibration.py (which re-runs the model), this script operates entirely
on the per-step telemetry already stored in the benchmark JSON output. No GPU
required. Use this after a full benchmark run to find optimal weights/scales/
threshold without spending another N hours of inference.

Usage:
    python experiments/calibrate_from_benchmark_json.py \
        --report-json experiments/outputs/driftguard_proper_v2_gemma3_4b.json \
        --out-json experiments/outputs/calibration_from_json_gemma3_4b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weight_candidates(step: float = 0.1) -> list[tuple[float, float, float]]:
    out = []
    grid = np.arange(0.0, 1.0 + step / 2, step)
    for c in grid:
        for l in grid:
            t = round(1.0 - c - l, 10)
            if t < -1e-9:
                continue
            out.append((float(c), float(l), float(max(t, 0.0))))
    return out


def _quantile_scales(values: np.ndarray, quantiles: tuple = (0.5, 0.75, 0.9)) -> list[float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 1e-8)]
    if vals.size == 0:
        return [1.0]
    qs = np.quantile(vals, list(quantiles))
    return sorted({float(max(q, 1e-4)) for q in qs})


def _apply_topo_scales(
    t_raw: np.ndarray,
    scales: tuple[float, float, float, float],
    sub_weights: tuple[float, float, float, float] = (0.40, 0.30, 0.15, 0.15),
) -> np.ndarray:
    dw, pw, b0w, b1w = sub_weights
    ds, ps, b0s, b1s = scales
    topo = (
        dw * (t_raw[:, 0] / max(ds, 1e-6))
        + pw * (t_raw[:, 1] / max(ps, 1e-6))
        + b0w * (t_raw[:, 2] / max(b0s, 1e-6))
        + b1w * (t_raw[:, 3] / max(b1s, 1e-6))
    )
    return np.clip(topo, 0.0, 2.0)


def _compose_score(
    c_raw: np.ndarray,
    l_raw: np.ndarray,
    topo_scaled: np.ndarray,
    weights: tuple[float, float, float],
    cl_scales: tuple[float, float],
) -> np.ndarray:
    c = c_raw / max(cl_scales[0], 1e-6)
    l = l_raw / max(cl_scales[1], 1e-6)
    return np.clip(weights[0] * c + weights[1] * l + weights[2] * topo_scaled, 0.0, 1.5)


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.clip(scores, 0.0, 1.5))
    best_t, best_j = float(candidates[0]), -1.0
    for t in candidates:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        j = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Extract per-session features from benchmark JSON
# ---------------------------------------------------------------------------

def _extract_session_features(
    sessions: list[dict[str, Any]],
    agg: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (c_raw, l_raw, t_raw[4], labels) from session list.

    agg: 'mean' or 'max' — how to aggregate step-level values per session.
    """
    c_list, l_list, t_list, y_list = [], [], [], []
    fn = np.mean if agg == "mean" else np.max

    for s in sessions:
        steps = s.get("steps", [])
        if not steps:
            continue

        cos_vals = [st["cosine_continuity"] for st in steps if st.get("cosine_continuity") is not None]
        lip_vals = [st["lipschitz_proxy"] for st in steps if st.get("lipschitz_proxy") is not None]
        diam_vals = [st["topology_diameter"] for st in steps if st.get("topology_diameter") is not None]
        p1_vals = [st["topology_persistence_l1"] for st in steps if st.get("topology_persistence_l1") is not None]
        b0_vals = [st["topology_beta0"] for st in steps if st.get("topology_beta0") is not None]
        b1_vals = [st["topology_beta1"] for st in steps if st.get("topology_beta1") is not None]

        # Continuity raw: how much cosine drops below floor (floor hardcoded to 0.85 to match config).
        cosine_floor = 0.85
        c_raw = float(fn([max(0.0, cosine_floor - float(v)) / max(1.0 - cosine_floor, 1e-6) for v in cos_vals])) if cos_vals else 0.0
        lipschitz_ceiling = 0.45
        l_raw = float(fn([max(0.0, float(v) - lipschitz_ceiling) / max(lipschitz_ceiling, 1e-6) for v in lip_vals])) if lip_vals else 0.0

        t_raw = np.array([
            float(fn([max(0.0, float(v)) for v in diam_vals])) if diam_vals else 0.0,
            float(fn([max(0.0, float(v)) for v in p1_vals])) if p1_vals else 0.0,
            float(fn([max(0.0, float(v)) for v in b0_vals])) if b0_vals else 0.0,
            float(fn([max(0.0, float(v)) for v in b1_vals])) if b1_vals else 0.0,
        ], dtype=np.float64)

        c_list.append(c_raw)
        l_list.append(l_raw)
        t_list.append(t_raw)
        y_list.append(int(s["unsafe_label"]))

    return (
        np.array(c_list, dtype=np.float64),
        np.array(l_list, dtype=np.float64),
        np.stack(t_list, axis=0) if t_list else np.zeros((0, 4)),
        np.array(y_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def calibrate(
    sessions: list[dict[str, Any]],
    agg: str = "mean",
    weight_step: float = 0.1,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    c_raw, l_raw, t_raw, y = _extract_session_features(sessions, agg=agg)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return {"error": "Need at least 2 classes and 1 session per class."}

    # Scale grids derived from empirical quantiles.
    c_grid  = _quantile_scales(c_raw)
    l_grid  = _quantile_scales(l_raw)
    dg      = _quantile_scales(t_raw[:, 0])
    pg      = _quantile_scales(t_raw[:, 1])
    b0g     = _quantile_scales(t_raw[:, 2])
    b1g     = _quantile_scales(t_raw[:, 3])

    k = min(n_folds, min(int(np.sum(y == 0)), int(np.sum(y == 1))))
    k = max(k, 2)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(skf.split(c_raw, y))

    best_auc = -1.0
    best_pr  = -1.0
    best_std = -1.0
    best_weights = (0.0, 0.0, 1.0)
    best_scales  = (1.0, 1.0, 2.0, 5.0, 8.0, 3.0)
    best_scores  = np.zeros(n)

    total = (len(c_grid) * len(l_grid) * len(dg) * len(pg) * len(b0g) * len(b1g)
             * len(_weight_candidates(weight_step)))
    print(f"Grid search: {total:,} candidates, {k}-fold CV, n={n} sessions...")

    for cs in c_grid:
        for ls in l_grid:
            for ds in dg:
                for ps in pg:
                    for b0s in b0g:
                        for b1s in b1g:
                            topo = _apply_topo_scales(t_raw, (ds, ps, b0s, b1s))
                            for cw, lw, tw in _weight_candidates(weight_step):
                                scores = _compose_score(c_raw, l_raw, topo, (cw, lw, tw), (cs, ls))
                                fold_aucs = []
                                for _, val_idx in folds:
                                    yv = y[val_idx]
                                    if len(np.unique(yv)) < 2:
                                        continue
                                    fold_aucs.append(float(roc_auc_score(yv, scores[val_idx])))
                                if not fold_aucs:
                                    continue
                                auc = float(np.mean(fold_aucs))
                                pr = float(average_precision_score(y, scores))
                                std = float(np.std(scores))
                                if (auc > best_auc + 1e-8
                                        or (abs(auc - best_auc) <= 1e-8 and pr > best_pr + 1e-8)
                                        or (abs(auc - best_auc) <= 1e-8 and abs(pr - best_pr) <= 1e-8 and std > best_std + 1e-8)):
                                    best_auc = auc
                                    best_pr  = pr
                                    best_std = std
                                    best_weights = (cw, lw, tw)
                                    best_scales  = (cs, ls, ds, ps, b0s, b1s)
                                    best_scores  = scores.copy()

    train_auc = float(roc_auc_score(y, best_scores))
    cv_aucs = [float(roc_auc_score(y[vi], best_scores[vi]))
               for _, vi in folds if len(np.unique(y[vi])) >= 2]
    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else train_auc
    threshold = _best_threshold(best_scores, y)

    fpr, tpr, roc_ths = roc_curve(y, best_scores)

    print(f"Best CV AUROC: {best_auc:.4f}  Train AUROC: {train_auc:.4f}  Overfit gap: {train_auc - cv_mean:.4f}")
    print(f"Optimal weights: continuity={best_weights[0]:.2f}  lipschitz={best_weights[1]:.2f}  topology={best_weights[2]:.2f}")
    print(f"Optimal scales:  c={best_scales[0]:.3f}  l={best_scales[1]:.3f}  diam={best_scales[2]:.3f}  p1={best_scales[3]:.3f}  b0={best_scales[4]:.3f}  b1={best_scales[5]:.3f}")
    print(f"Best threshold (Youden J): {threshold:.4f}")

    return {
        "n_sessions": int(n),
        "aggregation": agg,
        "cv_folds": k,
        "cv_auc_mean": cv_mean,
        "train_auc": train_auc,
        "overfit_gap": train_auc - cv_mean,
        "pr_auc": best_pr,
        "best_threshold": float(threshold),
        "optimal_weights": {
            "continuity_weight": best_weights[0],
            "lipschitz_weight": best_weights[1],
            "topology_weight": best_weights[2],
        },
        "optimal_scales": {
            "continuity_scale": best_scales[0],
            "lipschitz_scale": best_scales[1],
            "topology_diameter_scale": best_scales[2],
            "persistence_l1_scale": best_scales[3],
            "beta0_scale": best_scales[4],
            "beta1_scale": best_scales[5],
        },
        # Ready to paste into DriftGuardConfig or --flags.
        "recommended_config_update": {
            "risk_threshold": float(threshold),
            "continuity_weight": best_weights[0],
            "lipschitz_weight": best_weights[1],
            "topology_weight": best_weights[2],
            "continuity_scale": best_scales[0],
            "lipschitz_scale": best_scales[1],
            "topology_diameter_scale": best_scales[2],
            "persistence_l1_scale": best_scales[3],
            "beta0_scale": best_scales[4],
            "beta1_scale": best_scales[5],
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_ths.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate DriftGuard weights/scales from pre-computed benchmark JSON."
    )
    parser.add_argument("--report-json", type=Path, required=True,
                        help="Path to benchmark output JSON (from run_driftguard_benchmark.py).")
    parser.add_argument("--out-json", type=Path,
                        default=Path("experiments/outputs/calibration_from_json.json"))
    parser.add_argument("--agg", choices=["mean", "max"], default="mean",
                        help="Session-level aggregation (mean recommended).")
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Which seed run to calibrate on (0 = first).")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Pool sessions across all seeds before calibrating.")
    parser.add_argument("--weight-step", type=float, default=0.1,
                        help="Grid step for weight search (smaller = finer but slower).")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    payload = json.loads(args.report_json.read_text())
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("No runs found in benchmark JSON.")

    if args.all_seeds:
        sessions: list[dict] = []
        for run in runs:
            sessions.extend(run.get("sessions", {}).get("no_steer", []))
        print(f"Pooling {len(sessions)} sessions across {len(runs)} seeds.")
    else:
        idx = min(args.seed_idx, len(runs) - 1)
        sessions = runs[idx].get("sessions", {}).get("no_steer", [])
        print(f"Calibrating on seed index {idx} ({len(sessions)} sessions).")

    result = calibrate(
        sessions=sessions,
        agg=args.agg,
        weight_step=args.weight_step,
        n_folds=args.n_folds,
        seed=42,
    )
    result["source_json"] = str(args.report_json)
    result["model"] = payload.get("model", "unknown")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"\nWrote calibration results to {args.out_json}")


if __name__ == "__main__":
    main()
