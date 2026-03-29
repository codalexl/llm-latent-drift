#!/usr/bin/env python3
"""Generate DriftGuard benchmark figures for paper-ready reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_dynamics.hub import load_activations
from latent_dynamics.tda_metrics import pca_reduce
from latent_dynamics.utils import save_persistence_diagram


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_cosine_heatmap(run: dict, out_path: Path) -> None:
    sessions = run["sessions"]["steer"]
    max_len = max((len(s["steps"]) for s in sessions), default=1)
    mat = np.full((len(sessions), max_len), np.nan, dtype=np.float32)
    for i, sess in enumerate(sessions):
        for t, step in enumerate(sess["steps"]):
            cos = step.get("cosine_continuity")
            if cos is not None:
                mat[i, t] = 1.0 - float(cos)
    plt.figure(figsize=(10, 4))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="1 - cosine_continuity")
    plt.xlabel("Token index")
    plt.ylabel("Session")
    plt.title("Cosine drift heatmap (steered sessions)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_lead_time_hist(run: dict, out_path: Path) -> None:
    sessions = run["sessions"]["steer"]
    vals = [
        float(s["first_alarm_lead_time"])
        for s in sessions
        if s["unsafe_label"] == 1 and s["first_alarm_lead_time"] is not None
    ]
    if not vals:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=min(10, len(vals)), color="tab:red", alpha=0.8)
    plt.xlabel("Lead time (tokens)")
    plt.ylabel("Count")
    plt.title("Early-warning lead time on unsafe prompts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_latency_bar(run: dict, out_path: Path) -> None:
    no_steer = run["summary"]["no_steer"]["mean_step_latency_ms"]
    steer = run["summary"]["steer"]["mean_step_latency_ms"]
    plt.figure(figsize=(5, 4))
    plt.bar(["no_steer", "steer"], [no_steer, steer], color=["tab:blue", "tab:orange"])
    plt.ylabel("Mean per-token latency (ms)")
    plt.title("Runtime overhead")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_ablation_curves(run: dict, out_path: Path) -> None:
    ablations = run["summary"].get("ablations", {})
    if not ablations:
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    drew_roc = False
    drew_pr = False
    for profile, payload in ablations.items():
        roc = payload.get("roc_curve")
        auroc = payload.get("detection", {}).get("auroc")
        label = f"{profile} (AUROC={auroc:.3f})" if auroc is not None else profile
        if roc:
            fpr = np.asarray(roc.get("fpr", []), dtype=np.float32)
            tpr = np.asarray(roc.get("tpr", []), dtype=np.float32)
            if fpr.size > 0 and tpr.size > 0:
                ax_roc.plot(fpr, tpr, linewidth=2, label=label)
                drew_roc = True
        pr = payload.get("pr_curve")
        if pr:
            recall = np.asarray(pr.get("recall", []), dtype=np.float32)
            precision = np.asarray(pr.get("precision", []), dtype=np.float32)
            if recall.size > 0 and precision.size > 0:
                ax_pr.plot(recall, precision, linewidth=2, label=label)
                drew_pr = True
    if not drew_roc and not drew_pr:
        plt.close(fig)
        return
    if drew_roc:
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Ablation ROC")
    if drew_roc:
        ax_roc.legend(fontsize=8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Ablation PR")
    if drew_pr:
        ax_pr.legend(fontsize=8)
    fig.suptitle("Risk score ablations: ROC and PR curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_pca_paths(activations: Path, out_path: Path) -> None:
    trajectories, _texts, labels, _tokens, _gen, _cfg = load_activations(activations)
    if labels is None:
        return
    safe_idx = int(np.where(labels == 0)[0][0]) if np.any(labels == 0) else 0
    unsafe_idx = int(np.where(labels == 1)[0][0]) if np.any(labels == 1) else 0
    safe = trajectories[safe_idx]
    unsafe = trajectories[unsafe_idx]
    joint = np.concatenate([safe, unsafe], axis=0)
    joint_z = pca_reduce(joint, n_components=2)
    safe_z = joint_z[: safe.shape[0]]
    unsafe_z = joint_z[safe.shape[0] :]
    plt.figure(figsize=(6, 5))
    plt.plot(safe_z[:, 0], safe_z[:, 1], color="tab:green", label="safe", linewidth=2)
    plt.plot(unsafe_z[:, 0], unsafe_z[:, 1], color="tab:red", label="unsafe", linewidth=2)
    plt.scatter(safe_z[0, 0], safe_z[0, 1], color="tab:green", s=20)
    plt.scatter(unsafe_z[0, 0], unsafe_z[0, 1], color="tab:red", s=20)
    plt.title("PCA latent trajectories (example safe vs unsafe)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_persistence_diagram(activations: Path, out_path: Path) -> None:
    trajectories, _texts, labels, _tokens, _gen, _cfg = load_activations(activations)
    if labels is None or len(trajectories) == 0:
        return
    idx = int(np.where(labels == 1)[0][0]) if np.any(labels == 1) else 0
    points = pca_reduce(trajectories[idx], n_components=8)
    try:
        from ripser import ripser
    except Exception:
        return
    dgms = ripser(points, maxdim=1).get("dgms", [])
    if len(dgms) < 2:
        return
    d1 = dgms[1]
    if d1.size == 0:
        return
    save_persistence_diagram(d1, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from DriftGuard benchmark report.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/outputs/figures"))
    parser.add_argument(
        "--activations",
        type=Path,
        default=None,
        help="Optional activations leaf for PCA path and persistence figures.",
    )
    args = parser.parse_args()

    report = _load_report(args.report_json)
    _ensure_dir(args.out_dir)
    run = report["runs"][0]
    _plot_cosine_heatmap(run, args.out_dir / "driftguard_cosine_heatmap.png")
    _plot_lead_time_hist(run, args.out_dir / "driftguard_lead_time_hist.png")
    _plot_latency_bar(run, args.out_dir / "driftguard_latency_bar.png")
    _plot_ablation_curves(run, args.out_dir / "driftguard_ablation_roc.png")
    if args.activations is not None:
        _plot_pca_paths(args.activations, args.out_dir / "driftguard_pca_paths.png")
        _plot_persistence_diagram(
            args.activations,
            args.out_dir / "driftguard_persistence_diagram.png",
        )
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
