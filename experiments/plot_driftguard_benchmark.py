#!/usr/bin/env python3
"""Generate DriftGuard benchmark figures for paper-ready reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_dynamics.hub import load_activations
from latent_dynamics.tda_metrics import pca_reduce


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


def _plot_pca_paths(activations: Path, out_path: Path) -> None:
    trajectories, _texts, labels, _tokens, _gen, _cfg = load_activations(activations)
    if labels is None:
        return
    safe_idx = int(np.where(labels == 0)[0][0]) if np.any(labels == 0) else 0
    unsafe_idx = int(np.where(labels == 1)[0][0]) if np.any(labels == 1) else 0
    safe = trajectories[safe_idx]
    unsafe = trajectories[unsafe_idx]
    pca = PCA(n_components=2)
    pca.fit(np.concatenate([safe, unsafe], axis=0))
    safe_z = pca.transform(safe)
    unsafe_z = pca.transform(unsafe)
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
    d0, d1 = dgms[0], dgms[1]
    plt.figure(figsize=(6, 6))
    lim_max = 1.0
    if d0.size:
        lim_max = max(lim_max, float(np.nanmax(d0[np.isfinite(d0)])))
    if d1.size:
        lim_max = max(lim_max, float(np.nanmax(d1[np.isfinite(d1)])))
    if d0.size:
        plt.scatter(d0[:, 0], d0[:, 1], c="tab:blue", s=12, label="H0")
    if d1.size:
        plt.scatter(d1[:, 0], d1[:, 1], c="tab:orange", s=12, label="H1")
    plt.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
    plt.xlim(0, lim_max)
    plt.ylim(0, lim_max)
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence diagram (example unsafe trajectory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


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
    if args.activations is not None:
        _plot_pca_paths(args.activations, args.out_dir / "driftguard_pca_paths.png")
        _plot_persistence_diagram(
            args.activations,
            args.out_dir / "driftguard_persistence_diagram.png",
        )
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
