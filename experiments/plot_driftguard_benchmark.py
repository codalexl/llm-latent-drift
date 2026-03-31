#!/usr/bin/env python3
"""Generate DriftGuard benchmark figures for paper-ready reporting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _sessions(run: dict, condition: str = "no_steer") -> list[dict]:
    """Return sessions for a condition, falling back to no_steer if steer absent."""
    sess = run.get("sessions", {})
    if condition in sess and sess[condition]:
        return sess[condition]
    return sess.get("no_steer", [])


def _plot_cosine_heatmap(run: dict, out_path: Path) -> None:
    sessions = _sessions(run, "no_steer")
    if not sessions:
        return
    max_len = max((len(s["steps"]) for s in sessions), default=1)
    mat = np.full((len(sessions), max_len), np.nan, dtype=np.float32)
    for i, sess in enumerate(sessions):
        for t, step in enumerate(sess["steps"]):
            cos = step.get("cosine_continuity")
            if cos is not None:
                mat[i, t] = 1.0 - float(cos)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=0.5)
    fig.colorbar(im, ax=ax, label="1 − cosine_continuity")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Session")
    ax.set_title("Cosine drift heatmap (detection-only sessions)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_lead_time_hist(run: dict, out_path: Path) -> None:
    sessions = _sessions(run, "no_steer")
    vals = [
        float(s["first_alarm_lead_time"])
        for s in sessions
        if s.get("unsafe_label") == 1 and s.get("first_alarm_lead_time") is not None
    ]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=min(20, len(vals)), color="tab:red", alpha=0.8, edgecolor="white")
    ax.axvline(float(np.median(vals)), color="black", linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(vals):.0f}")
    ax.set_xlabel("Lead time (tokens before end of generation)")
    ax.set_ylabel("Count")
    ax.set_title("Early-warning lead time distribution (unsafe sessions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_latency_bar(run: dict, out_path: Path) -> None:
    summary = run.get("summary", {})
    no_steer = summary.get("no_steer", {}).get("mean_step_latency_ms")
    steer_summary = summary.get("steer") or {}
    steer = steer_summary.get("mean_step_latency_ms") if isinstance(steer_summary, dict) else None

    labels, values, colors = [], [], []
    if no_steer is not None:
        labels.append("Detection-only")
        values.append(float(no_steer))
        colors.append("tab:blue")
    if steer is not None:
        labels.append("+ Steering")
        values.append(float(steer))
        colors.append("tab:orange")
    if not values:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean per-token latency (ms)")
    ax.set_title("Runtime overhead")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_ablation_bar(aggregate: dict, out_path: Path) -> None:
    """AUROC bar chart with error bars from multi-seed aggregate."""
    auroc = aggregate.get("auroc", {})
    if not auroc:
        return

    rows = [
        ("DriftGuard\n(fused)",  "mean_driftguard_fused"),
        ("Topology\nonly",       "mean_topology_only"),
        ("Continuity\nonly",     "mean_continuity_only"),
        ("Dynamics\nonly",       "mean_dynamics_only"),
        ("Alarm\nfraction",      "alarm_frac_auroc"),
        ("No\nmonitor",          "no_monitor"),
    ]
    names, means, errs = [], [], []
    for label, key in rows:
        stat = auroc.get(key)
        if not isinstance(stat, dict) or stat.get("mean") is None:
            continue
        mean = float(stat["mean"])
        std = stat.get("std") or 0.0
        n = stat.get("n") or 1
        ci_half = 1.96 * float(std) / math.sqrt(max(int(n), 1))
        names.append(label)
        means.append(mean)
        errs.append(ci_half)

    if not names:
        return

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:gray"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=errs, capsize=5,
                  color=colors[:len(names)], alpha=0.85, error_kw={"linewidth": 1.5})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    for bar, val, err in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUROC (mean-session aggregation)")
    ax.set_title("Detection ablation — AUROC by risk component (mean ± 95% CI)")
    ax.set_ylim(0, min(max(means) + max(errs) + 0.15, 1.05))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_risk_score_distributions(run: dict, out_path: Path) -> None:
    """Violin/box plot of mean session risk by label."""
    sessions = _sessions(run, "no_steer")
    benign, unsafe = [], []
    for s in sessions:
        steps = s.get("steps", [])
        risks = [st["risk_score"] for st in steps if st.get("risk_score") is not None]
        if not risks:
            continue
        mean_r = float(np.mean(risks))
        if s.get("unsafe_label") == 1:
            unsafe.append(mean_r)
        else:
            benign.append(mean_r)
    if not benign or not unsafe:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    parts = ax.violinplot([benign, unsafe], positions=[0, 1], showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    parts["bodies"][0].set_facecolor("tab:blue")
    parts["bodies"][1].set_facecolor("tab:red")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benign", "Unsafe"])
    ax.set_ylabel("Mean session risk score")
    ax.set_title("Risk score distribution by label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_per_seed_auroc(payload: dict, out_path: Path) -> None:
    """Line plot of mean-fused and topo-only AUROC across seeds."""
    runs = payload.get("runs", [])
    if len(runs) < 2:
        return
    seeds = [r["seed"] for r in runs]
    fused = [r["auroc"].get("mean_driftguard_fused") for r in runs]
    topo  = [r["auroc"].get("mean_topology_only") for r in runs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(seeds, fused, marker="o", label="DriftGuard (fused)", color="tab:blue")
    ax.plot(seeds, topo,  marker="s", label="Topology-only",      color="tab:green")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Seed")
    ax.set_ylabel("AUROC (mean-session)")
    ax.set_title("Per-seed AUROC stability")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from DriftGuard benchmark report.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/outputs/figures"))
    args = parser.parse_args()

    report = _load_report(args.report_json)
    _ensure_dir(args.out_dir)
    run = report["runs"][0]
    agg = report.get("aggregate", {})

    _plot_cosine_heatmap(run,    args.out_dir / "driftguard_cosine_heatmap.png")
    _plot_lead_time_hist(run,    args.out_dir / "driftguard_lead_time_hist.png")
    _plot_latency_bar(run,       args.out_dir / "driftguard_latency_bar.png")
    _plot_ablation_bar(agg,      args.out_dir / "driftguard_ablation_auroc.png")
    _plot_risk_score_distributions(run, args.out_dir / "driftguard_risk_distributions.png")
    _plot_per_seed_auroc(report, args.out_dir / "driftguard_per_seed_auroc.png")

    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
