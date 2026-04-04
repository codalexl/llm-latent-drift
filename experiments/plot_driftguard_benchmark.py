#!/usr/bin/env python3
"""Generate DriftGuard benchmark figures for paper-ready reporting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _discover_benchmark_jsons(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("driftguard_benchmark*.json"))
    return [p for p in paths if p.is_file() and "pilot" not in p.name.lower()]


def _slug_from_path(path: Path) -> str:
    stem = path.stem.replace("driftguard_benchmark_", "")
    return stem or "benchmark"


def _session_mean_scores(session: dict) -> dict[str, float]:
    """Match `run_driftguard_benchmark._session_summary_row` mean aggregations."""
    steps = session.get("steps") or []
    if not steps:
        return {"fused": 0.0, "topology": 0.0, "continuity": 0.0, "dynamics": 0.0}
    fused, topo, cont, dyn = [], [], [], []
    for st in steps:
        rs = st.get("risk_score")
        if rs is not None:
            fused.append(float(rs))
        tr = st.get("topology_risk")
        if tr is not None:
            topo.append(float(tr))
        cr = st.get("continuity_risk")
        if cr is not None:
            cont.append(float(cr))
        dr = st.get("dynamics_risk")
        dyn.append(float(dr or 0.0))
    return {
        "fused": float(np.mean(fused)) if fused else 0.0,
        "topology": float(np.mean(topo)) if topo else 0.0,
        "continuity": float(np.mean(cont)) if cont else 0.0,
        "dynamics": float(np.mean(dyn)) if dyn else 0.0,
    }


def _pooled_session_arrays(report: dict, condition: str = "no_steer") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels and mean-session scores (fused, topology) pooled over all seeds."""
    y_list, s_f, s_t = [], [], []
    for run in report.get("runs", []):
        sessions = _sessions(run, condition)
        for sess in sessions:
            lab = sess.get("unsafe_label")
            if lab is None:
                continue
            m = _session_mean_scores(sess)
            y_list.append(int(lab))
            s_f.append(m["fused"])
            s_t.append(m["topology"])
    if not y_list:
        return np.array([]), np.array([]), np.array([])
    return (
        np.asarray(y_list, dtype=np.int64),
        np.asarray(s_f, dtype=np.float64),
        np.asarray(s_t, dtype=np.float64),
    )


def _plot_roc_curves(
    reports: list[tuple[str, dict]],
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(reports), 1)))
    for i, (label, report) in enumerate(reports):
        y, fused, _topo = _pooled_session_arrays(report)
        if y.size == 0 or len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, fused)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cmap[i % len(cmap)], lw=2, label=f"{label} (AUROC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_roc_fused_vs_topology(report: dict, out_path: Path, title: str) -> None:
    y, fused, topo = _pooled_session_arrays(report)
    fig, ax = plt.subplots(figsize=(6, 5))
    if y.size and len(np.unique(y)) >= 2:
        for scores, name, c in (
            (fused, "Fused (mean session)", "tab:blue"),
            (topo, "Topology-only (mean session)", "tab:green"),
        ):
            fpr, tpr, _ = roc_curve(y, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUROC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_multi_model_ablation(
    reports: list[tuple[str, dict]],
    out_path: Path,
) -> None:
    """Grouped bars: models × ablation methods (mean-session AUROC)."""
    method_keys = [
        ("Fused", "mean_driftguard_fused"),
        ("Topology", "mean_topology_only"),
        ("Continuity", "mean_continuity_only"),
        ("Dynamics", "mean_dynamics_only"),
    ]
    models = [label for label, _ in reports]
    n_m, n_k = len(models), len(method_keys)
    if n_m == 0:
        return
    means = np.zeros((n_m, n_k))
    errs = np.zeros((n_m, n_k))
    for mi, (_lbl, rep) in enumerate(reports):
        auroc = rep.get("aggregate", {}).get("auroc", {})
        for ki, (_name, key) in enumerate(method_keys):
            stat = auroc.get(key)
            if not isinstance(stat, dict) or stat.get("mean") is None:
                means[mi, ki] = float("nan")
                errs[mi, ki] = 0.0
                continue
            m = float(stat["mean"])
            std = float(stat.get("std") or 0.0)
            n = int(stat.get("n") or 1)
            means[mi, ki] = m
            errs[mi, ki] = 1.96 * std / math.sqrt(max(n, 1))

    x = np.arange(n_m)
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7, n_m * 2.2), 4.5))
    for ki, (name, _key) in enumerate(method_keys):
        offset = (ki - (n_k - 1) / 2) * width
        yerr = errs[:, ki]
        ax.bar(
            x + offset,
            means[:, ki],
            width,
            yerr=yerr,
            capsize=3,
            label=name,
            alpha=0.88,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("AUROC (mean-session)")
    ax.set_title("Cross-model ablation (mean ± 95% CI over seeds)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_persistence_example(run: dict, out_path: Path, max_sessions: int = 12) -> None:
    """Line plot of mean topology_persistence_l1 over tokens (first N unsafe+benign sessions)."""
    sessions = _sessions(run, "no_steer")[:max_sessions]
    if not sessions:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, sess in enumerate(sessions):
        ys = [
            float(st["topology_persistence_l1"])
            for st in sess.get("steps", [])
            if st.get("topology_persistence_l1") is not None
        ]
        if len(ys) < 2:
            continue
        label = "unsafe" if sess.get("unsafe_label") == 1 else "benign"
        color = "tab:red" if label == "unsafe" else "tab:blue"
        ax.plot(
            range(len(ys)),
            ys,
            color=color,
            alpha=0.35 + 0.4 * (i / max(len(sessions), 1)),
            linewidth=1,
        )
    ax.set_xlabel("Token index (steps with persistence)")
    ax.set_ylabel("topology_persistence_l1")
    ax.set_title("Persistence diagram summary (example sessions; opacity varies)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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


def _write_single_report_figures(report: Path | dict, out_dir: Path, slug: str) -> list[str]:
    """Generate standard figure set; `report` is path or loaded dict."""
    data = _load_report(report) if isinstance(report, Path) else report
    _ensure_dir(out_dir)
    run = data["runs"][0]
    agg = data.get("aggregate", {})
    written: list[str] = []
    pairs = [
        (_plot_cosine_heatmap, run, out_dir / f"driftguard_cosine_heatmap_{slug}.png"),
        (_plot_lead_time_hist, run, out_dir / f"driftguard_lead_time_hist_{slug}.png"),
        (_plot_latency_bar, run, out_dir / f"driftguard_latency_bar_{slug}.png"),
        (_plot_ablation_bar, agg, out_dir / f"driftguard_ablation_auroc_{slug}.png"),
        (_plot_risk_score_distributions, run, out_dir / f"driftguard_risk_distributions_{slug}.png"),
        (_plot_per_seed_auroc, data, out_dir / f"driftguard_per_seed_auroc_{slug}.png"),
    ]
    for fn, arg, dest in pairs:
        fn(arg, dest)
        if dest.exists():
            written.append(dest.name)
    roc_path = out_dir / f"driftguard_roc_mean_session_{slug}.png"
    _plot_roc_fused_vs_topology(
        data,
        roc_path,
        title=f"ROC (mean-session scores) — {slug}",
    )
    if roc_path.exists():
        written.append(roc_path.name)
    pers_path = out_dir / f"driftguard_persistence_traces_{slug}.png"
    _plot_persistence_example(run, pers_path)
    if pers_path.exists():
        written.append(pers_path.name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from DriftGuard benchmark report.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--report-json", type=Path, help="Single benchmark JSON.")
    src.add_argument(
        "--input-dir",
        type=Path,
        help="Directory with driftguard_benchmark*.json — per-model + combined figures.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/outputs/figures"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Alias for --out-dir (paper workflow docs).",
    )
    args = parser.parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.out_dir

    all_written: list[str] = []

    if args.input_dir is not None:
        paths = _discover_benchmark_jsons(args.input_dir)
        if not paths:
            raise SystemExit(f"No driftguard_benchmark*.json under {args.input_dir}")
        reports: list[tuple[str, dict]] = []
        for p in paths:
            slug = _slug_from_path(p)
            rep = _load_report(p)
            label = str(rep.get("model", slug))
            reports.append((label, rep))
            w = _write_single_report_figures(rep, out_dir, slug)
            all_written.extend(w)

        combined_roc = out_dir / "driftguard_roc_combined_fused.png"
        _plot_roc_curves(
            reports,
            combined_roc,
            title="ROC — fused mean-session score (all models)",
        )
        if combined_roc.exists():
            all_written.append(combined_roc.name)

        multi_ab = out_dir / "driftguard_ablation_cross_model.png"
        _plot_multi_model_ablation(reports, multi_ab)
        if multi_ab.exists():
            all_written.append(multi_ab.name)
    else:
        slug = _slug_from_path(args.report_json)
        all_written = _write_single_report_figures(args.report_json, out_dir, slug)

    print(f"Wrote figures to {out_dir}")
    for name in sorted(set(all_written)):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
