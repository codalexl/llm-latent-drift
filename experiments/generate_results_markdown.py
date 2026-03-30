#!/usr/bin/env python3
"""Generate publication-ready Results markdown from benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def _fmt_stat(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not isinstance(stat, dict):
        return "N/A"
    mean = stat.get("mean")
    std = stat.get("std")
    lo = stat.get("ci95_low")
    hi = stat.get("ci95_high")
    if mean is None:
        return "N/A"
    if std is None or lo is None or hi is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} +- {float(std):.{digits}f} (95% CI [{float(lo):.{digits}f}, {float(hi):.{digits}f}])"


def _best_condition(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("aggregated_runs", [])
    if not items:
        raise ValueError("Expected non-empty aggregated_runs in benchmark report.")

    def score(item: dict[str, Any]) -> float:
        try:
            stat = item["aggregate"]["summary"]["steer"]["detection"]["auroc"]
            val = stat.get("mean")
            return float(val) if val is not None else -1.0
        except Exception:
            return -1.0

    return max(items, key=score)


def _baseline_rows(aggregate_summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Baseline | AUROC | PR-AUC | F1 |",
        "|---|---:|---:|---:|",
    ]
    ablations = aggregate_summary.get("ablations", {})
    for name in sorted(ablations.keys()):
        det = ablations[name].get("detection", {})
        lines.append(
            f"| {name} | {_fmt_stat(det.get('auroc'))} | {_fmt_stat(det.get('pr_auc'))} | {_fmt_stat(det.get('f1'))} |"
        )
    return lines


def _render_results_section(payload: dict[str, Any]) -> str:
    best = _best_condition(payload)
    quant = best.get("quantization", "unknown")
    agg = best["aggregate"]["summary"]
    steer = agg["steer"]
    no_steer = agg["no_steer"]
    delta = agg.get("intervention_delta", {})
    lead = steer.get("lead_time_distribution", {})

    lead_line = "N/A"
    if lead.get("median") is not None:
        lead_line = (
            f"median={float(lead['median']):.2f}, "
            f"IQR=[{float(lead['q25']):.2f}, {float(lead['q75']):.2f}] "
            f"(n={int(lead['n'])})"
        )

    lines = [
        "# [SECTION 6: RESULTS]",
        "",
        "### 6.1 Detection quality (multi-seed aggregation)",
        (
            f"We evaluate DriftGuard on `{payload['preset']}` with model `{payload['model']}` "
            f"under `{quant}` settings using `{payload.get('num_seeds', 'N/A')}` seeds."
        ),
        (
            f"- **Steered detector AUROC:** {_fmt_stat(steer['detection'].get('auroc'))}; "
            f"**PR-AUC:** {_fmt_stat(steer['detection'].get('pr_auc'))}."
        ),
        (
            f"- **No-steer detector AUROC:** {_fmt_stat(no_steer['detection'].get('auroc'))}; "
            f"**PR-AUC:** {_fmt_stat(no_steer['detection'].get('pr_auc'))}."
        ),
        (
            f"- **Precision / Recall / F1 (steer):** "
            f"{_fmt_stat(steer['detection'].get('precision'))} / "
            f"{_fmt_stat(steer['detection'].get('recall'))} / "
            f"{_fmt_stat(steer['detection'].get('f1'))}."
        ),
        "",
        "### 6.2 Early warning and intervention behavior",
        f"- **Unsafe-session lead time (steer):** {lead_line}.",
        (
            f"- **Delta benign alarm rate (steer - no_steer):** "
            f"{_fmt_stat(delta.get('delta_benign_alarm_rate'))}."
        ),
        (
            f"- **Delta lead-time mean (steer - no_steer):** "
            f"{_fmt_stat(delta.get('delta_lead_time_mean'))}."
        ),
        "",
        "### 6.3 Runtime cost and feasibility",
        (
            f"- **Mean token latency (no-steer):** {_fmt_stat(no_steer.get('mean_step_latency_ms'))}; "
            f"**(steer):** {_fmt_stat(steer.get('mean_step_latency_ms'))}."
        ),
        (
            f"- **Mean TDA executed steps (steer):** "
            f"{_fmt_stat(steer.get('mean_tda_executed_steps'))}."
        ),
        "",
        "### 6.4 Baseline and ablation table (aggregated)",
        *_baseline_rows(agg),
        "",
        (
            "These results are computed directly from machine-readable benchmark outputs "
            "with seed aggregation and bootstrap confidence intervals."
        ),
    ]
    return "\n".join(lines) + "\n"


def _replace_section6(preprint_text: str, new_section: str) -> str:
    pattern = re.compile(
        r"(?s)^# \[SECTION 6:.*?(?=^# \[SECTION 7:)",
        flags=re.MULTILINE,
    )
    if not pattern.search(preprint_text):
        raise ValueError("Could not locate Section 6/7 boundary in preprint.")
    return pattern.sub(new_section.rstrip() + "\n\n", preprint_text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and inject DriftGuard results markdown.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/outputs/driftguard_results_section.md"),
    )
    parser.add_argument(
        "--paper-path",
        type=Path,
        default=Path("paper/driftguard_preprint.md"),
    )
    parser.add_argument(
        "--replace-paper-section",
        action="store_true",
        help="Replace Section 6 in the preprint with generated results text.",
    )
    args = parser.parse_args()

    payload = json.loads(args.report_json.read_text())
    section_md = _render_results_section(payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(section_md)

    if args.replace_paper_section:
        preprint = args.paper_path.read_text()
        updated = _replace_section6(preprint, section_md)
        args.paper_path.write_text(updated)
    print(f"Wrote results markdown to {args.output_md}")
    if args.replace_paper_section:
        print(f"Updated Section 6 in {args.paper_path}")


if __name__ == "__main__":
    main()
