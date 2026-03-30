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


def _baseline_rows(aggregate_summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Baseline | AUROC | PR-AUC | F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, key in (
        ("DriftGuard (fused)", "driftguard_fused"),
        ("Continuity-only", "continuity_only"),
        ("Topology-only", "topology_only"),
        ("Logit-lens", "logit_lens"),
        ("No-monitor", "no_monitor"),
    ):
        det = {"auroc": aggregate_summary["auroc"].get(key)}
        lines.append(
            f"| {name} | {_fmt_stat(det.get('auroc'))} | N/A | N/A |"
        )
    return lines


def _render_results_section(payload: dict[str, Any]) -> str:
    agg = payload["aggregate"]

    # Lead-time stats.
    lt = agg.get("lead_time_unsafe", {})
    lt_mean = lt.get("mean")
    lt_median = lt.get("median")
    lt_std = lt.get("std")

    # Latency breakdown.
    lat = agg.get("latency_ms", {})
    lat_ns = lat.get("no_steer_mean")
    lat_st = lat.get("steer_mean")

    # Intervention delta.
    idelta = agg.get("intervention_delta", {})

    lines = [
        "# [SECTION 6: RESULTS]",
        "",
        "### 6.1 Detection quality (multi-seed aggregation)",
        (
            f"On the **WildChat-1M + WildJailbreak** hybrid suite "
            f"(benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak "
            f"`adversarial_harmful` prompts in synthetic two-turn chats; "
            f"N\\approx200+200, {payload.get('num_seeds', 'N/A')} seeds) using model `{payload['model']}`, "
            f"the fused detector reaches AUROC {_fmt_stat(agg['auroc'].get('driftguard_fused'))}."
        ),
        "",
        "### 6.2 Early warning and runtime profile",
        f"- **Mean unsafe-session lead time:** {_fmt_stat({'mean': lt_mean, 'std': lt_std})} tokens"
        + (f" (median {lt_median:.1f})" if lt_median is not None else "")
        + ".",
        f"- **Mean token latency (no-steer):** {_fmt_stat({'mean': lat_ns})} ms/token.",
        f"- **Mean token latency (steer):** {_fmt_stat({'mean': lat_st})} ms/token.",
        "",
        "### 6.3 Causal steering efficacy",
    ]
    if idelta:
        ns_u = idelta.get("no_steer_unsafe_alarm_rate")
        st_u = idelta.get("steer_unsafe_alarm_rate")
        d_u = idelta.get("delta_unsafe_alarm_rate")
        ns_b = idelta.get("no_steer_benign_alarm_rate")
        st_b = idelta.get("steer_benign_alarm_rate")
        d_b = idelta.get("delta_benign_alarm_rate")
        lines.extend([
            f"- Unsafe alarm rate: no-steer {_fmt_stat({'mean': ns_u})} → steer {_fmt_stat({'mean': st_u})} (Δ = {_fmt_stat({'mean': d_u})}).",
            f"- Benign alarm rate: no-steer {_fmt_stat({'mean': ns_b})} → steer {_fmt_stat({'mean': st_b})} (Δ = {_fmt_stat({'mean': d_b})}).",
        ])
    else:
        lines.append("Intervention delta data not available.")
    lines.extend([
        "",
        "### 6.4 Baseline table",
        *_baseline_rows(agg),
        "",
        "These values are generated directly from benchmark JSON and exported baselines.",
    ])
    return "\n".join(lines) + "\n"


def _replace_section6(preprint_text: str, new_section: str) -> str:
    pattern = re.compile(
        r"(?s)^#+ *\[?SECTION 6[:\]].*?(?=^#+ *\[?SECTION 7[:\]])",
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
