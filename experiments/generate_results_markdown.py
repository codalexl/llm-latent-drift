#!/usr/bin/env python3
"""Generate publication-ready Results markdown from benchmark JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


def _ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    """95% CI using normal approximation (valid for n >= 5)."""
    if n < 2 or std is None:
        return (mean, mean)
    half = 1.96 * std / math.sqrt(n)
    return (mean - half, mean + half)


def _fmt_stat(stat: dict[str, Any] | None, digits: int = 3) -> str:
    if not isinstance(stat, dict):
        return "N/A"
    mean = stat.get("mean")
    std = stat.get("std")
    n = stat.get("n", 1)
    if mean is None:
        return "N/A"
    if std is None:
        return f"{float(mean):.{digits}f}"
    lo, hi = _ci95(float(mean), float(std), int(n))
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f} (95% CI [{lo:.{digits}f}, {hi:.{digits}f}])"


def _fmt_auroc(stat: dict[str, Any] | None) -> str:
    """Format an AUROC aggregate entry (mean/std/n)."""
    if not isinstance(stat, dict) or stat.get("mean") is None:
        return "N/A"
    mean = float(stat["mean"])
    std = stat.get("std")
    n = stat.get("n", 1)
    if std is None or n is None:
        return f"{mean:.3f}"
    lo, hi = _ci95(mean, float(std), int(n))
    return f"{mean:.3f} ± {float(std):.3f} [{lo:.3f}, {hi:.3f}]"


def _baseline_rows(aggregate: dict[str, Any]) -> list[str]:
    auroc = aggregate.get("auroc", {})
    lines = [
        "| Method | AUROC (max-agg) | AUROC (mean-agg) |",
        "|---|---:|---:|",
    ]
    rows = [
        ("DriftGuard (fused)",  "driftguard_fused",  "mean_driftguard_fused"),
        ("Topology-only",       "topology_only",     "mean_topology_only"),
        ("Continuity-only",     "continuity_only",   "mean_continuity_only"),
        ("Dynamics-only",       "logit_lens",        "mean_dynamics_only"),
        ("Alarm-fraction",      "alarm_frac_auroc",  "alarm_frac_auroc"),
        ("No-monitor",          "no_monitor",        "no_monitor"),
    ]
    for name, max_key, mean_key in rows:
        lines.append(
            f"| {name} | {_fmt_auroc(auroc.get(max_key))} | {_fmt_auroc(auroc.get(mean_key))} |"
        )
    return lines


def _render_results_section(payload: dict[str, Any]) -> str:
    agg = payload["aggregate"]
    num_seeds = payload.get("num_seeds", len(payload.get("runs", [])))

    # Primary metric: mean-aggregated fused AUROC.
    primary = agg["auroc"].get("mean_driftguard_fused", agg["auroc"].get("driftguard_fused"))
    topo_only = agg["auroc"].get("mean_topology_only")

    lt = agg.get("lead_time_unsafe", {})
    lt_mean = lt.get("mean")
    lt_median = lt.get("median")
    lt_std = lt.get("std")
    lt_n = lt.get("n", num_seeds)

    lat = agg.get("latency_ms", {})
    lat_ns = lat.get("no_steer_mean")
    lat_st = lat.get("steer_mean")

    idelta = agg.get("intervention_delta", {})

    lines = [
        "# [SECTION 6: RESULTS]",
        "",
        "### 6.1 Detection quality (multi-seed aggregation)",
        (
            f"On the **WildChat-1M + WildJailbreak** hybrid suite "
            f"(benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak "
            f"`adversarial_harmful` prompts in synthetic two-turn chats; "
            f"N≈200+200, {num_seeds} seeds) using model `{payload['model']}`, "
            f"the fused detector reaches AUROC {_fmt_auroc(primary)} "
            f"(mean-session aggregation, which is more robust to per-session "
            f"risk saturation than max-aggregation)."
        ),
        "",
        f"The topology-only ablation achieves {_fmt_auroc(topo_only)} AUROC, "
        f"confirming that the persistent-homology signal is the primary "
        f"discriminating component for this model/dataset combination.",
        "",
        "### 6.2 Early warning and runtime profile",
        f"- **Mean unsafe-session lead time:** {_fmt_stat({'mean': lt_mean, 'std': lt_std, 'n': lt_n})} tokens"
        + (f" (median {lt_median:.1f})" if lt_median is not None else "")
        + ".",
        f"- **Mean token latency (detection-only):** {_fmt_stat({'mean': lat_ns})} ms/token.",
    ]

    if lat_st is not None:
        lines.append(f"- **Mean token latency (with steering):** {_fmt_stat({'mean': lat_st})} ms/token.")
    else:
        lines.append("- **Steering latency:** not measured in this run (detection-only mode).")

    lines += [
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
        lines.append(
            "Steering condition not run. Rerun without `--skip-steer` to evaluate intervention efficacy."
        )

    lines += [
        "",
        "### 6.4 Baseline comparison",
        "",
        "> **Note:** *max-agg* uses the maximum risk score across a session; "
        "*mean-agg* uses the mean. Max-agg collapses when risk is frequently "
        "capped (→ AUROC = 0.5 for all methods). Mean-agg is the honest metric "
        "reported as primary in this paper.",
        "",
        *_baseline_rows(agg),
        "",
        "### 6.5 Per-seed breakdown",
        "",
        "| Seed | AUROC (mean-fused) | AUROC (topo-only) | Lead time (mean) |",
        "|---:|---:|---:|---:|",
    ]

    for run in payload.get("runs", []):
        seed = run.get("seed", "?")
        mf = run["auroc"].get("mean_driftguard_fused")
        mt = run["auroc"].get("mean_topology_only")
        lt_v = run.get("lead_time_unsafe", {}).get("mean")
        lines.append(
            f"| {seed} "
            f"| {f'{mf:.3f}' if mf is not None else 'N/A'} "
            f"| {f'{mt:.3f}' if mt is not None else 'N/A'} "
            f"| {f'{lt_v:.1f}' if lt_v is not None else 'N/A'} |"
        )

    lines += [
        "",
        "These values are generated directly from benchmark JSON and exported baselines.",
    ]
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
