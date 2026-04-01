"""
Convert driftguard_proper_v2_*.json to a single flat CSV.

Row granularity: one row per step, with seed-level and session-level
fields propagated down to each row.

Seed-level columns (repeated per session/step):
  seed, model, preset, device
  auroc_driftguard_fused, auroc_continuity_only, auroc_topology_only,
  auroc_logit_lens, auroc_no_monitor,
  auroc_mean_driftguard_fused, auroc_mean_continuity_only,
  auroc_mean_topology_only, auroc_mean_dynamics_only,
  auroc_alarm_frac,
  seed_latency_no_steer_mean_ms

Session-level columns (repeated per step):
  session_idx, condition, unsafe_label,
  session_alarms, first_alarm_token, first_alarm_lead_time,
  tda_attempted_steps, tda_executed_steps,
  tda_skipped_budget_steps, tda_skipped_stride_steps,
  session_mean_latency_ms

Step-level columns:
  step_idx, token_id,
  cosine_continuity, lipschitz_proxy,
  topology_diameter, topology_beta0, topology_beta1, topology_persistence_l1,
  risk_score, continuity_risk, lipschitz_risk, topology_risk,
  topology_diameter_risk, topology_persistence_l1_risk,
  topology_beta0_risk, topology_beta1_risk,
  tda_backend, tda_approximate,
  alarm, steered,
  probe_risk, dynamics_risk, steering_delta_norm, latency_ms
"""

import csv
import json
import sys
from pathlib import Path

STEP_FIELDS = [
    "token_id",
    "cosine_continuity",
    "lipschitz_proxy",
    "topology_diameter",
    "topology_beta0",
    "topology_beta1",
    "topology_persistence_l1",
    "risk_score",
    "continuity_risk",
    "lipschitz_risk",
    "topology_risk",
    "topology_diameter_risk",
    "topology_persistence_l1_risk",
    "topology_beta0_risk",
    "topology_beta1_risk",
    "tda_backend",
    "tda_approximate",
    "alarm",
    "steered",
    "probe_risk",
    "dynamics_risk",
    "steering_delta_norm",
    "latency_ms",
]

FIELDNAMES = [
    # Seed-level
    "seed",
    "model",
    "preset",
    "device",
    "auroc_driftguard_fused",
    "auroc_continuity_only",
    "auroc_topology_only",
    "auroc_logit_lens",
    "auroc_no_monitor",
    "auroc_mean_driftguard_fused",
    "auroc_mean_continuity_only",
    "auroc_mean_topology_only",
    "auroc_mean_dynamics_only",
    "auroc_alarm_frac",
    "seed_latency_no_steer_mean_ms",
    # Session-level
    "session_idx",
    "condition",
    "unsafe_label",
    "session_alarms",
    "first_alarm_token",
    "first_alarm_lead_time",
    "tda_attempted_steps",
    "tda_executed_steps",
    "tda_skipped_budget_steps",
    "tda_skipped_stride_steps",
    "session_mean_latency_ms",
    # Step-level
    "step_idx",
    *STEP_FIELDS,
]


def convert(src: Path, dst: Path) -> None:
    print(f"Loading {src} ...", flush=True)
    with src.open() as f:
        data = json.load(f)

    model = data.get("model", "")
    preset = data.get("preset", "")
    device = data.get("device", "")

    print(f"Writing {dst} ...", flush=True)
    total_rows = 0
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for run in data["runs"]:
            seed = run["seed"]
            auroc = run.get("auroc", {})
            latency = run.get("latency_ms", {})

            seed_base = {
                "seed": seed,
                "model": model,
                "preset": preset,
                "device": device,
                "auroc_driftguard_fused": auroc.get("driftguard_fused"),
                "auroc_continuity_only": auroc.get("continuity_only"),
                "auroc_topology_only": auroc.get("topology_only"),
                "auroc_logit_lens": auroc.get("logit_lens"),
                "auroc_no_monitor": auroc.get("no_monitor"),
                "auroc_mean_driftguard_fused": auroc.get("mean_driftguard_fused"),
                "auroc_mean_continuity_only": auroc.get("mean_continuity_only"),
                "auroc_mean_topology_only": auroc.get("mean_topology_only"),
                "auroc_mean_dynamics_only": auroc.get("mean_dynamics_only"),
                "auroc_alarm_frac": auroc.get("alarm_frac_auroc"),
                "seed_latency_no_steer_mean_ms": latency.get("no_steer_mean"),
            }

            sessions = run.get("sessions", {})
            for condition in ("no_steer", "steer"):
                session_list = sessions.get(condition, [])
                for sess_idx, sess in enumerate(session_list):
                    sess_base = {
                        **seed_base,
                        "session_idx": sess_idx,
                        "condition": condition,
                        "unsafe_label": sess.get("unsafe_label"),
                        "session_alarms": sess.get("alarms"),
                        "first_alarm_token": sess.get("first_alarm_token"),
                        "first_alarm_lead_time": sess.get("first_alarm_lead_time"),
                        "tda_attempted_steps": sess.get("tda_attempted_steps"),
                        "tda_executed_steps": sess.get("tda_executed_steps"),
                        "tda_skipped_budget_steps": sess.get("tda_skipped_budget_steps"),
                        "tda_skipped_stride_steps": sess.get("tda_skipped_stride_steps"),
                        "session_mean_latency_ms": sess.get("mean_step_latency_ms"),
                    }

                    for step_idx, step in enumerate(sess.get("steps", [])):
                        row = {
                            **sess_base,
                            "step_idx": step_idx,
                            **{f: step.get(f) for f in STEP_FIELDS},
                        }
                        writer.writerow(row)
                        total_rows += 1

    print(f"Done. Wrote {total_rows:,} rows to {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        src = Path("experiments/outputs/driftguard_proper_v2_gemma3_4b.json")
    else:
        src = Path(sys.argv[1])

    dst = src.with_suffix(".csv")
    convert(src, dst)
