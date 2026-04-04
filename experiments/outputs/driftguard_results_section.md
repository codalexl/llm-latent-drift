# [SECTION 6: RESULTS]

### 6.1 Cross-model summary (multi-seed)

All runs aggregate **five random seeds** over the hybrid WildChat-1M / WildJailbreak benchmark (detection-only / no-steer condition unless otherwise noted). Primary scores use **mean-session** risk aggregation (see subsection notes below).

| Model | Source JSON | AUROC fused (mean-sess.) | AUROC topology-only | Lead time unsafe (mean, 95% CI tokens) | Latency (ms/token) | TDA exec. ratio | Steering |
|---|---|---:|---:|---|---:|---:|---|
| `qwen3_8b` | `driftguard_benchmark_qwen.json` | 0.716 ± 0.021 [0.697, 0.734] | 0.716 ± 0.021 [0.697, 0.734] | 77.6 [77.4, 77.9] | 27.39 | 1.000 | off |
| `llama_3_1_8b` | `driftguard_benchmark_llama.json` | 0.657 ± 0.015 [0.643, 0.670] | 0.641 ± 0.013 [0.630, 0.652] | 60.8 [59.9, 61.6] | 29.77 | 1.000 | off |
| `gemma3_4b` | `driftguard_benchmark_gemma.json` | 0.647 ± 0.017 [0.631, 0.662] | 0.479 ± 0.024 [0.458, 0.500] | 53.2 [52.0, 54.4] | 28.69 | 1.000 | off |

### 6.2 Interpretation (reported honestly from aggregates)

- **Qwen3-8B** shows the strongest mean-session separation in this suite; fused and topology-only AUROC are effectively identical, so the persistent-homology channel carries the fused signal.
- **Gemma-3-4B** exhibits a **topology-only AUROC near chance** on mean-session aggregation, while the fused score remains higher—continuity/dynamics terms provide the usable discrimination.
- **Llama-3.1-8B** is intermediate: topology-only AUROC is well above chance but still trails the fused score.
- **Overall scale:** fused mean-session AUROC spans roughly **0.65–0.72** on this benchmark; Qwen is the only backbone where topology alone matches the full fused detector.

### 6.3 Verification checklist (automated from JSON)

- **TDA execution ratio ≥ 0.8 (all models):** yes
- **Steering scope (benchmark JSON):** all detection-only (no steer sessions)
- **Calibration / thresholds:** confirm deployed weights match the calibration JSONs referenced in the paper (this markdown only reflects benchmark exports).

### 6.4 Per-model detailed tables

## `qwen3_8b` — full breakdown (*driftguard_benchmark_qwen.json*)

### Detection quality (multi-seed aggregation)
On the **WildChat-1M + WildJailbreak** hybrid suite (benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak `adversarial_harmful` prompts in synthetic two-turn chats; N≈200+200, 5 seeds) using model `qwen3_8b`, the fused detector reaches AUROC 0.716 ± 0.021 [0.697, 0.734] (mean-session aggregation, which is more robust to per-session risk saturation than max-aggregation).

The topology-only ablation achieves 0.716 ± 0.021 [0.697, 0.734] AUROC, matching the fused score (within numerical tolerance), so persistent-homology risk dominates the fused detector here.

### Early warning and runtime profile
- **Mean unsafe-session lead time:** 77.612 ± 5.517 (95% CI [77.370, 77.854]) tokens (median 80.0).
- **Mean token latency (detection-only):** 27.388 ms/token.
- **Steering latency:** not measured in this run (detection-only mode).

### Causal steering efficacy
Steering condition not run. Rerun without `--skip-steer` to evaluate intervention efficacy.

### Baseline comparison

> **Note:** *max-agg* uses the maximum risk score across a session; *mean-agg* uses the mean. Max-agg collapses when risk is frequently capped (→ AUROC = 0.5 for all methods). Mean-agg is the honest metric reported as primary in this paper.

| Method | AUROC (max-agg) | AUROC (mean-agg) |
|---|---:|---:|
| DriftGuard (fused) | 0.652 ± 0.018 [0.636, 0.668] | 0.716 ± 0.021 [0.697, 0.734] |
| Topology-only | 0.652 ± 0.018 [0.636, 0.668] | 0.716 ± 0.021 [0.697, 0.734] |
| Continuity-only | 0.447 ± 0.027 [0.423, 0.470] | 0.400 ± 0.017 [0.385, 0.414] |
| Dynamics-only | 0.652 ± 0.018 [0.636, 0.668] | 0.716 ± 0.021 [0.697, 0.734] |
| Alarm-fraction | 0.712 ± 0.020 [0.695, 0.730] | 0.712 ± 0.020 [0.695, 0.730] |
| No-monitor | 0.500 ± 0.000 [0.500, 0.500] | 0.500 ± 0.000 [0.500, 0.500] |

### Per-seed breakdown

| Seed | AUROC (mean-fused) | AUROC (topo-only) | Lead time (mean) |
|---:|---:|---:|---:|
| 42 | 0.705 | 0.705 | 78.0 |
| 43 | 0.690 | 0.690 | 78.0 |
| 44 | 0.736 | 0.736 | 77.2 |
| 45 | 0.708 | 0.708 | 77.7 |
| 46 | 0.740 | 0.740 | 77.3 |

These values are generated directly from benchmark JSON and exported baselines.

## `llama_3_1_8b` — full breakdown (*driftguard_benchmark_llama.json*)

### Detection quality (multi-seed aggregation)
On the **WildChat-1M + WildJailbreak** hybrid suite (benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak `adversarial_harmful` prompts in synthetic two-turn chats; N≈200+200, 5 seeds) using model `llama_3_1_8b`, the fused detector reaches AUROC 0.657 ± 0.015 [0.643, 0.670] (mean-session aggregation, which is more robust to per-session risk saturation than max-aggregation).

The topology-only ablation achieves 0.641 ± 0.013 [0.630, 0.652] AUROC; fused AUROC is modestly higher, so fusion still adds measurable lift beyond topology on this backbone.

### Early warning and runtime profile
- **Mean unsafe-session lead time:** 60.766 ± 19.197 (95% CI [59.902, 61.630]) tokens (median 68.0).
- **Mean token latency (detection-only):** 29.774 ms/token.
- **Steering latency:** not measured in this run (detection-only mode).

### Causal steering efficacy
Steering condition not run. Rerun without `--skip-steer` to evaluate intervention efficacy.

### Baseline comparison

> **Note:** *max-agg* uses the maximum risk score across a session; *mean-agg* uses the mean. Max-agg collapses when risk is frequently capped (→ AUROC = 0.5 for all methods). Mean-agg is the honest metric reported as primary in this paper.

| Method | AUROC (max-agg) | AUROC (mean-agg) |
|---|---:|---:|
| DriftGuard (fused) | 0.623 ± 0.016 [0.609, 0.637] | 0.657 ± 0.015 [0.643, 0.670] |
| Topology-only | 0.626 ± 0.019 [0.609, 0.642] | 0.641 ± 0.013 [0.630, 0.652] |
| Continuity-only | 0.475 ± 0.031 [0.448, 0.502] | 0.538 ± 0.023 [0.518, 0.559] |
| Dynamics-only | 0.623 ± 0.016 [0.609, 0.637] | 0.657 ± 0.015 [0.643, 0.670] |
| Alarm-fraction | 0.620 ± 0.013 [0.609, 0.632] | 0.620 ± 0.013 [0.609, 0.632] |
| No-monitor | 0.500 ± 0.000 [0.500, 0.500] | 0.500 ± 0.000 [0.500, 0.500] |

### Per-seed breakdown

| Seed | AUROC (mean-fused) | AUROC (topo-only) | Lead time (mean) |
|---:|---:|---:|---:|
| 42 | 0.669 | 0.654 | 60.8 |
| 43 | 0.635 | 0.623 | 60.1 |
| 44 | 0.672 | 0.652 | 61.9 |
| 45 | 0.661 | 0.641 | 60.0 |
| 46 | 0.647 | 0.635 | 61.0 |

These values are generated directly from benchmark JSON and exported baselines.

## `gemma3_4b` — full breakdown (*driftguard_benchmark_gemma.json*)

### Detection quality (multi-seed aggregation)
On the **WildChat-1M + WildJailbreak** hybrid suite (benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak `adversarial_harmful` prompts in synthetic two-turn chats; N≈200+200, 5 seeds) using model `gemma3_4b`, the fused detector reaches AUROC 0.647 ± 0.017 [0.631, 0.662] (mean-session aggregation, which is more robust to per-session risk saturation than max-aggregation).

The topology-only ablation achieves 0.479 ± 0.024 [0.458, 0.500] AUROC—near chance under mean-session aggregation—while fused AUROC is higher; continuity and dynamics terms provide the ranked separation.

### Early warning and runtime profile
- **Mean unsafe-session lead time:** 53.173 ± 23.245 (95% CI [51.972, 54.374]) tokens (median 60.0).
- **Mean token latency (detection-only):** 28.687 ms/token.
- **Steering latency:** not measured in this run (detection-only mode).

### Causal steering efficacy
Steering condition not run. Rerun without `--skip-steer` to evaluate intervention efficacy.

### Baseline comparison

> **Note:** *max-agg* uses the maximum risk score across a session; *mean-agg* uses the mean. Max-agg collapses when risk is frequently capped (→ AUROC = 0.5 for all methods). Mean-agg is the honest metric reported as primary in this paper.

| Method | AUROC (max-agg) | AUROC (mean-agg) |
|---|---:|---:|
| DriftGuard (fused) | 0.528 ± 0.016 [0.514, 0.541] | 0.647 ± 0.017 [0.631, 0.662] |
| Topology-only | 0.436 ± 0.015 [0.424, 0.449] | 0.479 ± 0.024 [0.458, 0.500] |
| Continuity-only | 0.665 ± 0.009 [0.658, 0.673] | 0.799 ± 0.018 [0.783, 0.815] |
| Dynamics-only | 0.528 ± 0.016 [0.514, 0.541] | 0.647 ± 0.017 [0.631, 0.662] |
| Alarm-fraction | 0.540 ± 0.015 [0.527, 0.554] | 0.540 ± 0.015 [0.527, 0.554] |
| No-monitor | 0.500 ± 0.000 [0.500, 0.500] | 0.500 ± 0.000 [0.500, 0.500] |

### Per-seed breakdown

| Seed | AUROC (mean-fused) | AUROC (topo-only) | Lead time (mean) |
|---:|---:|---:|---:|
| 42 | 0.656 | 0.495 | 53.4 |
| 43 | 0.657 | 0.501 | 53.1 |
| 44 | 0.646 | 0.472 | 51.6 |
| 45 | 0.617 | 0.442 | 53.7 |
| 46 | 0.658 | 0.485 | 54.0 |

These values are generated directly from benchmark JSON and exported baselines.
