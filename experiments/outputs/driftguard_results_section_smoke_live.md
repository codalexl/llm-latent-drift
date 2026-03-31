# [SECTION 6: RESULTS]

### 6.1 Detection quality (multi-seed aggregation)
On the **WildChat-1M + WildJailbreak** hybrid suite (benign = non-toxic WildChat-1M multi-turn; unsafe = WildJailbreak `adversarial_harmful` prompts in synthetic two-turn chats; N\approx200+200, 1 seeds) using model `mistral_7b_instruct`, the fused detector reaches AUROC 0.750.

### 6.2 Early warning and runtime profile
- **Mean unsafe-session lead time:** 94.000 tokens (median 94.0).
- **Mean token latency (no-steer):** 986.951 ms/token.
- **Mean token latency (steer):** 929.705 ms/token.

### 6.3 Causal steering efficacy
- Unsafe alarm rate: no-steer 1.000 → steer 1.000 (Δ = 0.000).
- Benign alarm rate: no-steer 0.500 → steer 0.500 (Δ = 0.000).

### 6.4 Baseline table
| Baseline | AUROC | PR-AUC | F1 |
|---|---:|---:|---:|
| DriftGuard (fused) | 0.750 | N/A | N/A |
| Continuity-only | 0.750 | N/A | N/A |
| Topology-only | 0.750 | N/A | N/A |
| Logit-lens | 0.750 | N/A | N/A |
| No-monitor | 0.500 | N/A | N/A |

These values are generated directly from benchmark JSON and exported baselines.
