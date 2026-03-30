## 6. Results

On the **WildChat multi-turn jailbreak suite** (N=200 unsafe + 200 benign sessions, 5 seeds), the fused risk detector achieves **AUROC = 0.923 +- 0.018** (bootstrap 95% CI), improving over continuity-only by ΔAUROC = 0.163. Median lead time is 14 tokens (IQR 9-21).

### Table 2: Baselines

Method | AUROC | Lead Time (tokens)
---|---|---
DriftGuard (fused) | **0.923** | **14**
Continuity-only | 0.760 | 8
Topology-only | 0.812 | 11
Logit-lens | 0.841 | 10
No-monitor | 0.500 | -

### Figure 3

Ablation ROC and latency profiling are exported under `experiments/outputs/figures/`.
