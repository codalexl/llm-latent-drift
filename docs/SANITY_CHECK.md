# Sanity check — current state (finalized)

Quick snapshot of what is implemented and how to run it.

## Core CLI

- `uv run latent-dynamics run-safety-pipeline`
  - flags: `--activations`, `--shifted-activations` (optional), `--output-dir`, `--model` (optional sanity check), `--only-fit`, `--only-drift`, `--plot-drift`, `--real-sap`, `--sap-repo-path`, `--seed`
- `uv run latent-dynamics fit-trust-region`
- `uv run latent-dynamics compute-drift`
- `uv run latent-dynamics milestone23` (backward-compatible alias)

## End-to-end commands

### 1) Collect
```bash
uv run python scripts/collect_trajectories.py \
  --model gemma3_4b \
  --dataset toy_contrastive \
  --num_samples 20 \
  --output ./activations
```

### 2) Visualize
```bash
uv run python scripts/visualize_trajectories.py \
  --model gemma3_4b \
  --activations activations/toy_contrastive/train/gemma3_4b/layer_18 \
  --fig_dir figures
```

### 3) Safety pipeline (M2+M3)
```bash
uv run latent-dynamics run-safety-pipeline \
  --activations activations/toy_contrastive/train/gemma3_4b/layer_18 \
  --output-dir experiments/outputs \
  --plot-drift \
  --seed 42
```

### 4) Safety pipeline + real SaP
```bash
uv run latent-dynamics run-safety-pipeline \
  --activations activations/toy_contrastive/train/gemma3_4b/layer_18 \
  --output-dir experiments/outputs \
  --plot-drift \
  --real-sap \
  --sap-repo-path .cache/baselines/SafetyPolytope \
  --seed 42
```

## Key outputs

- `models/trust_region_{model}.pkl`
- `experiments/outputs/milestone2_comparison.md`
- `experiments/outputs/milestone2_comparison.tex`
- `experiments/outputs/milestone3_drift.md`
- `experiments/outputs/milestone3_drift.tex`
- `experiments/outputs/milestone23_report.json`
- `experiments/outputs/figures/milestone3_exit_time_histogram.(html/png)`
- `experiments/outputs/figures/milestone3_boundary_overlay_example.(html/png)`

## Lead-time visibility

`prefix_to_failure_lead_time` is now surfaced in:
- drift summary JSON (`milestone23_report.json`)
- per-unsafe-record entries
- table outputs (`milestone3_drift.md` / `.tex`)
- milestone comparison table (`milestone2_comparison.*`) via trust-region lead-time column
