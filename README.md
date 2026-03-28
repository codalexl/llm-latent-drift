# latent-dynamics

Dynamic trust regions from latent representation trajectories for LLM safety monitoring.

This repo now supports:
- Milestone 1: trajectory collection + visualization
- Milestone 2: trust-region fitting + comparative baselines
- Milestone 3: drift metrics + generator-shift evaluation
- DriftGuard runtime: online drift monitoring + activation steering (inference-time)

## Setup

### Option A: uv (recommended)
```bash
uv sync
```

### Option B: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Core commands

### 1) Collect trajectories
```bash
uv run python scripts/collect_trajectories.py \
  --model qwen3_8b \
  --dataset xstest \
  --num_samples 200 \
  --layers 24 \
  --4bit \
  --output ./activations
```

### 2) Visualize trajectories
```bash
uv run python scripts/visualize_trajectories.py \
  --model qwen3_8b \
  --activations activations/xstest/train/qwen3_8b/layer_24 \
  --fig_dir figures
```

### 3) Full safety pipeline (Milestone 2 + 3)
```bash
uv run latent-dynamics run-safety-pipeline \
  --activations activations/xstest/train/qwen3_8b/layer_24 \
  --output-dir experiments/outputs \
  --plot-drift \
  --seed 42
```

### Optional: real SaP baseline integration
```bash
uv run latent-dynamics run-safety-pipeline \
  --activations activations/xstest/train/qwen3_8b/layer_24 \
  --output-dir experiments/outputs \
  --plot-drift \
  --real-sap \
  --sap-repo-path .cache/baselines/SafetyPolytope \
  --seed 42
```

### Optional: shifted-domain eval
```bash
uv run latent-dynamics run-safety-pipeline \
  --activations activations/xstest/train/qwen3_8b/layer_24 \
  --shifted-activations activations/harmbench/train/qwen3_8b/layer_24 \
  --output-dir experiments/outputs \
  --plot-drift \
  --seed 42
```

### Modular commands
```bash
uv run latent-dynamics fit-trust-region \
  --activations activations/xstest/train/qwen3_8b/layer_24

uv run latent-dynamics compute-drift \
  --activations activations/xstest/train/qwen3_8b/layer_24 \
  --model-path models/trust_region_qwen3_8b.pkl

uv run latent-dynamics run-driftguard-session \
  --model llama_3_1_8b \
  --prompt "Let's talk about lab safety best practices." \
  --safe-reference-prompt "How should I safely store household chemicals?" \
  --safe-reference-prompt "Explain responsible disclosure in cybersecurity."
```

## One-command run (finalization helper)

```bash
uv run python experiments/run_all.py \
  --model gemma3_4b \
  --dataset toy_contrastive \
  --num-samples 20 \
  --output-dir experiments/outputs
```

This executes:
1. collect trajectories
2. visualize trajectories
3. run-safety-pipeline

## Main outputs

- `models/trust_region_{model}.pkl`
- `experiments/outputs/milestone2_comparison.md`
- `experiments/outputs/milestone2_comparison.tex`
- `experiments/outputs/milestone3_drift.md`
- `experiments/outputs/milestone3_drift.tex`
- `experiments/outputs/milestone23_report.json`
- `experiments/outputs/figures/` (drift histogram + boundary overlay, when `--plot-drift`)
- `figures/` (trajectory visualizations + `captions.json`)

## Reproducibility defaults

- Global random seed defaults to 42 in core modules.
- 70/15/15 split helpers use deterministic seed by default.
- `--seed` is exposed on safety pipeline commands.

## Notes

- For static PNG export from Plotly, install `kaleido`.
- 4-bit loading requires CUDA-compatible `bitsandbytes`.
- Real SaP integration calls SafetyPolytope's beaver pipeline script if available.

