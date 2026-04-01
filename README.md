# latent-dynamics

Inference-time DriftGuard: latent hidden-state drift detection, TDA-based risk scoring, online monitoring, and calibration for fixed LLMs.

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

## Core DriftGuard Commands

### 1) Collect trajectories
```bash
uv run python scripts/collect_trajectories.py \
  --model qwen3_8b \
  --dataset wildchat \
  --num_samples 200 \
  --layers 24 \
  --output ./activations
```

### 2) Visualize trajectories
```bash
uv run python scripts/visualize_trajectories.py \
  --model qwen3_8b \
  --activations activations/xstest/qwen3_8b/layer_24 \
  --fig_dir figures
```

### 3) Run online DriftGuard session
```bash
uv run latent-dynamics run-driftguard-session \
  --model llama_3_1_8b \
  --prompt "Let's discuss responsible cybersecurity disclosure." \
  --risk-threshold 0.5
```

### 4) DriftGuard threat preset benchmark
Uses a **hybrid suite**: benign sessions from `allenai/WildChat-1M`, unsafe prompts from `allenai/wildjailbreak` (`adversarial_harmful`) prepended with a short synthetic two-turn chat.

```bash
uv run python experiments/run_driftguard_benchmark.py \
  --model llama-3.1-8b \
  --preset wildchat_multi_turn \
  --num-seeds 5 \
  --output-json experiments/outputs/driftguard_benchmark.json
```

### 5) Calibrate DriftGuard risk scores (label-driven)
```bash
uv run latent-dynamics calibrate \
  --model-key gemma3_4b \
  --dataset-key wildchat \
  --max-samples 24 \
  --output calibration_results.json
```
This command now performs a label-driven search over risk weights, topology scales, and threshold, then writes recommended config updates plus ROC/PR curves.

### 6) Generate benchmark figures
```bash
uv run python experiments/plot_driftguard_benchmark.py \
  --report-json experiments/outputs/driftguard_benchmark.json \
  --activations activations/wildchat/gemma3_4b/layer_18 \
  --out-dir experiments/outputs/figures
```

### 7) Generate paper-ready Results section
```bash
uv run python experiments/generate_results_markdown.py \
  --report-json experiments/outputs/driftguard_benchmark.json \
  --output-md experiments/outputs/driftguard_results_section.md \
  --replace-paper-section
```

## Main outputs

- `experiments/outputs/driftguard_benchmark.json`
- `experiments/outputs/driftguard_baseline_table.csv`
- `experiments/outputs/figures/` (cosine heatmap, lead-time distribution, latency bar, ablation ROC, optional PCA/persistence)
- `experiments/outputs/driftguard_results_section.md`
- `activations/{dataset}/{model}/layer_{N}/` (trajectory shards + metadata)

## Reproducibility defaults

- `--seed` is exposed on extraction and benchmark scripts.
- Benchmark presets are fixed and version-controlled for repeatability.

## Future Work

Causal activation steering (applying hidden-state interventions toward a safe reference at alarm time) is planned as a v2 extension. The code is preserved in `src/latent_dynamics/steering.py` and can be re-enabled with `--enable-steering` on the CLI or `enable_steering=True` in `DriftGuardConfig`. The full integrated version is archived on the `archive/full-steering-integration` branch.

The current focus is robust latent drift detection, TDA-based risk scoring, online monitoring, and calibration.

## Notes

- For static PNG export from Plotly, install `kaleido`.
- Calibration in this repository is fully label-driven.

