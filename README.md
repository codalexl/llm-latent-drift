# latent-dynamics

Inference-time DriftGuard research code for hidden-state drift detection and causal activation steering on fixed LLMs.

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
  --activations activations/xstest/qwen3_8b/layer_24 \
  --fig_dir figures
```

### 3) Run online DriftGuard session
```bash
uv run latent-dynamics run-driftguard-session \
  --model llama_3_1_8b \
  --prompt "Let's discuss responsible cybersecurity disclosure." \
  --safe-reference-prompt "How do safety policies reduce misuse?" \
  --safe-reference-prompt "Give safe guidance for lab procedures." \
  --risk-threshold 0.5 \
  --use-nnsight
```

### 4) DriftGuard threat preset benchmark
```bash
uv run python experiments/run_driftguard_benchmark.py \
  --model gemma3_4b \
  --preset multi_turn_jailbreak \
  --quantization-sweep \
  --output-json experiments/outputs/driftguard_benchmark.json
```

### 5) Calibrate DriftGuard risk scores (label-driven)
```bash
uv run latent-dynamics calibrate \
  --model-key gemma3_4b \
  --dataset-key toy_contrastive \
  --max-samples 24 \
  --output calibration_results.json
```
This command now performs a label-driven search over risk weights, topology scales, and threshold, then writes recommended config updates plus ROC/PR curves.

### 6) Generate benchmark figures
```bash
uv run python experiments/plot_driftguard_benchmark.py \
  --report-json experiments/outputs/driftguard_benchmark.json \
  --activations activations/toy_contrastive/gemma3_4b/layer_18 \
  --out-dir experiments/outputs/figures
```

## Main outputs

- `experiments/outputs/driftguard_benchmark.json`
- `experiments/outputs/figures/` (cosine heatmap, lead-time distribution, latency bar, ablation ROC, optional PCA/persistence)
- `activations/{dataset}/{model}/layer_{N}/` (trajectory shards + metadata)

## Reproducibility defaults

- `--seed` is exposed on extraction and benchmark scripts.
- Benchmark presets are fixed and version-controlled for repeatability.

## Notes

- For static PNG export from Plotly, install `kaleido`.
- 4-bit loading requires CUDA-compatible `bitsandbytes`.
- Calibration in this repository is fully label-driven.

