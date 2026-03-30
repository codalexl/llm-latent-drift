from __future__ import annotations

import json
import random
from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import typer
import numpy as np

from latent_dynamics.config import DATASET_REGISTRY, MODEL_REGISTRY

app = typer.Typer(
    name="latent-dynamics",
    help="Latent trajectory analysis for language models.",
    add_completion=False,
)

class ModelKey(str, Enum):
    llama_3_1_8b = "llama_3_1_8b"
    llama_3_1_70b = "llama_3_1_70b"
    mistral_7b_instruct = "mistral_7b_instruct"
    gemma2_9b = "gemma2_9b"
    gemma2_27b = "gemma2_27b"
    qwen3_8b = "qwen3_8b"
    gemma3_4b = "gemma3_4b"
    gemma3_270m = "gemma3_270m"
    gemma3_12b = "gemma3_12b"


class DatasetKey(str, Enum):
    toy_contrastive = "toy_contrastive"
    wildjailbreak = "wildjailbreak"
    xstest = "xstest"


@app.command()
def extract(
    model: Annotated[
        ModelKey, typer.Option(help="Model key from registry.")
    ] = ModelKey.gemma3_4b,
    dataset: Annotated[
        DatasetKey, typer.Option(help="Dataset key from registry.")
    ] = DatasetKey.toy_contrastive,
    layer: Annotated[
        List[int],
        typer.Option(help="Layer index (repeat for multiple: --layer 5 --layer 10)."),
    ] = [],
    all_layers: Annotated[
        bool,
        typer.Option("--all-layers/--no-all-layers", help="Extract all layers."),
    ] = False,
    max_samples: Annotated[int, typer.Option(help="Maximum number of examples.")] = 120,
    max_input_tokens: Annotated[
        int, typer.Option(help="Maximum token sequence length.")
    ] = 256,
    output: Annotated[
        Path,
        typer.Option(
            help="Root output directory. Activations are saved under {dataset}/{model}/layer_{N}/."
        ),
    ] = Path("./activations"),
    push_to_hub: Annotated[
        Optional[str], typer.Option(help="HuggingFace Hub repo id (e.g. user/repo).")
    ] = None,
    device: Annotated[
        Optional[str], typer.Option(help="Device override (cuda/mps/cpu).")
    ] = None,
    use_generate: Annotated[
        bool, typer.Option(help="Generate continuation before extracting.")
    ] = False,
    max_new_tokens: Annotated[
        int, typer.Option(help="Tokens to generate (requires --use-generate).")
    ] = 256,
    include_prompt: Annotated[
        bool,
        typer.Option(
            "--include-prompt/--no-include-prompt",
            help="Include prompt tokens in trajectory when generating.",
        ),
    ] = True,
    load_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit/--no-4bit", help="Load model in 4-bit (CUDA + bitsandbytes)."
        ),
    ] = False,
    use_true_batch_inference: Annotated[
        bool,
        typer.Option(
            "--use-true-batch-inference",
            help="Use true batch inference.",
            is_flag=True,
        ),
    ] = False,
    inference_batch_size: Annotated[
        int,
        typer.Option(
            "--inference-batch-size",
            help=(
                "Batch size for true batch inference "
                "(used with --use-true-batch-inference)."
            ),
        ),
    ] = 16,
) -> None:
    """Extract hidden-state trajectories from a model and dataset, save to disk.

    Activations are stored under a structured path:
      {output}/{dataset}/{model}/layer_{N}/
    Each layer directory contains a metadata.json that records the layer index,
    input prompts, and generated text (when using --use-generate) so that
    activations can be matched back to their source data.
    """
    from tqdm.auto import tqdm

    from latent_dynamics.activations import extract_multi_layer_trajectories
    from latent_dynamics.config import DriftGuardConfig
    from latent_dynamics.data import load_examples, prepare_text_and_labels
    from latent_dynamics.hub import (
        METADATA_FILE,
        TRAJECTORIES_FILE,
        activation_subpath,
        save_activations_shard,
        write_activation_metadata,
    )
    from latent_dynamics.hub import (
        push_to_hub as _push,
    )
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device
    from latent_dynamics.utils import (
        TRAJECTORY_SHARD_MANIFEST_FILE,
        list_trajectory_shards,
        write_trajectory_shard_manifest,
    )

    if inference_batch_size < 1:
        raise typer.BadParameter("--inference-batch-size must be >= 1.")

    layer_indices: list[int] | None = None
    if all_layers:
        layer_indices = None
    elif layer:
        layer_indices = layer
    else:
        layer_indices = [5]

    cfg = DriftGuardConfig(
        model_key=model.value,
        dataset_key=dataset.value,
        max_samples=max_samples,
        max_input_tokens=max_input_tokens,
        layer_idx=layer_indices[0] if layer_indices else 0,
        device=resolve_device(device),
        use_generate=use_generate,
        max_new_tokens=max_new_tokens,
        include_prompt_in_trajectory=include_prompt,
        use_true_batch_inference=use_true_batch_inference,
        inference_batch_size=inference_batch_size,
    )

    typer.echo(f"Device: {cfg.device}")
    typer.echo(f"Model:  {cfg.model_key}")
    typer.echo(f"Dataset: {cfg.dataset_key} (max_samples={cfg.max_samples})")
    typer.echo(f"Layers:  {'all' if layer_indices is None else layer_indices}")
    typer.echo(
        f"True batch inference: {cfg.use_true_batch_inference} "
        f"(batch_size={cfg.inference_batch_size})"
    )

    ds, spec = load_examples(
        cfg.dataset_key,
        cfg.max_samples,
        stratify_labels=True,
        seed=cfg.random_state,
    )
    texts, labels, metadata = prepare_text_and_labels(ds, spec, return_metadata=True)

    if labels is None:
        typer.echo(
            "Warning: no labels produced. Downstream analysis will be limited.",
            err=True,
        )

    typer.echo(f"Loaded {len(texts)} examples, loading model...")
    mdl, tokenizer = load_model_and_tokenizer(
        cfg.model_key, cfg.device, load_in_4bit=load_4bit
    )

    typer.echo(
        f"Extracting trajectories for layers "
        f"{'all' if layer_indices is None else layer_indices} ..."
    )
    empty_result = extract_multi_layer_trajectories(
        mdl,
        tokenizer,
        [],
        layer_indices=layer_indices,
        cfg=cfg,
        show_progress=False,
    )
    extracted_layers = sorted(empty_result.per_layer.keys())

    layer_dirs = {
        li: output / activation_subpath(cfg.dataset_key, cfg.model_key, li)
        for li in extracted_layers
    }
    for layer_dir in layer_dirs.values():
        layer_dir.mkdir(parents=True, exist_ok=True)
        single_path = layer_dir / TRAJECTORIES_FILE
        if single_path.exists():
            single_path.unlink()
        metadata_path = layer_dir / METADATA_FILE
        if metadata_path.exists():
            metadata_path.unlink()
        manifest_path = layer_dir / TRAJECTORY_SHARD_MANIFEST_FILE
        if manifest_path.exists():
            manifest_path.unlink()
        for shard_path in list_trajectory_shards(layer_dir):
            shard_path.unlink()

    all_token_texts: list[list[str]] = []
    all_generated_texts: list[str | None] = []
    next_idx_by_layer: dict[int, int] = {li: 0 for li in extracted_layers}
    shard_idx_by_layer: dict[int, int] = {li: 0 for li in extracted_layers}
    shard_manifest_entries_by_layer: dict[int, list[dict[str, object]]] = {
        li: [] for li in extracted_layers
    }

    chunk_size = max(1, int(cfg.inference_batch_size))
    chunk_starts = range(0, len(texts), chunk_size)
    chunk_iter = tqdm(chunk_starts, desc="extract chunks", disable=len(texts) == 0)
    for start in chunk_iter:
        chunk_texts = texts[start : start + chunk_size]
        chunk_result = extract_multi_layer_trajectories(
            mdl,
            tokenizer,
            chunk_texts,
            layer_indices=layer_indices,
            cfg=cfg,
            show_progress=False,
        )
        all_token_texts.extend(chunk_result.token_texts)
        all_generated_texts.extend(chunk_result.generated_texts)

        for li in extracted_layers:
            start_idx = next_idx_by_layer[li]
            next_idx, _ = save_activations_shard(
                output_dir=layer_dirs[li],
                trajectories=chunk_result.per_layer[li],
                start_idx=start_idx,
                shard_idx=shard_idx_by_layer[li],
                manifest_entries=shard_manifest_entries_by_layer[li],
            )
            next_idx_by_layer[li] = next_idx
            shard_idx_by_layer[li] += 1

    typer.echo(f"Extracted {len(texts)} trajectories x {len(extracted_layers)} layers.")

    extra_metadata: dict[str, object] = {}
    if metadata is not None:
        extra_metadata["example_metadata"] = metadata

    for li in extracted_layers:
        layer_cfg = DriftGuardConfig(**{**cfg.model_dump(), "layer_idx": li})
        write_trajectory_shard_manifest(
            output_dir=layer_dirs[li],
            entries=shard_manifest_entries_by_layer[li],
        )
        write_activation_metadata(
            output_dir=layer_dirs[li],
            trajectories_count=next_idx_by_layer[li],
            texts=texts,
            labels=labels,
            token_texts=all_token_texts,
            cfg=layer_cfg,
            generated_texts=all_generated_texts,
            extra_metadata=(extra_metadata if extra_metadata else None),
        )
        typer.echo(f"  Saved layer {li} -> {layer_dirs[li]}")

    if push_to_hub:
        url = _push(output, push_to_hub)
        typer.echo(f"Pushed to {url}")


@app.command("run-driftguard-session")
def run_driftguard_session_cmd(
    model: Annotated[
        ModelKey, typer.Option(help="Model key from registry.")
    ] = ModelKey.gemma3_12b,
    prompt: Annotated[
        str, typer.Option(help="Prompt to evaluate with online drift monitoring.")
    ] = "You are a helpful assistant.",
    device: Annotated[
        Optional[str], typer.Option(help="Device override (cuda/mps/cpu).")
    ] = None,
    layer_idx: Annotated[
        int, typer.Option(help="Layer index used for drift monitoring.")
    ] = -1,
    max_new_tokens: Annotated[
        int, typer.Option(help="Maximum generated tokens.")
    ] = 128,
    cosine_floor: Annotated[
        float, typer.Option(help="Continuity floor for cosine(h_t, h_{t-1}).")
    ] = 0.96,
    lipschitz_ceiling: Annotated[
        float, typer.Option(help="Ceiling for local Lipschitz proxy.")
    ] = 0.20,
    risk_threshold: Annotated[
        float, typer.Option(help="Alarm threshold on fused risk score.")
    ] = 0.5,
    topology_window: Annotated[
        int, typer.Option(help="Window size for TDA snapshots.")
    ] = 8,
    topology_stride: Annotated[
        int, typer.Option(help="Stride for TDA snapshot updates.")
    ] = 4,
    tda_latency_budget_ms: Annotated[
        float, typer.Option(help="Per-step latency budget for running TDA.")
    ] = 50.0,
    tda_enabled: Annotated[
        bool,
        typer.Option(
            "--tda-enabled/--no-tda-enabled",
            help="Enable or disable ripser-based topology features.",
        ),
    ] = True,
    pca_components: Annotated[
        int,
        typer.Option(help="PCA components used before topology metrics."),
    ] = 3,
    topology_diameter_ceiling: Annotated[
        float, typer.Option(help="Ceiling for topology diameter risk term.")
    ] = 1.5,
    topology_beta1_ceiling: Annotated[
        float, typer.Option(help="Ceiling for topology beta1 risk term.")
    ] = 1.0,
    continuity_weight: Annotated[
        float, typer.Option(help="Weight for cosine continuity risk term.")
    ] = 0.40,
    lipschitz_weight: Annotated[
        float, typer.Option(help="Weight for Lipschitz risk term.")
    ] = 0.35,
    topology_weight: Annotated[
        float, typer.Option(help="Weight for topology risk term.")
    ] = 0.25,
    probe_weight: Annotated[
        float, typer.Option(help="Weight for contrastive probe in hybrid risk.")
    ] = 0.60,
    steering_strength: Annotated[
        float, typer.Option(help="Intervention strength toward safe reference.")
    ] = 0.20,
    contrastive_steering_strength: Annotated[
        float, typer.Option(help="Activation delta scale for contrastive steering.")
    ] = 0.25,
    use_contrastive_probe: Annotated[
        bool,
        typer.Option(
            "--use-contrastive-probe/--no-use-contrastive-probe",
            help="Fuse projection-based probe risk with dynamics/TDA risk.",
        ),
    ] = True,
    use_nnsight: Annotated[
        bool,
        typer.Option(
            "--use-nnsight/--no-use-nnsight",
            help="Use nnsight tracing for hidden capture and steering interventions.",
        ),
    ] = False,
    nnsight_full_prefix_trace: Annotated[
        bool,
        typer.Option(
            "--nnsight-full-prefix-trace/--no-nnsight-full-prefix-trace",
            help=(
                "When using --use-nnsight, trace full running prefix each step "
                "(higher fidelity, higher latency)."
            ),
        ),
    ] = True,
    nnsight_fail_open: Annotated[
        bool,
        typer.Option(
            "--nnsight-fail-open/--no-nnsight-fail-open",
            help=(
                "If nnsight steering raises at runtime, skip steering for that step "
                "instead of failing the full session."
            ),
        ),
    ] = True,
    random_seed: Annotated[
        int,
        typer.Option(help="Random seed for numpy/torch/python RNGs."),
    ] = 42,
    load_4bit: Annotated[
        bool,
        typer.Option(
            "--4bit/--no-4bit",
            help="Load model in 4-bit mode when supported (CUDA only).",
        ),
    ] = False,
    safe_reference_prompt: Annotated[
        List[str],
        typer.Option(
            help=(
                "Repeatable safe prompts used to estimate safe reference state. "
                "Provide multiple values for better calibration."
            )
        ),
    ] = [],
    harmful_reference_prompt: Annotated[
        List[str],
        typer.Option(
            help=(
                "Repeatable harmful prompts used to compute contrastive direction. "
                "When provided together with safe prompts, hybrid probe and "
                "contrastive steering are enabled."
            )
        ),
    ] = [],
    output_json: Annotated[
        Optional[Path],
        typer.Option(help="Optional output JSON path for per-step drift logs."),
    ] = None,
) -> None:
    """Run a single online session with early-warning + activation steering."""
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device
    from latent_dynamics.contrastive_vectors import compute_contrastive_vector
    from latent_dynamics.online_runtime import (
        DriftGuardConfig,
        estimate_safe_reference,
        load_nnsight_model,
        run_driftguard_session,
    )

    resolved_device = resolve_device(device)
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    except Exception:
        pass

    mdl, tokenizer = load_model_and_tokenizer(
        model.value,
        resolved_device,
        load_in_4bit=bool(load_4bit and resolved_device == "cuda"),
    )
    cfg = DriftGuardConfig(
        layer_idx=layer_idx,
        max_new_tokens=max_new_tokens,
        cosine_floor=cosine_floor,
        lipschitz_ceiling=lipschitz_ceiling,
        risk_threshold=risk_threshold,
        topology_window=topology_window,
        topology_stride=topology_stride,
        tda_latency_budget_ms=tda_latency_budget_ms,
        tda_enabled=tda_enabled,
        pca_components=pca_components,
        topology_diameter_ceiling=topology_diameter_ceiling,
        topology_beta1_ceiling=topology_beta1_ceiling,
        continuity_weight=continuity_weight,
        lipschitz_weight=lipschitz_weight,
        topology_weight=topology_weight,
        probe_weight=probe_weight,
        steering_strength=steering_strength,
        contrastive_steering_strength=contrastive_steering_strength,
        use_contrastive_probe=use_contrastive_probe,
        safe_prompts=list(safe_reference_prompt),
        harmful_prompts=list(harmful_reference_prompt),
        use_nnsight=use_nnsight,
        nnsight_full_prefix_trace=nnsight_full_prefix_trace,
        nnsight_fail_open=nnsight_fail_open,
        random_seed=random_seed,
    )

    safe_reference = None
    if safe_reference_prompt:
        safe_reference = estimate_safe_reference(
            model=mdl,
            tokenizer=tokenizer,
            prompts=safe_reference_prompt,
            device=resolved_device,
            layer_idx=layer_idx,
        )
    contrastive_vectors: dict[str, list[float]] | None = None
    if safe_reference_prompt and harmful_reference_prompt:
        try:
            vec = compute_contrastive_vector(
                model=mdl,
                tokenizer=tokenizer,
                safe_prompts=safe_reference_prompt,
                harmful_prompts=harmful_reference_prompt,
                layer_idx=layer_idx,
                cfg=cfg,
                device=resolved_device,
            )
            contrastive_vectors = {f"layer_{layer_idx}": vec.astype(float).tolist()}
        except Exception as exc:
            typer.echo(f"Warning: contrastive vector computation failed: {exc}", err=True)

    active_model = mdl
    if use_nnsight:
        hf_id = MODEL_REGISTRY[model.value]["hf_id"]
        active_model = load_nnsight_model(
            model_path=hf_id,
            device_map="auto",
            load_in_4bit=bool(load_4bit and resolved_device == "cuda"),
        )
    result = run_driftguard_session(
        model=active_model,
        tokenizer=tokenizer,
        prompt=prompt,
        cfg=cfg,
        device=resolved_device,
        safe_reference=safe_reference,
        contrastive_vectors=contrastive_vectors,
    )
    payload = {
        "model": model.value,
        "device": resolved_device,
        "prompt": prompt,
        "generated_text": result.generated_text,
        "alarms": result.alarms,
        "steered_steps": result.steered_steps,
        "first_alarm_token": result.first_alarm_token,
        "first_alarm_lead_time": result.first_alarm_lead_time,
        "mean_step_latency_ms": result.mean_step_latency_ms,
        "tda_attempted_steps": result.tda_attempted_steps,
        "tda_executed_steps": result.tda_executed_steps,
        "tda_skipped_budget_steps": result.tda_skipped_budget_steps,
        "tda_skipped_stride_steps": result.tda_skipped_stride_steps,
        "steps": [
            {
                "token_id": s.token_id,
                "cosine_continuity": s.cosine_continuity,
                "lipschitz_proxy": s.lipschitz_proxy,
                "topology_diameter": s.topology_diameter,
                "topology_beta0": s.topology_beta0,
                "topology_beta1": s.topology_beta1,
                "topology_persistence_l1": s.topology_persistence_l1,
                "risk_score": s.risk_score,
                "probe_risk": s.probe_risk,
                "dynamics_risk": s.dynamics_risk,
                "continuity_risk": s.continuity_risk,
                "lipschitz_risk": s.lipschitz_risk,
                "topology_risk": s.topology_risk,
                "topology_diameter_risk": s.topology_diameter_risk,
                "topology_persistence_l1_risk": s.topology_persistence_l1_risk,
                "topology_beta0_risk": s.topology_beta0_risk,
                "topology_beta1_risk": s.topology_beta1_risk,
                "tda_backend": s.tda_backend,
                "tda_approximate": s.tda_approximate,
                "alarm": s.alarm,
                "steered": s.steered,
                "steering_delta_norm": s.steering_delta_norm,
                "latency_ms": s.latency_ms,
            }
            for s in result.steps
        ],
    }
    text = json.dumps(payload, indent=2)
    typer.echo(text)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text)
        typer.echo(f"Wrote report to {output_json}")


@app.command()
def calibrate(
    model_key: Annotated[
        ModelKey, typer.Option("--model-key", help="Model key from registry.")
    ] = ModelKey.gemma3_4b,
    dataset_key: Annotated[
        DatasetKey, typer.Option("--dataset-key", help="Dataset key from registry.")
    ] = DatasetKey.toy_contrastive,
    max_samples: Annotated[
        int, typer.Option(help="Maximum labeled samples for calibration run.")
    ] = 24,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "--calibration-output",
            help="Where to write calibration JSON.",
        ),
    ] = Path("calibration_results.json"),
    layer_idx: Annotated[
        int, typer.Option(help="Layer index used by online runtime.")
    ] = -1,
    max_new_tokens: Annotated[
        int, typer.Option(help="Generated tokens per prompt during calibration.")
    ] = 24,
    device: Annotated[
        Optional[str], typer.Option(help="Device override (cuda/mps/cpu).")
    ] = None,
    random_seed: Annotated[
        int, typer.Option(help="Reproducible random seed.")
    ] = 42,
) -> None:
    """Calibrate risk threshold/weights against labeled prompts."""
    from latent_dynamics.calibration import calibrate_risk_score
    from latent_dynamics.config import DriftGuardConfig
    from latent_dynamics.data import load_examples, prepare_text_and_labels
    from latent_dynamics.models import load_model_and_tokenizer, resolve_device

    ds, spec = load_examples(dataset_key.value, max_samples=max_samples, stratify_labels=True, seed=random_seed)
    texts, labels = prepare_text_and_labels(ds, spec)
    if labels is None:
        raise typer.BadParameter("Selected dataset does not provide labels for calibration.")
    resolved_device = resolve_device(device)
    mdl, tokenizer = load_model_and_tokenizer(model_key.value, resolved_device, load_in_4bit=False)
    cfg = DriftGuardConfig(
        model_key=model_key.value,
        dataset_key=dataset_key.value,
        layer_idx=layer_idx,
        max_new_tokens=max_new_tokens,
        random_seed=random_seed,
        use_nnsight=False,
    )
    result = calibrate_risk_score(
        cfg=cfg,
        model=mdl,
        tokenizer=tokenizer,
        prompts=texts,
        labels=labels.tolist(),
        device=resolved_device,
        output_path=output,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def list_models() -> None:
    """List available model keys."""
    for key, spec in MODEL_REGISTRY.items():
        typer.echo(f"  {key:20s}  {spec['hf_id']}")


@app.command()
def list_datasets() -> None:
    """List available dataset keys."""
    for key, spec in DATASET_REGISTRY.items():
        source = spec.path or "built-in"
        typer.echo(f"  {key:20s}  {source}")


if __name__ == "__main__":
    app()
