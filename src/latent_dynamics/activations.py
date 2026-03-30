from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.models import resolve_device

# Global seeding for reproducible trajectory extraction.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


@dataclass
class ExtractionResult:
    """Result of multi-layer trajectory extraction."""

    per_layer: dict[int, list[np.ndarray]]
    token_texts: list[list[str]]
    input_prompts: list[str]
    generated_texts: list[str | None]


def _trim_trailing_pad(ids: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if ids.numel() == 0 or pad_token_id is None:
        return ids
    non_pad = torch.nonzero(ids != pad_token_id, as_tuple=False)
    if non_pad.numel() == 0:
        return ids[:0]
    last_idx = int(non_pad[-1].item()) + 1
    return ids[:last_idx]


def _extract_ids_and_positions_generate(
    seq_row: torch.Tensor,
    prompt_mask_row: torch.Tensor,
    prompt_len: int,
    include_prompt: bool,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if include_prompt:
        prompt_positions = torch.nonzero(prompt_mask_row, as_tuple=False).squeeze(-1)
        prompt_ids = seq_row[:prompt_len].index_select(0, prompt_positions)
        tail_ids = _trim_trailing_pad(seq_row[prompt_len:], pad_token_id)
        if tail_ids.numel() > 0:
            tail_positions = torch.arange(
                prompt_len,
                prompt_len + int(tail_ids.shape[0]),
                device=seq_row.device,
                dtype=torch.long,
            )
            ids = torch.cat([prompt_ids, tail_ids], dim=0)
            positions = torch.cat([prompt_positions, tail_positions], dim=0)
            return ids, positions
        return prompt_ids, prompt_positions

    ids = _trim_trailing_pad(seq_row, pad_token_id)
    if ids.numel() > 0:
        positions = torch.arange(
            int(ids.shape[0]), device=seq_row.device, dtype=torch.long
        )
        return ids, positions

    fallback_pos = torch.tensor(
        [seq_row.shape[0] - 1], device=seq_row.device, dtype=torch.long
    )
    fallback_ids = seq_row.index_select(0, fallback_pos)
    return fallback_ids, fallback_pos


def generate_full_sequence(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: DriftGuardConfig,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    gen_kwargs: dict = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        gen_kwargs["temperature"] = cfg.temperature
        gen_kwargs["top_p"] = cfg.top_p

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    if cfg.include_prompt_in_trajectory:
        return generated

    prompt_len = input_ids.shape[1]
    only_new = generated[:, prompt_len:]
    if only_new.shape[1] == 0:
        return generated[:, -1:]
    return only_new


def _decode_generated(
    seq_ids: torch.Tensor,
    prompt_len: int,
    include_prompt: bool,
    tokenizer: AutoTokenizer,
    pad_token_id: int | None,
) -> str:
    """Decode only the generated (non-prompt) portion of a sequence."""
    if include_prompt:
        gen_ids = seq_ids[prompt_len:]
    else:
        gen_ids = seq_ids
    gen_ids = _trim_trailing_pad(gen_ids, pad_token_id)
    return tokenizer.decode(gen_ids.cpu(), skip_special_tokens=True)


def _resolve_layer_indices(
    model: AutoModelForCausalLM,
    layer_indices: list[int] | None,
) -> list[int]:
    if layer_indices is not None:
        return layer_indices

    text_cfg = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    n_layers = text_cfg.num_hidden_layers
    return list(range(n_layers + 1))


def _extract_multi_layer_true_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_indices: list[int],
    cfg: DriftGuardConfig,
    device: str,
    show_progress: bool,
) -> ExtractionResult:
    max_length = cfg.max_input_tokens
    if cfg.inference_batch_size < 1:
        raise ValueError(
            f"inference_batch_size must be >= 1. Got {cfg.inference_batch_size}."
        )
    batch_size = int(cfg.inference_batch_size)

    per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    token_texts: list[list[str]] = []
    generated_texts: list[str | None] = []

    for start in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            if cfg.use_generate:
                seq_ids = generate_full_sequence(
                    model,
                    input_ids,
                    attention_mask,
                    cfg,
                    tokenizer,
                )
                seq_attn = torch.ones_like(seq_ids, device=device)
                out = model(
                    input_ids=seq_ids,
                    attention_mask=seq_attn,
                    output_hidden_states=True,
                )
                prompt_len = int(input_ids.shape[1])
                for i in range(len(batch_texts)):
                    ids, positions = _extract_ids_and_positions_generate(
                        seq_row=seq_ids[i],
                        prompt_mask_row=attention_mask[i].bool(),
                        prompt_len=prompt_len,
                        include_prompt=cfg.include_prompt_in_trajectory,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    token_texts.append(
                        tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
                    )
                    generated_texts.append(
                        _decode_generated(
                            seq_ids[i],
                            prompt_len,
                            cfg.include_prompt_in_trajectory,
                            tokenizer,
                            tokenizer.pad_token_id,
                        )
                    )
                    for li in layer_indices:
                        hs_row = out.hidden_states[li][i]
                        hs = hs_row.index_select(0, positions).float().cpu().numpy()
                        per_layer[li].append(hs)
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                for i in range(len(batch_texts)):
                    positions = torch.nonzero(
                        attention_mask[i].bool(), as_tuple=False
                    ).squeeze(-1)
                    ids = input_ids[i].index_select(0, positions)
                    token_texts.append(
                        tokenizer.convert_ids_to_tokens(ids.cpu().tolist())
                    )
                    generated_texts.append(None)
                    for li in layer_indices:
                        hs_row = out.hidden_states[li][i]
                        hs = hs_row.index_select(0, positions).float().cpu().numpy()
                        per_layer[li].append(hs)

    return ExtractionResult(
        per_layer=per_layer,
        token_texts=token_texts,
        input_prompts=list(texts),
        generated_texts=generated_texts,
    )


def _extract_multi_layer_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_indices: list[int],
    cfg: DriftGuardConfig,
    device: str,
    show_progress: bool,
) -> ExtractionResult:
    max_length = cfg.max_input_tokens
    per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    token_texts: list[list[str]] = []
    generated_texts: list[str | None] = []

    for text in tqdm(texts, disable=not show_progress):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            if cfg.use_generate:
                seq_ids = generate_full_sequence(
                    model,
                    input_ids,
                    attention_mask,
                    cfg,
                    tokenizer,
                )
                seq_attn = torch.ones_like(seq_ids, device=device)
                out = model(
                    input_ids=seq_ids,
                    attention_mask=seq_attn,
                    output_hidden_states=True,
                )
                prompt_len = int(input_ids.shape[1])
                generated_texts.append(
                    _decode_generated(
                        seq_ids[0],
                        prompt_len,
                        cfg.include_prompt_in_trajectory,
                        tokenizer,
                        tokenizer.pad_token_id,
                    )
                )
                ids = seq_ids[0].cpu().tolist()
                for li in layer_indices:
                    hs = out.hidden_states[li][0].float().cpu().numpy()
                    per_layer[li].append(hs)
            else:
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                attn = attention_mask[0].bool()
                ids = input_ids[0][attn].cpu().tolist()
                generated_texts.append(None)
                for li in layer_indices:
                    hs = out.hidden_states[li][0][attn].float().cpu().numpy()
                    per_layer[li].append(hs)

        token_texts.append(tokenizer.convert_ids_to_tokens(ids))

    return ExtractionResult(
        per_layer=per_layer,
        token_texts=token_texts,
        input_prompts=list(texts),
        generated_texts=generated_texts,
    )


def extract_multi_layer_trajectories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_indices: list[int] | None,
    cfg: DriftGuardConfig,
    show_progress: bool = True,
) -> ExtractionResult:
    """Collect trajectories for requested layers, with optional true batched inference.

    Args:
        layer_indices: Specific layer indices to extract. Pass ``None`` to
            extract all layers (embedding + every transformer layer).
    """
    device = resolve_device(cfg.device)
    resolved_layers = _resolve_layer_indices(model, layer_indices)

    if cfg.use_true_batch_inference and texts:
        return _extract_multi_layer_true_batch(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layer_indices=resolved_layers,
            cfg=cfg,
            device=device,
            show_progress=show_progress,
        )

    return _extract_multi_layer_single(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_indices=resolved_layers,
        cfg=cfg,
        device=device,
        show_progress=show_progress,
    )


def extract_hidden_trajectories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_idx: int,
    cfg: DriftGuardConfig,
) -> ExtractionResult:
    """Convenience wrapper for a single layer."""
    return extract_multi_layer_trajectories(
        model,
        tokenizer,
        texts,
        [layer_idx],
        cfg,
    )


def pool_trajectory(traj: np.ndarray, mode: str = "last") -> np.ndarray:
    if mode == "last":
        return traj[-1]
    if mode == "mean":
        return traj.mean(axis=0)
    if mode == "max_norm":
        return traj[np.linalg.norm(traj, axis=1).argmax()]
    raise ValueError(f"Unknown pooling mode: {mode}")


def build_feature_matrix(
    trajectories: list[np.ndarray],
    pooling: str,
) -> np.ndarray:
    return np.stack([pool_trajectory(t, pooling) for t in trajectories], axis=0)


def collect_trajectories_nnsight(
    model_path: str,
    prompts: list[str],
    layer_idx: int,
    device_map: str = "auto",
) -> list[np.ndarray]:
    """
    Collect hidden trajectories with nnsight in one traced forward per prompt.

    This is an optional high-fidelity path used when nnsight is available.
    """
    try:
        from latent_dynamics.online_runtime import load_nnsight_model
    except Exception as exc:
        raise ImportError(
            "nnsight runtime helpers are unavailable; cannot run nnsight extraction."
        ) from exc

    nns_model = load_nnsight_model(
        model_path=model_path,
        device_map=device_map,
    )

    def _layer_stack(model_obj: object) -> object:
        roots = [getattr(model_obj, "model", None), model_obj]
        for root in roots:
            if root is None:
                continue
            if hasattr(root, "layers"):
                return root.layers
            if hasattr(root, "model") and hasattr(root.model, "layers"):
                return root.model.layers
            if hasattr(root, "language_model") and hasattr(root.language_model, "layers"):
                return root.language_model.layers
            if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
                return root.transformer.h
            if hasattr(root, "gpt_neox") and hasattr(root.gpt_neox, "layers"):
                return root.gpt_neox.layers
        raise AttributeError("Could not locate nnsight layer stack on wrapped model.")

    layers = _layer_stack(nns_model)
    trajectories: list[np.ndarray] = []
    for prompt in prompts:
        with nns_model.trace(prompt):
            # Most decoder-only models expose transformer layers here.
            hidden_proxy = layers[layer_idx].output[0].save()
        hidden = hidden_proxy.value if hasattr(hidden_proxy, "value") else hidden_proxy
        if isinstance(hidden, torch.Tensor):
            hidden_arr = hidden.squeeze(0).float().cpu().numpy()
        else:
            hidden_arr = np.asarray(hidden).squeeze(0)
        trajectories.append(hidden_arr)
    return trajectories
