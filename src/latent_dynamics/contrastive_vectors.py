from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from latent_dynamics.config import DriftGuardConfig


def _infer_device(model: object, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    try:
        params = getattr(model, "parameters", None)
        if callable(params):
            first = next(params())
            return str(first.device)
    except Exception:
        pass
    return "cpu"


def _collect_last_hidden_states(
    model: object,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    *,
    layer_idx: int,
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    if not prompts:
        raise ValueError("prompts must be non-empty.")

    states: list[torch.Tensor] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = out.hidden_states[layer_idx][:, -1, :].detach().cpu()
        states.append(hidden.squeeze(0))
    return torch.stack(states, dim=0)


def compute_contrastive_vector(
    model: object,
    tokenizer: AutoTokenizer,
    safe_prompts: list[str],
    harmful_prompts: list[str],
    layer_idx: int,
    cfg: DriftGuardConfig,
    *,
    device: str | None = None,
) -> np.ndarray:
    """
    Compute mean(safe last-token state) - mean(harmful last-token state).

    The returned vector is L2-normalized for stable projection/steering.
    """
    if not safe_prompts:
        raise ValueError("safe_prompts must be non-empty.")
    if not harmful_prompts:
        raise ValueError("harmful_prompts must be non-empty.")

    active_device = _infer_device(model, fallback=device or cfg.device)
    safe_states = _collect_last_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=safe_prompts,
        layer_idx=layer_idx,
        device=active_device,
        max_length=cfg.max_input_tokens,
    )
    harmful_states = _collect_last_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=harmful_prompts,
        layer_idx=layer_idx,
        device=active_device,
        max_length=cfg.max_input_tokens,
    )
    safe_centroid = safe_states.mean(dim=0)
    harmful_centroid = harmful_states.mean(dim=0)
    direction = (safe_centroid - harmful_centroid).to(torch.float32).cpu().numpy()
    denom = float(np.linalg.norm(direction)) + 1e-8
    return (direction / denom).astype(np.float32)
