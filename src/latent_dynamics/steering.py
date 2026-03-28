from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SteeringResult:
    applied: bool
    delta_norm: float


def steer_toward_reference(
    hidden: torch.Tensor,
    reference: torch.Tensor,
    strength: float = 0.25,
    max_delta_norm: float = 8.0,
) -> tuple[torch.Tensor, SteeringResult]:
    """
    Move hidden state toward a safe reference direction.

    Args:
        hidden: Tensor with shape (1, hidden_dim).
        reference: Tensor with shape (hidden_dim,) or (1, hidden_dim).
    """
    if hidden.ndim != 2 or hidden.shape[0] != 1:
        raise ValueError("hidden must have shape (1, hidden_dim).")

    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if reference.ndim != 2 or reference.shape != hidden.shape:
        raise ValueError("reference must match hidden shape.")

    alpha = float(np.clip(strength, 0.0, 1.0))
    if alpha <= 0.0:
        return hidden, SteeringResult(applied=False, delta_norm=0.0)

    target = (1.0 - alpha) * hidden + alpha * reference
    delta = target - hidden
    delta_norm = float(torch.norm(delta, p=2).item())
    if delta_norm > max_delta_norm > 0:
        scale = max_delta_norm / max(delta_norm, 1e-8)
        delta = delta * scale
        delta_norm = float(torch.norm(delta, p=2).item())
    steered = hidden + delta
    return steered, SteeringResult(applied=True, delta_norm=delta_norm)


def steer_with_nnsight(
    nns_model: object,
    prompt: str,
    safe_ref_hidden: torch.Tensor,
    layer_idx: int,
    alpha: float = 0.1,
) -> dict[str, object]:
    """
    Apply a causal activation intervention with nnsight.

    Returns a minimal artifact containing next-token logits and predicted token id.
    """
    if safe_ref_hidden.ndim == 1:
        safe_ref_hidden = safe_ref_hidden.unsqueeze(0)

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
    with nns_model.trace(prompt):
        layer_out = layers[layer_idx].output[0]
        delta = alpha * (safe_ref_hidden.to(layer_out.device) - layer_out[:, -1, :])
        layer_out[:, -1, :] = layer_out[:, -1, :] + delta
        logits_proxy = nns_model.lm_head.output.save()

    logits = logits_proxy.value if hasattr(logits_proxy, "value") else logits_proxy
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    return {
        "next_token_id": next_token,
        "logits": logits[:, -1, :].detach().cpu(),
    }
