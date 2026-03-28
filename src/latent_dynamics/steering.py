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
