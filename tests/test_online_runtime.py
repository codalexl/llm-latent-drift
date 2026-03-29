import pytest
import torch

from latent_dynamics.online_runtime import (
    DriftGuardConfig,
    _resolve_nnsight_layer_stack,
    _tda_within_budget,
)


class _LayersRoot:
    def __init__(self) -> None:
        self.layers = [object(), object()]


class _ModelWithNestedModel:
    def __init__(self) -> None:
        self.model = _LayersRoot()


def test_resolve_nnsight_layer_stack_nested_model() -> None:
    wrapped = _ModelWithNestedModel()
    layers = _resolve_nnsight_layer_stack(wrapped)
    assert isinstance(layers, list)
    assert len(layers) == 2


def test_runtime_config_exposes_new_tda_controls() -> None:
    cfg = DriftGuardConfig()
    assert cfg.tda_enabled is True
    assert cfg.pca_components == 8


def test_nnsight_guard_zero_delta_raises_when_fail_closed() -> None:
    cfg = DriftGuardConfig(nnsight_fail_open=False)
    assert cfg.nnsight_fail_open is False

    logits_last = torch.zeros((1, 10), dtype=torch.float32)
    logits_for_sample = torch.zeros((1, 10), dtype=torch.float32)
    delta_norm = float(torch.norm(logits_for_sample - logits_last, p=2).item())
    with pytest.raises(RuntimeError):
        if delta_norm <= 1e-8:
            raise RuntimeError("nnsight steering produced zero logit delta.")


def test_tda_budget_gate_accounts_for_estimated_cost() -> None:
    assert _tda_within_budget(2.0, 5.0, 2.5)
    assert not _tda_within_budget(3.0, 5.0, 2.5)
