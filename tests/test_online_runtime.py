import pytest
import torch
import numpy as np

from latent_dynamics.online_runtime import (
    DriftGuardConfig,
    _GpuHiddenRingBuffer,
    _TdaState,
    _compute_step_metrics,
    _resolve_nnsight_layer_stack,
    _tda_within_budget,
)
from latent_dynamics.tda_metrics import TopologySnapshot


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
    assert cfg.pca_components == 8  # config default (CLI run-driftguard-session uses 3)


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
    assert _tda_within_budget(5.0, 2.5)
    assert not _tda_within_budget(5.0, 6.0)
    # First eligible step (no estimate yet) should always be allowed.
    assert _tda_within_budget(5.0, None)


# ---------------------------------------------------------------------------
# force_tda tests
# ---------------------------------------------------------------------------

def _make_full_buffer(window: int = 16, hidden_dim: int = 8) -> _GpuHiddenRingBuffer:
    """Return a ring buffer pre-filled with random vectors so has_window is True."""
    buf = _GpuHiddenRingBuffer.create(
        window=window,
        hidden_dim=hidden_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    for _ in range(window):
        buf.append(torch.randn(hidden_dim))
    return buf


def test_force_tda_default_is_false() -> None:
    cfg = DriftGuardConfig()
    assert cfg.force_tda is False


def test_force_tda_field_set_true() -> None:
    cfg = DriftGuardConfig(force_tda=True)
    assert cfg.force_tda is True


def test_force_tda_bypasses_budget_gate() -> None:
    """With force_tda=True, TDA executes even when estimated cost exceeds budget."""
    hidden_dim = 8
    window = 16
    buf = _make_full_buffer(window=window, hidden_dim=hidden_dim)
    hidden = torch.randn(hidden_dim)

    # Simulate budget already exhausted: estimated_ms >> budget.
    state_normal = _TdaState(estimated_ms=9999.0)
    state_forced = _TdaState(estimated_ms=9999.0)

    cfg_normal = DriftGuardConfig(
        tda_enabled=True,
        force_tda=False,
        topology_window=window,
        tda_latency_budget_ms=1.0,  # 1 ms — far below the 9999 ms estimate
        topology_stride=1,
        pca_components=4,
    )
    cfg_forced = cfg_normal.model_copy(update={"force_tda": True})

    # step_idx=0 satisfies stride (0 % 1 == 0), but budget is blown.
    _, _ = _compute_step_metrics(hidden, None, buf, None, 0, cfg_normal, state_normal, None)
    _, _ = _compute_step_metrics(hidden, None, buf, None, 0, cfg_forced, state_forced, None)

    # Normal: budget gate should have blocked TDA.
    assert state_normal.executed == 0
    assert state_normal.skipped_budget == 1

    # Forced: TDA must have executed despite budget.
    assert state_forced.executed == 1
    assert state_forced.skipped_budget == 0


def test_force_tda_bypasses_stride_gate() -> None:
    """With force_tda=True, TDA executes on non-stride steps too."""
    hidden_dim = 8
    window = 16
    buf = _make_full_buffer(window=window, hidden_dim=hidden_dim)
    hidden = torch.randn(hidden_dim)
    # A real TopologySnapshot as the cached value — non-None triggers the stride check.
    prev_topo = TopologySnapshot(
        diameter=0.5, beta0=1, beta1=0, persistence_l1=0.1,
        tda_enabled=True, tda_backend="ripser", tda_approximate=False,
    )

    state_normal = _TdaState()
    state_forced = _TdaState()

    cfg_normal = DriftGuardConfig(
        tda_enabled=True,
        force_tda=False,
        topology_window=window,
        tda_latency_budget_ms=9999.0,
        topology_stride=10,  # stride=10, so step_idx=1 misses stride
        pca_components=4,
    )
    cfg_forced = cfg_normal.model_copy(update={"force_tda": True})

    # step_idx=1 with stride=10: 1 % 10 != 0, so stride gate fires for normal config.
    _, _ = _compute_step_metrics(hidden, None, buf, prev_topo, 1, cfg_normal, state_normal, None)
    _, _ = _compute_step_metrics(hidden, None, buf, prev_topo, 1, cfg_forced, state_forced, None)

    assert state_normal.executed == 0
    assert state_normal.skipped_stride == 1

    assert state_forced.executed == 1
    assert state_forced.skipped_stride == 0


def test_force_tda_skipped_counters_stay_zero() -> None:
    """With force_tda=True, skipped_budget and skipped_stride must remain 0."""
    hidden_dim = 8
    window = 16
    buf = _make_full_buffer(window=window, hidden_dim=hidden_dim)
    hidden = torch.randn(hidden_dim)
    state = _TdaState(estimated_ms=9999.0)

    cfg = DriftGuardConfig(
        tda_enabled=True,
        force_tda=True,
        topology_window=window,
        tda_latency_budget_ms=1.0,
        topology_stride=100,
        pca_components=4,
    )
    for step in range(5):
        _, _ = _compute_step_metrics(hidden, None, buf, None, step, cfg, state, None)

    assert state.skipped_budget == 0
    assert state.skipped_stride == 0
    assert state.executed == 5
