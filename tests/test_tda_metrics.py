import numpy as np
import pytest
from hypothesis import given, strategies as st

from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.tda_metrics import compute_risk_score, topology_snapshot

pytestmark = [
    pytest.mark.filterwarnings("ignore:The input matrix is square.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*matmul.*:RuntimeWarning"),
]


def test_topology_snapshot_empty_window(config: DriftGuardConfig) -> None:
    window = np.array([], dtype=np.float32).reshape(0, 64)
    result = topology_snapshot(window, config=config)
    assert result.beta0 == 0
    assert result.beta1 == 0
    assert result.diameter == 0.0
    assert result.persistence_l1 == 0.0


def test_topology_snapshot_single_point(config: DriftGuardConfig) -> None:
    window = np.random.randn(1, 64).astype(np.float32)
    result = topology_snapshot(window, config=config, tda_enabled=False)
    assert result.beta1 >= 0
    assert result.diameter == 0.0


@given(st.integers(min_value=2, max_value=24))
def test_cloud_diameter_nonnegative(n_points: int) -> None:
    pts = np.random.randn(n_points, 8).astype(np.float32)
    result = topology_snapshot(
        pts,
        config=DriftGuardConfig(pca_components=8, tda_enabled=False),
    )
    assert result.diameter >= 0.0


def test_persistence_l1_integrated(config: DriftGuardConfig) -> None:
    window = np.random.randn(12, 64).astype(np.float32)
    result = topology_snapshot(window, config=config)
    assert result.persistence_l1 >= 0.0
    risk = compute_risk_score(
        {
            "cosine_cont": 0.95,
            "lipschitz": 0.12,
            "cloud_diameter": result.diameter,
            "beta0": result.beta0,
            "beta1": result.beta1,
            "persistence_l1": result.persistence_l1,
        },
        config=config,
    )
    assert risk >= 0.0


def test_risk_score_normalization(config: DriftGuardConfig) -> None:
    safe_metrics = {
        "cosine_cont": 0.98,
        "lipschitz": 0.05,
        "cloud_diameter": 5.0,
        "beta0": 12,
        "beta1": 0,
        "persistence_l1": 0.2,
    }
    risk = compute_risk_score(safe_metrics, config=config)
    assert 0.0 <= risk <= 0.3
