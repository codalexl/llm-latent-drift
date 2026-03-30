from hypothesis import given, strategies as st

from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.tda_metrics import compute_risk_score


def test_high_risk_regime_scores_higher() -> None:
    cfg = DriftGuardConfig()
    safe = compute_risk_score(
        {
            "cosine_cont": 0.99,
            "lipschitz": 0.04,
            "cloud_diameter": 3.0,
            "beta0": 8,
            "beta1": 0,
            "persistence_l1": 0.1,
        },
        cfg,
    )
    unsafe = compute_risk_score(
        {
            "cosine_cont": 0.70,
            "lipschitz": 0.40,
            "cloud_diameter": 35.0,
            "beta0": 24,
            "beta1": 4,
            "persistence_l1": 4.0,
        },
        cfg,
    )
    assert unsafe > safe
    assert 0.0 <= safe <= 1.5
    assert 0.0 <= unsafe <= 1.5


def test_continuity_and_lipschitz_scales_reduce_component_dominance() -> None:
    metrics = {
        "cosine_cont": 0.20,
        "lipschitz": 1.20,
        "cloud_diameter": 0.0,
        "beta0": 0,
        "beta1": 0,
        "persistence_l1": 0.0,
    }
    base = compute_risk_score(metrics, DriftGuardConfig())
    scaled = compute_risk_score(
        metrics,
        DriftGuardConfig(continuity_scale=20.0, lipschitz_scale=40.0),
    )
    assert scaled < base


@given(st.floats(min_value=0.0, max_value=30.0))
def test_topology_monotonicity_for_diameter(diam: float) -> None:
    cfg = DriftGuardConfig()
    base = compute_risk_score(
        {
            "cosine_cont": 0.97,
            "lipschitz": 0.08,
            "cloud_diameter": diam,
            "beta0": 4,
            "beta1": 0,
            "persistence_l1": 0.2,
        },
        cfg,
    )
    higher = compute_risk_score(
        {
            "cosine_cont": 0.97,
            "lipschitz": 0.08,
            "cloud_diameter": diam + 1.0,
            "beta0": 4,
            "beta1": 0,
            "persistence_l1": 0.2,
        },
        cfg,
    )
    assert higher >= base
