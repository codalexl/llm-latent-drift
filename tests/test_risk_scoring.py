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
