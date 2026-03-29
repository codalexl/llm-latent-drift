from latent_dynamics.calibration import summarize_calibration_from_scores
from latent_dynamics.config import DriftGuardConfig


def test_calibration_reports_strong_auc_for_separable_scores() -> None:
    cfg = DriftGuardConfig()
    scores = [0.05, 0.12, 0.18, 0.75, 0.86, 0.93]
    labels = [0, 0, 0, 1, 1, 1]
    out = summarize_calibration_from_scores(scores=scores, labels=labels, cfg=cfg)
    assert out["roc_auc"] > 0.85
    assert out["pr_auc"] > 0.85
    assert 0.0 <= out["best_threshold"] <= 1.5


def test_calibration_output_has_weight_fields() -> None:
    cfg = DriftGuardConfig()
    out = summarize_calibration_from_scores(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        cfg=cfg,
    )
    assert set(out["weights"].keys()) == {"continuity", "lipschitz", "topology"}
