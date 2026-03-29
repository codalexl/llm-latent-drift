import pytest

from latent_dynamics.config import Config


@pytest.fixture
def config() -> Config:
    return Config(
        pca_components=8,
        tda_enabled=True,
        tda_stride=4,
        cosine_floor=0.96,
        lipschitz_ceiling=0.20,
        risk_threshold=0.5,
    )
