from __future__ import annotations

from dataclasses import dataclass
import warnings
from collections.abc import Mapping

import numpy as np

from latent_dynamics.config import DriftGuardConfig

try:
    from scipy.spatial.distance import pdist
except Exception:  # pragma: no cover - fallback for minimal envs
    pdist = None


@dataclass
class TopologySnapshot:
    """Compact topological summary for a trajectory window."""

    diameter: float
    beta0: int
    beta1: int
    persistence_l1: float
    tda_enabled: bool


def cloud_diameter(points: np.ndarray) -> float:
    """Return max pairwise distance for a point cloud."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, n_dims).")
    if points.shape[0] < 2:
        return 0.0
    if pdist is not None:
        # Still O(n^2), but fast for tiny online windows.
        return float(np.max(pdist(points)))
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(np.max(dists))


def pca_reduce(points: np.ndarray, n_components: int = 8) -> np.ndarray:
    """Project points with an SVD PCA implementation."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, n_dims).")
    if points.shape[0] == 0:
        return points
    k = max(1, min(int(n_components), points.shape[1], points.shape[0]))
    centered = points - points.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    return u[:, :k] * s[:k]


def _ripser_diagrams(points: np.ndarray, maxdim: int) -> list[np.ndarray] | None:
    try:
        from ripser import ripser  # type: ignore[import]
    except Exception as exc:
        warnings.warn(
            f"ripser import failed ({exc!r}); using geometry-only fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        out = ripser(points, maxdim=maxdim)
    except Exception as exc:
        warnings.warn(
            f"ripser runtime failed ({exc!r}); using geometry-only fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    diagrams = out.get("dgms")
    if not isinstance(diagrams, list):
        warnings.warn(
            "ripser returned unexpected diagram format; using geometry-only fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return diagrams


def persistence_summary(points: np.ndarray, maxdim: int = 1) -> tuple[int, int, float]:
    """Compute lightweight Betti proxies and persistence mass."""
    diagrams = _ripser_diagrams(points, maxdim=maxdim)
    if diagrams is None:
        return 0, 0, 0.0

    def finite_lifetimes(dgm: np.ndarray) -> np.ndarray:
        if dgm.size == 0:
            return np.array([], dtype=np.float32)
        life = dgm[:, 1] - dgm[:, 0]
        return life[np.isfinite(life)]

    dgm0 = diagrams[0] if len(diagrams) > 0 else np.array([])
    dgm1 = diagrams[1] if len(diagrams) > 1 else np.array([])
    life0 = finite_lifetimes(dgm0)
    life1 = finite_lifetimes(dgm1)

    # Threshold tiny numerical bars.
    beta0 = int(np.sum(life0 > 1e-6))
    beta1 = int(np.sum(life1 > 1e-6))
    persistence_l1 = float(np.sum(life0) + np.sum(life1))
    return beta0, beta1, persistence_l1


def topology_snapshot(
    points: np.ndarray,
    config: DriftGuardConfig | None = None,
    *,
    pca_components: int | None = None,
    tda_enabled: bool | None = None,
) -> TopologySnapshot:
    cfg = config or DriftGuardConfig()
    n_components = cfg.pca_components if pca_components is None else int(pca_components)
    do_tda = cfg.tda_enabled if tda_enabled is None else bool(tda_enabled)
    reduced = pca_reduce(points, n_components=n_components)
    if reduced.shape[0] == 0:
        return TopologySnapshot(
            diameter=0.0,
            beta0=0,
            beta1=0,
            persistence_l1=0.0,
            tda_enabled=False,
        )
    if not do_tda:
        return TopologySnapshot(
            diameter=cloud_diameter(reduced),
            beta0=0,
            beta1=0,
            persistence_l1=0.0,
            tda_enabled=False,
        )
    beta0, beta1, persistence_l1 = persistence_summary(reduced, maxdim=1)
    return TopologySnapshot(
        diameter=cloud_diameter(reduced),
        beta0=beta0,
        beta1=beta1,
        persistence_l1=persistence_l1,
        tda_enabled=True,
    )


def compute_risk_score(
    metrics: Mapping[str, float | int | None],
    config: DriftGuardConfig | None = None,
) -> float:
    """Compute fused drift risk from continuity, smoothness, and topology terms."""
    cfg = config or DriftGuardConfig()
    cosine = metrics.get("cosine_cont")
    lipschitz = metrics.get("lipschitz")
    cloud_diam = metrics.get("cloud_diameter")
    beta0 = metrics.get("beta0")
    beta1 = metrics.get("beta1")
    persistence_l1 = metrics.get("persistence_l1")

    continuity_risk = 0.0
    if cosine is not None:
        continuity_risk = max(0.0, cfg.cosine_floor - float(cosine)) / max(
            1.0 - cfg.cosine_floor,
            1e-6,
        )

    lipschitz_risk = 0.0
    if lipschitz is not None:
        lipschitz_risk = max(0.0, float(lipschitz) - cfg.lipschitz_ceiling) / max(
            cfg.lipschitz_ceiling,
            1e-6,
        )

    topology_risk = 0.0
    if cloud_diam is not None:
        topology_risk += 0.40 * (max(float(cloud_diam), 0.0) / max(cfg.topology_diameter_scale, 1e-6))
    if persistence_l1 is not None:
        topology_risk += 0.30 * (
            max(float(persistence_l1), 0.0) / max(cfg.persistence_l1_scale, 1e-6)
        )
    if beta0 is not None:
        topology_risk += 0.15 * (max(float(beta0), 0.0) / max(cfg.beta0_scale, 1e-6))
    if beta1 is not None:
        topology_risk += 0.15 * (max(float(beta1), 0.0) / max(cfg.beta1_scale, 1e-6))
    topology_risk = min(topology_risk, 2.0)

    score = (
        cfg.continuity_weight * continuity_risk
        + cfg.lipschitz_weight * lipschitz_risk
        + cfg.topology_weight * topology_risk
    )
    return float(np.clip(score, 0.0, 1.5))
