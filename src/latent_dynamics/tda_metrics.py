from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopologySnapshot:
    """Compact topological summary for a trajectory window."""

    diameter: float
    beta0: int | None
    beta1: int | None
    persistence_l1: float | None


def cloud_diameter(points: np.ndarray) -> float:
    """Return max pairwise distance for a point cloud."""
    if points.ndim != 2:
        raise ValueError("points must have shape (n_points, n_dims).")
    if points.shape[0] < 2:
        return 0.0
    # O(n^2) is acceptable for small online windows.
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
    except Exception:
        return None
    out = ripser(points, maxdim=maxdim)
    diagrams = out.get("dgms")
    if not isinstance(diagrams, list):
        return None
    return diagrams


def persistence_summary(points: np.ndarray, maxdim: int = 1) -> tuple[int | None, int | None, float | None]:
    """Compute lightweight Betti proxies and persistence mass."""
    diagrams = _ripser_diagrams(points, maxdim=maxdim)
    if diagrams is None:
        return None, None, None

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


def topology_snapshot(points: np.ndarray, pca_components: int = 8) -> TopologySnapshot:
    reduced = pca_reduce(points, n_components=pca_components)
    beta0, beta1, persistence_l1 = persistence_summary(reduced, maxdim=1)
    return TopologySnapshot(
        diameter=cloud_diameter(reduced),
        beta0=beta0,
        beta1=beta1,
        persistence_l1=persistence_l1,
    )
