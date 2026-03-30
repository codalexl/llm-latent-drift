from __future__ import annotations

from dataclasses import dataclass
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA

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
    tda_backend: str = "none"
    tda_approximate: bool = False


@dataclass
class RiskComponents:
    continuity: float
    lipschitz: float
    topology: float
    topology_diameter: float
    topology_persistence_l1: float
    topology_beta0: float
    topology_beta1: float


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


class ManifoldReducer(Protocol):
    def transform(self, points: np.ndarray) -> np.ndarray: ...


class PreFitUMAP:
    """Inference-only UMAP reducer fit once on calibration trajectories."""

    def __init__(self, safe_trajectories: np.ndarray, cfg: DriftGuardConfig, n_components: int) -> None:
        try:
            import umap  # type: ignore[import]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("UMAP is unavailable.") from exc
        n_points = max(int(safe_trajectories.shape[0]), 2)
        n_neighbors = min(15, max(2, int(cfg.topology_window) // 2), n_points - 1)
        self.reducer = umap.UMAP(
            n_components=max(1, min(int(n_components), safe_trajectories.shape[1])),
            n_neighbors=n_neighbors,
            min_dist=min(0.5, max(0.01, 0.1 * (8.0 / max(float(cfg.topology_window), 1.0)))),
            random_state=int(cfg.random_seed or 0),
            n_jobs=1,
        )
        self.reducer.fit(np.asarray(safe_trajectories, dtype=np.float32))

    def transform(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(self.reducer.transform(points), dtype=np.float32)


class FastPCA:
    """Stateful PCA for inference-time transforms."""

    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        self._pca: PCA | None = None

    def fit(self, points: np.ndarray) -> None:
        if points.shape[0] == 0:
            self._pca = None
            return
        k = max(1, min(int(self.n_components), points.shape[0], points.shape[1]))
        self._pca = PCA(n_components=k, svd_solver="full")
        try:
            self._pca.fit(points)
        except Exception:
            self._pca = None

    def transform(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points
        if self._pca is None:
            self.fit(points)
        if self._pca is None:
            return pca_reduce(points, n_components=self.n_components)
        return np.asarray(self._pca.transform(points), dtype=np.float32)


_REDUCER_CACHE: dict[str, ManifoldReducer] = {}


def _safe_trajectory_candidates(cfg: DriftGuardConfig) -> list[Path]:
    extra = getattr(cfg, "safe_trajectories_path", None)
    out: list[Path] = []
    if isinstance(extra, str) and extra.strip():
        out.append(Path(extra))
    out.extend(
        [
            Path("calibration_safe_trajectories.npy"),
            Path("artifacts/calibration_safe_trajectories.npy"),
        ]
    )
    return out


def _load_safe_trajectories(cfg: DriftGuardConfig) -> np.ndarray | None:
    for candidate in _safe_trajectory_candidates(cfg):
        if candidate.exists():
            arr = np.load(candidate)
            if arr.ndim == 2 and arr.shape[0] > 1:
                return np.asarray(arr, dtype=np.float32)
    return None


def _reducer_key(cfg: DriftGuardConfig, n_components: int, input_dim: int) -> str:
    return f"{cfg.reduction_method}:{n_components}:{input_dim}:{cfg.topology_window}:{cfg.random_seed}"


def _reduce_points(points: np.ndarray, cfg: DriftGuardConfig, n_components: int) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    method = str(cfg.reduction_method).lower()
    if method == "none":
        return points
    cache_key = _reducer_key(cfg, n_components, points.shape[1])
    if method == "umap":
        reducer = _REDUCER_CACHE.get(cache_key)
        if reducer is None:
            safe = _load_safe_trajectories(cfg)
            if safe is None:
                warnings.warn(
                    "UMAP requested without pre-fit safe trajectories; falling back to PCA.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return pca_reduce(points, n_components=n_components)
            try:
                reducer = PreFitUMAP(safe_trajectories=safe, cfg=cfg, n_components=n_components)
            except Exception:
                warnings.warn(
                    "UMAP pre-fit failed; falling back to PCA.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return pca_reduce(points, n_components=n_components)
            _REDUCER_CACHE[cache_key] = reducer
        return reducer.transform(np.asarray(points, dtype=np.float32))
    if method == "pca":
        if points.shape[0] < 2:
            return pca_reduce(points, n_components=n_components)
        reducer = _REDUCER_CACHE.get(cache_key)
        if reducer is None:
            reducer = FastPCA(n_components=n_components)
            _REDUCER_CACHE[cache_key] = reducer
        return reducer.transform(np.asarray(points, dtype=np.float32))
    if method != "pca":
        warnings.warn(
            f"Unknown reduction method '{method}'; falling back to PCA.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pca_reduce(points, n_components=n_components)
    return pca_reduce(points, n_components=n_components)


def _subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> tuple[np.ndarray, bool]:
    if points.shape[0] <= max_points:
        return points, False
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(idx)], True


def _ripser_diagrams(points: np.ndarray, maxdim: int) -> tuple[list[np.ndarray] | None, str]:
    """
    Preferred PH backend.

    Library notes:
    - Exact dependency: ripser.py package (`ripser`) when available.
    - Optional fallback: GUDHI (`gudhi`) if ripser import/runtime fails.
    """
    try:
        from ripser import ripser  # type: ignore[import]
    except Exception as exc:
        warnings.warn(f"ripser import failed ({exc!r}); trying gudhi fallback.", RuntimeWarning, stacklevel=2)
        return _gudhi_diagrams(points, maxdim=maxdim)
    try:
        out = ripser(points, maxdim=maxdim)
    except Exception as exc:
        warnings.warn(f"ripser runtime failed ({exc!r}); trying gudhi fallback.", RuntimeWarning, stacklevel=2)
        return _gudhi_diagrams(points, maxdim=maxdim)
    diagrams = out.get("dgms")
    if not isinstance(diagrams, list):
        warnings.warn(
            "ripser returned unexpected diagram format; trying gudhi fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _gudhi_diagrams(points, maxdim=maxdim)
    return diagrams, "ripser"


def _gudhi_diagrams(points: np.ndarray, maxdim: int) -> tuple[list[np.ndarray] | None, str]:
    try:
        import gudhi as gd  # type: ignore[import]
    except Exception:
        return None, "none"
    try:
        rips = gd.RipsComplex(points=points)
        simplex_tree = rips.create_simplex_tree(max_dimension=maxdim + 1)
        diag = simplex_tree.persistence()
        by_dim: list[list[list[float]]] = [[] for _ in range(maxdim + 1)]
        for dim, pair in diag:
            if 0 <= dim <= maxdim:
                by_dim[dim].append([float(pair[0]), float(pair[1])])
        return [np.asarray(rows, dtype=np.float32) for rows in by_dim], "gudhi"
    except Exception:
        return None, "none"


def persistence_summary(
    points: np.ndarray,
    maxdim: int = 1,
    *,
    approximate_max_points: int = 32,
    subsample_seed: int = 0,
) -> tuple[int, int, float, str, bool]:
    """Compute lightweight Betti proxies and persistence mass."""
    tda_points, approximate = _subsample_points(points, max_points=max(2, int(approximate_max_points)), seed=subsample_seed)
    diagrams, backend = _ripser_diagrams(tda_points, maxdim=maxdim)
    if diagrams is None:
        return 0, 0, 0.0, "none", approximate

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
    return beta0, beta1, persistence_l1, backend, approximate


def _normalize_cloud_median_dist(points: np.ndarray) -> np.ndarray:
    """Scale point cloud so median pairwise distance is 1.0.

    This standard TDA preprocessing preserves relative geometry (shape,
    holes, cluster structure) while making diameter and persistence
    comparable across models and layers with different hidden-state scales.
    """
    if points.shape[0] < 2:
        return points
    if pdist is not None:
        dists = pdist(points)
    else:
        diffs = points[:, None, :] - points[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)[np.triu_indices(points.shape[0], k=1)]
    med = float(np.median(dists))
    if med < 1e-12:
        return points
    return points / med


def topology_snapshot(
    points: np.ndarray,
    config: DriftGuardConfig | None = None,
    *,
    pca_components: int | None = None,
    tda_enabled: bool | None = None,
) -> TopologySnapshot:
    """Compute a compact TDA proxy over small trajectory windows.

    Notes:
    - Online windows are intentionally small; resulting Betti/persistence signals
      are heuristic and used only as an auxiliary risk term.
    - PCA/UMAP projection in this path is a pragmatic variance proxy and should
      not be interpreted as a strict guarantee of manifold topology.
    - `cfg.reduction_method` controls pre-TDA reduction (`pca`, `umap`, `none`),
      with `pca` as the default latency-friendly mode.
    """
    cfg = config or DriftGuardConfig()
    n_components = cfg.pca_components if pca_components is None else int(pca_components)
    do_tda = cfg.tda_enabled if tda_enabled is None else bool(tda_enabled)
    reduced = _normalize_cloud_median_dist(_reduce_points(points, cfg=cfg, n_components=n_components))
    if reduced.shape[0] == 0:
        return TopologySnapshot(
            diameter=0.0,
            beta0=0,
            beta1=0,
            persistence_l1=0.0,
            tda_enabled=False,
            tda_backend="none",
            tda_approximate=False,
        )
    if not do_tda:
        return TopologySnapshot(
            diameter=cloud_diameter(reduced),
            beta0=0,
            beta1=0,
            persistence_l1=0.0,
            tda_enabled=False,
            tda_backend="none",
            tda_approximate=False,
        )
    beta0, beta1, persistence_l1, backend, approximate = persistence_summary(
        reduced,
        maxdim=1,
        approximate_max_points=cfg.topology_window,
        subsample_seed=int(cfg.random_seed or 0),
    )
    return TopologySnapshot(
        diameter=cloud_diameter(reduced),
        beta0=beta0,
        beta1=beta1,
        persistence_l1=persistence_l1,
        tda_enabled=True,
        tda_backend=backend,
        tda_approximate=approximate,
    )


def decompose_risk_components(
    metrics: Mapping[str, float | int | None],
    config: DriftGuardConfig | None = None,
) -> RiskComponents:
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
        continuity_risk = continuity_risk / max(float(cfg.continuity_scale), 1e-6)

    lipschitz_risk = 0.0
    if lipschitz is not None:
        lipschitz_risk = max(0.0, float(lipschitz) - cfg.lipschitz_ceiling) / max(
            cfg.lipschitz_ceiling,
            1e-6,
        )
        lipschitz_risk = lipschitz_risk / max(float(cfg.lipschitz_scale), 1e-6)

    topo_diam_risk = 0.0
    if cloud_diam is not None:
        topo_diam_risk = cfg.topology_diameter_weight * (
            max(float(cloud_diam), 0.0) / max(cfg.topology_diameter_scale, 1e-6)
        )

    topo_p1_risk = 0.0
    if persistence_l1 is not None:
        topo_p1_risk = cfg.topology_persistence_l1_weight * (
            max(float(persistence_l1), 0.0) / max(cfg.persistence_l1_scale, 1e-6)
        )

    topo_b0_risk = 0.0
    if beta0 is not None:
        topo_b0_risk = cfg.topology_beta0_weight * (
            max(float(beta0), 0.0) / max(cfg.beta0_scale, 1e-6)
        )

    topo_b1_risk = 0.0
    if beta1 is not None:
        topo_b1_risk = cfg.topology_beta1_weight * (
            max(float(beta1), 0.0) / max(cfg.beta1_scale, 1e-6)
        )

    topology_risk = min(topo_diam_risk + topo_p1_risk + topo_b0_risk + topo_b1_risk, 2.0)
    return RiskComponents(
        continuity=continuity_risk,
        lipschitz=lipschitz_risk,
        topology=topology_risk,
        topology_diameter=topo_diam_risk,
        topology_persistence_l1=topo_p1_risk,
        topology_beta0=topo_b0_risk,
        topology_beta1=topo_b1_risk,
    )


def compute_risk_score(
    metrics: Mapping[str, float | int | None],
    config: DriftGuardConfig | None = None,
) -> float:
    """Compute fused drift risk from continuity, smoothness, and topology terms."""
    cfg = config or DriftGuardConfig()
    parts = decompose_risk_components(metrics, config=cfg)
    score = (
        cfg.continuity_weight * parts.continuity
        + cfg.lipschitz_weight * parts.lipschitz
        + cfg.topology_weight * parts.topology
    )
    return float(np.clip(score, 0.0, 1.5))
