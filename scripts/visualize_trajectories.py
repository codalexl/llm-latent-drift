#!/usr/bin/env python3
"""Publication-ready trajectory visualizations.

Run from repo root with project env (e.g. uv):

  uv run python scripts/visualize_trajectories.py --model qwen3_8b --activations activations/xstest/train/qwen3_8b/layer_24

Produces:
- UMAP of mean-pooled trajectories (safe/unsafe colored).
- Token-level single-trajectory paths (PCA → 2D/3D).
- Optional speed arrows overlaid on path plots.
- Histograms of speed ||z_{t+1}-z_t|| and curvature.
- Captions JSON at figures/captions.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA

# Ensure package is on path (src layout)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_dynamics.hub import load_activations  # type: ignore[import]
from latent_dynamics.tda_metrics import topology_snapshot  # type: ignore[import]

try:
    import umap  # type: ignore[import]

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import plotly.graph_objects as go


def _maybe_write_image(fig: go.Figure, path: Path, dpi: int = 300) -> None:
    """Write HTML and best-effort PNG/SVG for a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    try:
        fig.write_image(str(path.with_suffix(".png")), scale=max(1, dpi // 100))
    except Exception:
        pass
    try:
        fig.write_image(str(path.with_suffix(".svg")))
    except Exception:
        pass


def _sanitize_for_projection(values: np.ndarray, clip_value: float = 1e3) -> np.ndarray:
    """Keep projection numerics stable for noisy hidden-state tensors."""
    arr = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=clip_value, neginf=-clip_value)
    return np.clip(arr, -clip_value, clip_value)


def _project_2d(features: np.ndarray) -> np.ndarray:
    """Project to 2D with graceful fallback for tiny sample counts."""
    if features.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_dims).")
    n_samples, n_dims = features.shape
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if n_samples < 2 or n_dims < 2:
        out = np.zeros((n_samples, 2), dtype=np.float32)
        out[:, 0] = features[:, 0].astype(np.float32)
        return out
    pca = PCA(n_components=2)
    return pca.fit_transform(features).astype(np.float32)


def _mean_pool_trajectories(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([traj.mean(axis=0) for traj in trajectories], axis=0)


def _compute_speed_and_curvature(
    trajectories: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    speeds: list[float] = []
    curvatures: list[float] = []

    for traj in trajectories:
        if traj.shape[0] < 3:
            continue
        diffs = np.diff(traj, axis=0)
        step_speeds = np.linalg.norm(diffs, axis=1)
        speeds.extend(step_speeds.tolist())

        v1 = diffs[:-1]
        v2 = diffs[1:]
        num = (v1 * v2).sum(axis=1)
        denom = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
        cos_theta = np.clip(num / denom, -1.0, 1.0)
        angles = np.arccos(cos_theta)
        curvatures.extend(angles.tolist())

    return np.array(speeds, dtype=np.float32), np.array(curvatures, dtype=np.float32)


def _pick_indices(labels: np.ndarray, value: int, k: int) -> list[int]:
    idx = np.where(labels == value)[0].tolist()
    return idx[:k]


def _cosine_drift_matrix(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    max_steps = max((max(0, traj.shape[0] - 1) for traj in trajectories), default=1)
    mat = np.full((len(trajectories), max_steps), np.nan, dtype=np.float32)
    for i, traj in enumerate(trajectories):
        if traj.shape[0] < 2:
            continue
        prev = traj[:-1]
        nxt = traj[1:]
        denom = (np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)) + 1e-8
        cos = np.sum(prev * nxt, axis=1) / denom
        mat[i, : cos.shape[0]] = 1.0 - np.clip(cos, -1.0, 1.0)
    return mat


def _rolling_beta1_matrix(
    trajectories: Sequence[np.ndarray],
    window: int,
    stride: int,
) -> np.ndarray:
    counts = []
    for traj in trajectories:
        if traj.shape[0] < window:
            counts.append(0)
            continue
        counts.append(1 + max(0, (traj.shape[0] - window) // stride))
    max_windows = max(counts, default=1)
    mat = np.full((len(trajectories), max_windows), np.nan, dtype=np.float32)
    for i, traj in enumerate(trajectories):
        if traj.shape[0] < window:
            continue
        col = 0
        for start in range(0, traj.shape[0] - window + 1, stride):
            topo = topology_snapshot(traj[start : start + window])
            mat[i, col] = float(topo.beta1) if topo.beta1 is not None else np.nan
            col += 1
    return mat


def _plot_heatmap(
    matrix: np.ndarray,
    title: str,
    x_label: str,
    color_label: str,
    out_path: Path,
    zmin: float | None = None,
    zmax: float | None = None,
) -> None:
    if matrix.size == 0:
        return
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                colorscale="Viridis",
                colorbar={"title": color_label},
                zmin=zmin,
                zmax=zmax,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Trajectory index",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def plot_pca_manifold(
    features: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    features = _sanitize_for_projection(features)
    emb = _project_2d(features)
    fig = go.Figure()
    for value, name, color in [(0, "safe", "seagreen"), (1, "unsafe", "crimson")]:
        mask = labels == value
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.8},
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    features = _sanitize_for_projection(features)
    if HAS_UMAP:
        try:
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
            )
            emb = reducer.fit_transform(features)
        except Exception:
            emb = _project_2d(features)
    else:
        # Fallback: PCA-only if UMAP is unavailable.
        emb = _project_2d(features)

    fig = go.Figure()
    for value, name, color in [(0, "safe", "seagreen"), (1, "unsafe", "crimson")]:
        mask = labels == value
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.8},
                name=name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def plot_trajectory_paths(
    trajectories: Sequence[np.ndarray],
    labels: np.ndarray,
    title: str,
    out_path: Path,
    max_per_class: int = 3,
    path_dims: int = 2,
    show_arrows: bool = False,
    arrow_step: int = 1,
) -> None:
    if path_dims not in (2, 3):
        raise ValueError(f"path_dims must be 2 or 3, got {path_dims}")
    if arrow_step < 1:
        raise ValueError(f"arrow_step must be >= 1, got {arrow_step}")

    # Fit PCA on all token vectors.
    all_tokens = _sanitize_for_projection(np.concatenate(trajectories, axis=0))
    n_comp = min(path_dims, all_tokens.shape[0], all_tokens.shape[1])
    if n_comp < 1:
        return
    pca = PCA(n_components=n_comp)
    pca.fit(all_tokens)

    safe_idx = _pick_indices(labels, 0, max_per_class)
    unsafe_idx = _pick_indices(labels, 1, max_per_class)

    fig = go.Figure()

    def _add_arrows_2d(traj_proj: np.ndarray, color: str) -> None:
        if traj_proj.shape[0] < 2:
            return
        starts = traj_proj[:-1:arrow_step]
        deltas = np.diff(traj_proj, axis=0)[::arrow_step]
        if starts.size == 0:
            return
        scale = 0.35
        ends = starts + (scale * deltas)
        x_segments: list[float | None] = []
        y_segments: list[float | None] = []
        for s, e in zip(starts, ends):
            x_segments.extend([float(s[0]), float(e[0]), None])
            y_segments.extend([float(s[1]), float(e[1]), None])
        fig.add_trace(
            go.Scatter(
                x=x_segments,
                y=y_segments,
                mode="lines",
                line={"color": color, "width": 1},
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    def _add_arrows_3d(traj_proj: np.ndarray, color: str) -> None:
        if traj_proj.shape[0] < 2:
            return
        starts = traj_proj[:-1:arrow_step]
        deltas = np.diff(traj_proj, axis=0)[::arrow_step]
        if starts.size == 0:
            return
        norms = np.linalg.norm(deltas, axis=1)
        max_norm = float(np.max(norms)) if norms.size else 0.0
        sizeref = max(max_norm * 0.25, 0.05)
        fig.add_trace(
            go.Cone(
                x=starts[:, 0],
                y=starts[:, 1],
                z=starts[:, 2],
                u=deltas[:, 0],
                v=deltas[:, 1],
                w=deltas[:, 2],
                anchor="tail",
                showscale=False,
                colorscale=[[0.0, color], [1.0, color]],
                cmin=0.0,
                cmax=1.0,
                sizemode="absolute",
                sizeref=sizeref,
                opacity=0.45,
                hoverinfo="skip",
            )
        )

    for i in safe_idx:
        traj_proj = pca.transform(_sanitize_for_projection(trajectories[i]))
        if n_comp < path_dims:
            traj_proj = np.pad(
                traj_proj,
                ((0, 0), (0, path_dims - n_comp)),
                mode="constant",
            )
        if path_dims == 2:
            fig.add_trace(
                go.Scatter(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    mode="lines+markers",
                    line={"color": "seagreen"},
                    name=f"safe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_2d(traj_proj, "seagreen")
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    z=traj_proj[:, 2],
                    mode="lines+markers",
                    line={"color": "seagreen", "width": 4},
                    marker={"size": 3},
                    name=f"safe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_3d(traj_proj, "seagreen")

    for i in unsafe_idx:
        traj_proj = pca.transform(_sanitize_for_projection(trajectories[i]))
        if n_comp < path_dims:
            traj_proj = np.pad(
                traj_proj,
                ((0, 0), (0, path_dims - n_comp)),
                mode="constant",
            )
        if path_dims == 2:
            fig.add_trace(
                go.Scatter(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    mode="lines+markers",
                    line={"color": "crimson"},
                    name=f"unsafe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_2d(traj_proj, "crimson")
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_proj[:, 0],
                    y=traj_proj[:, 1],
                    z=traj_proj[:, 2],
                    mode="lines+markers",
                    line={"color": "crimson", "width": 4},
                    marker={"size": 3},
                    name=f"unsafe_{i}",
                    opacity=0.8,
                )
            )
            if show_arrows:
                _add_arrows_3d(traj_proj, "crimson")

    if path_dims == 2:
        fig.update_layout(
            title=title,
            xaxis_title="PCA dim 1",
            yaxis_title="PCA dim 2",
            template="plotly_white",
        )
    else:
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "PCA dim 1",
                "yaxis_title": "PCA dim 2",
                "zaxis_title": "PCA dim 3",
            },
            template="plotly_white",
        )
    _maybe_write_image(fig, out_path)


def plot_histogram(
    values: np.ndarray,
    title: str,
    x_label: str,
    out_path: Path,
    bins: int = 50,
) -> None:
    if values.size == 0:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values.tolist(),
            nbinsx=bins,
            marker={"color": "steelblue"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Count",
        template="plotly_white",
    )
    _maybe_write_image(fig, out_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize latent trajectories (UMAP, paths, dynamics histograms)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key (used only for naming).",
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations leaf directory (with metadata.json & trajectories.safetensors).",
    )
    parser.add_argument(
        "--fig_dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures and captions.json.",
    )
    parser.add_argument(
        "--path-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of PCA dimensions for token-path plot (2 or 3).",
    )
    parser.add_argument(
        "--show-arrows",
        action="store_true",
        help="Overlay per-step velocity arrows on trajectory path plots.",
    )
    parser.add_argument(
        "--arrow-step",
        type=int,
        default=1,
        help="Subsample factor for arrows (1 = every step, 2 = every other step).",
    )
    parser.add_argument(
        "--max-paths-per-class",
        type=int,
        default=3,
        help="Max number of safe and unsafe trajectories to draw in path plots.",
    )
    parser.add_argument(
        "--betti-window",
        type=int,
        default=24,
        help="Window size for rolling beta1 heatmap.",
    )
    parser.add_argument(
        "--betti-stride",
        type=int,
        default=4,
        help="Stride for rolling beta1 heatmap.",
    )
    args = parser.parse_args()
    if args.arrow_step < 1:
        parser.error("--arrow-step must be >= 1")
    if args.max_paths_per_class < 1:
        parser.error("--max-paths-per-class must be >= 1")
    if args.betti_window < 3:
        parser.error("--betti-window must be >= 3")
    if args.betti_stride < 1:
        parser.error("--betti-stride must be >= 1")

    trajectories, texts, labels, token_texts, _generated, cfg = load_activations(args.activations)
    labels_arr = labels if labels is not None else np.zeros(len(trajectories), dtype=np.int64)

    model_key = cfg.model_key
    layer = cfg.layer_idx
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[str, str] = {}

    # 1. UMAP of mean-pooled trajectories.
    pooled = _mean_pool_trajectories(trajectories)
    umap_path = fig_dir / f"umap_{model_key}_L{layer}"
    plot_umap(
        pooled,
        labels_arr,
        title=f"UMAP of mean-pooled trajectories — {model_key}, layer {layer}",
        out_path=umap_path,
    )
    captions[umap_path.with_suffix(".png").name] = (
        f"UMAP of mean-pooled hidden states for {model_key} layer {layer}, colored by safe (0) vs unsafe (1)."
    )

    # 1b. PCA manifold of mean-pooled trajectories.
    pca_manifold_path = fig_dir / f"pca_manifold_{model_key}_L{layer}"
    plot_pca_manifold(
        pooled,
        labels_arr,
        title=f"PCA manifold of mean-pooled trajectories — {model_key}, layer {layer}",
        out_path=pca_manifold_path,
    )
    captions[pca_manifold_path.with_suffix(".png").name] = (
        f"PCA manifold of mean-pooled hidden states for {model_key} layer {layer}, "
        "colored by safe (0) vs unsafe (1)."
    )

    # 2. Safe/unsafe token paths in PCA space (2D or 3D).
    path_name = "paths3d" if args.path_dims == 3 else "paths"
    paths_path = fig_dir / f"{path_name}_{model_key}_L{layer}"
    plot_trajectory_paths(
        trajectories,
        labels_arr,
        title=f"Example trajectories in PCA-{args.path_dims} space — {model_key}, layer {layer}",
        out_path=paths_path,
        max_per_class=args.max_paths_per_class,
        path_dims=args.path_dims,
        show_arrows=args.show_arrows,
        arrow_step=args.arrow_step,
    )
    arrow_text = " with per-step velocity arrows" if args.show_arrows else ""
    captions[paths_path.with_suffix(".png").name] = (
        f"Up to {args.max_paths_per_class} safe and {args.max_paths_per_class} unsafe token-level trajectories "
        f"projected into the first {args.path_dims} PCA components for {model_key} layer {layer}{arrow_text}."
    )

    # 3. Speed and curvature histograms.
    speeds, curvatures = _compute_speed_and_curvature(trajectories)
    speed_path = fig_dir / f"speed_hist_{model_key}_L{layer}"
    plot_histogram(
        speeds,
        title=f"Speed histogram — {model_key}, layer {layer}",
        x_label="||z_{t+1} - z_t||",
        out_path=speed_path,
    )
    captions[speed_path.with_suffix(".png").name] = (
        f"Histogram of per-token speed ||z_(t+1) - z_t|| for {model_key} layer {layer}."
    )

    curvature_path = fig_dir / f"curvature_hist_{model_key}_L{layer}"
    plot_histogram(
        curvatures,
        title=f"Curvature histogram — {model_key}, layer {layer}",
        x_label="Angle between successive steps (radians)",
        out_path=curvature_path,
    )
    captions[curvature_path.with_suffix(".png").name] = (
        f"Histogram of per-step curvature (angle between successive step vectors) for {model_key} layer {layer}."
    )

    # 4. Cosine-drift heatmap over token transitions.
    cosine_mat = _cosine_drift_matrix(trajectories)
    cosine_heatmap_path = fig_dir / f"cosine_drift_heatmap_{model_key}_L{layer}"
    _plot_heatmap(
        cosine_mat,
        title=f"Cosine drift heatmap — {model_key}, layer {layer}",
        x_label="Token transition index",
        color_label="1 - cosine",
        out_path=cosine_heatmap_path,
        zmin=0.0,
        zmax=1.0,
    )
    captions[cosine_heatmap_path.with_suffix(".png").name] = (
        f"Per-trajectory heatmap of token-to-token drift (1 - cosine) for {model_key} layer {layer}."
    )

    # 5. Rolling Betti-1 heatmap for topological warning signals.
    beta1_mat = _rolling_beta1_matrix(
        trajectories,
        window=args.betti_window,
        stride=args.betti_stride,
    )
    beta1_heatmap_path = fig_dir / f"betti1_heatmap_{model_key}_L{layer}"
    _plot_heatmap(
        beta1_mat,
        title=f"Rolling Betti-1 heatmap — {model_key}, layer {layer}",
        x_label="Window index",
        color_label="beta1",
        out_path=beta1_heatmap_path,
        zmin=0.0,
        zmax=float(np.nanmax(beta1_mat)) if np.isfinite(beta1_mat).any() else 1.0,
    )
    captions[beta1_heatmap_path.with_suffix(".png").name] = (
        f"Rolling-window Betti-1 heatmap (window={args.betti_window}, stride={args.betti_stride}) "
        f"for {model_key} layer {layer}."
    )

    # Note: PaCE support-set churn plots will be added once sparse codes are integrated.

    captions_path = fig_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2))
    print(f"Wrote captions to {captions_path}")


if __name__ == "__main__":
    main()

