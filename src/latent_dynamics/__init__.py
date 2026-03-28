"""Drift-focused latent trajectory analysis utilities."""

from latent_dynamics.activations import (
    ExtractionResult,
    build_feature_matrix,
    collect_trajectories_nnsight,
    extract_hidden_trajectories,
    extract_multi_layer_trajectories,
    pool_trajectory,
)
from latent_dynamics.config import DATASET_REGISTRY, MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.drift import compute_drift_metrics, evaluate_generator_shift, first_exit_time
from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub, push_to_hub, save_activations
from latent_dynamics.models import load_model_and_tokenizer, resolve_device
from latent_dynamics.online_runtime import (
    DriftGuardConfig,
    estimate_safe_reference,
    load_nnsight_model,
    run_driftguard_session,
    run_driftguard_session_nnsight,
)
from latent_dynamics.steering import steer_toward_reference, steer_with_nnsight
from latent_dynamics.tda_metrics import TopologySnapshot, topology_snapshot

__all__ = [
    "RunConfig",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "ExtractionResult",
    "DriftGuardConfig",
    "TopologySnapshot",
    "activation_subpath",
    "build_feature_matrix",
    "collect_trajectories_nnsight",
    "compute_drift_metrics",
    "estimate_safe_reference",
    "evaluate_generator_shift",
    "extract_hidden_trajectories",
    "extract_multi_layer_trajectories",
    "first_exit_time",
    "load_activations",
    "load_examples",
    "load_model_and_tokenizer",
    "pool_trajectory",
    "prepare_text_and_labels",
    "pull_from_hub",
    "push_to_hub",
    "resolve_device",
    "load_nnsight_model",
    "run_driftguard_session",
    "run_driftguard_session_nnsight",
    "save_activations",
    "steer_toward_reference",
    "steer_with_nnsight",
    "topology_snapshot",
]
