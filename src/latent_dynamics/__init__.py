"""Drift-focused latent trajectory analysis utilities."""

from latent_dynamics.activations import (
    ExtractionResult,
    build_feature_matrix,
    extract_hidden_trajectories,
    extract_multi_layer_trajectories,
    pool_trajectory,
)
from latent_dynamics.config import DATASET_REGISTRY, MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.drift import compute_drift_metrics, evaluate_generator_shift, first_exit_time
from latent_dynamics.hub import activation_subpath, load_activations, pull_from_hub, push_to_hub, save_activations
from latent_dynamics.models import load_model_and_tokenizer, resolve_device
from latent_dynamics.online_runtime import DriftGuardConfig, estimate_safe_reference, run_driftguard_session
from latent_dynamics.steering import steer_toward_reference
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
    "run_driftguard_session",
    "save_activations",
    "steer_toward_reference",
    "topology_snapshot",
]
