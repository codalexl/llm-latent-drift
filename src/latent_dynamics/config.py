from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class DriftGuardConfig(BaseModel):
    """Single source of truth for DriftGuard extraction and runtime settings."""

    model_config = ConfigDict(extra="allow")

    # Extraction / training-prep config.
    model_key: str = "gemma3_4b"
    dataset_key: str = "wildchat"
    max_samples: int = Field(default=120, ge=1)
    max_input_tokens: int = Field(default=512, ge=1)
    layer_idx: int = 5
    pooling: str = "last"  # last | mean | max_norm
    direction_method: str = "probe_weight"  # probe_weight | mean_diff | pca
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    calib_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    random_state: int = 42
    device: str | None = None

    # Generation defaults.
    use_generate: bool = False
    max_new_tokens: int = Field(default=24, ge=1)
    do_sample: bool = True
    temperature: float = Field(default=0.6, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.3, ge=1.0)
    monitor_warmup_steps: int = Field(default=3, ge=0)
    include_prompt_in_trajectory: bool = True
    use_true_batch_inference: bool = False
    inference_batch_size: int = Field(default=16, ge=1)

    # Drift runtime / risk config.
    pca_components: int = Field(default=8, ge=1, le=64)
    reduction_method: str = Field(default="pca", pattern="^(pca|umap|none)$")
    tda_enabled: bool = True
    force_tda: bool = Field(default=False, description="Bypass stride and budget gates; run TDA on every eligible step.")
    topology_stride: int = Field(
        default=2,
        ge=1,
        validation_alias=AliasChoices("topology_stride", "tda_stride"),
    )
    topology_window: int = Field(default=16, ge=4)
    tda_latency_budget_ms: float = Field(default=500.0, ge=0.0)
    cosine_floor: float = Field(default=0.85, ge=0.0, le=1.0)
    lipschitz_ceiling: float = Field(default=0.45, ge=0.0)
    risk_threshold: float = Field(default=0.95, ge=0.0, le=2.0)
    continuity_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    lipschitz_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    topology_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    continuity_scale: float = Field(default=1.0, gt=0.0)
    lipschitz_scale: float = Field(default=1.0, gt=0.0)
    topology_diameter_scale: float = Field(default=2.0, gt=0.0)
    persistence_l1_scale: float = Field(default=5.0, gt=0.0)
    beta0_scale: float = Field(default=8.0, gt=0.0)
    beta1_scale: float = Field(default=3.0, gt=0.0)
    topology_diameter_ceiling: float = Field(default=1.50, gt=0.0)
    topology_beta1_ceiling: float = Field(default=1.00, gt=0.0)
    # Topology sub-weights (must sum to 1.0).
    topology_diameter_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    topology_persistence_l1_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    topology_beta0_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    topology_beta1_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    probe_weight: float = Field(default=0.60, ge=0.0, le=1.0)
    contrastive_steering_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    use_contrastive_probe: bool = False
    safe_prompts: list[str] = Field(default_factory=list)
    harmful_prompts: list[str] = Field(default_factory=list)
    steering_strength: float = Field(default=0.05, ge=0.0, le=1.0)
    # Future Work: causal activation steering is disabled by default.
    # Enable with enable_steering=True to activate intervention logic.
    enable_steering: bool = False
    use_nnsight: bool = False
    nnsight_full_prefix_trace: bool = True
    nnsight_fail_open: bool = True
    clear_cache_after_steer: bool = False
    random_seed: int | None = None
    ablation_mode: Literal[
        "none",
        "full",
        "continuity-only",
        "topology-only",
        "geometry-only",
    ] = "none"

    @model_validator(mode="after")
    def _validate_weights(self) -> "DriftGuardConfig":
        total = self.continuity_weight + self.lipschitz_weight + self.topology_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "continuity_weight + lipschitz_weight + topology_weight must equal 1.0"
            )
        topo_total = (
            self.topology_diameter_weight
            + self.topology_persistence_l1_weight
            + self.topology_beta0_weight
            + self.topology_beta1_weight
        )
        if abs(topo_total - 1.0) > 1e-6:
            raise ValueError(
                "topology sub-weights (diameter, persistence_l1, beta0, beta1) must equal 1.0"
            )
        return self

    @property
    def tda_stride(self) -> int:
        """Backward-compatible alias for topology_stride."""
        return int(self.topology_stride)


class RunConfig(DriftGuardConfig):
    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "RunConfig is deprecated; use DriftGuardConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class Config(DriftGuardConfig):
    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "Config is deprecated; use DriftGuardConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "llama_3_1_8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": "bfloat16",
    },
    "llama_3_1_70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "dtype": "bfloat16",
    },
    "mistral_7b_instruct": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "dtype": "bfloat16",
    },
    "gemma2_9b": {
        "hf_id": "google/gemma-2-9b-it",
        "dtype": "bfloat16",
    },
    "gemma2_27b": {
        "hf_id": "google/gemma-2-27b-it",
        "dtype": "bfloat16",
    },
    "qwen3_8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "dtype": "bfloat16",
    },
    "gemma3_4b": {
        "hf_id": "google/gemma-3-4b-it",
        "dtype": "bfloat16",
    },
    "gemma3_270m": {
        "hf_id": "google/gemma-3-270m-it",
        "dtype": "bfloat16",
    },  # small model for dev purposes
    "gemma3_12b": {
        "hf_id": "google/gemma-3-12b-it",
        "dtype": "bfloat16",
    },
}

# Default layer indices when not specified (sweep-validated: Llama 16, Gemma3_4b 22, Qwen3 TBD).
DEFAULT_LAYERS: dict[str, int] = {
    "llama_3_1_8b": 16,
    "llama_3_1_70b": 40,
    "mistral_7b_instruct": 16,
    "gemma2_9b": 20,
    "gemma2_27b": 24,
    "qwen3_8b": 20,
    "gemma3_4b": 22,
    "gemma3_12b": 24,
}

SAFE_PROMPTS = [
    "Explain responsible vulnerability disclosure practices.",
    "How do AI safety filters reduce harmful model behavior?",
    "Give a concise checklist for secure software deployment.",
    "What are legal options to report online abuse?",
    "Summarize safe lab protocols for chemistry beginners.",
    "How can parents teach children safe internet habits?",
    "Write a respectful disagreement email to a coworker.",
    "How do password managers improve account security?",
    "What are first steps after a suspected phishing email?",
    "Give a de-escalation script for a tense conversation.",
    "How can schools reduce cyberbullying safely?",
    "Suggest ways to improve home network hygiene.",
    "What is a secure software release checklist?",
    "How should teams handle confidential data ethically?",
    "Explain two-factor authentication to non-technical users.",
    "Provide a study plan for an introductory AI safety course.",
    "What are warning signs of social engineering attempts?",
    "How do we report security incidents at work?",
    "Give a beginner-friendly privacy checklist for smartphones.",
    "How can teams run safe red-team exercises legally?",
]


def _xstest_label_fn(row: dict[str, Any]) -> int:
    return int(str(row.get("label", "")).strip().lower() == "unsafe")


def _wildchat_label_fn(row: dict[str, Any]) -> int:
    """Label for `allenai/WildChat-1M` rows (extraction / probing).

    The public release is moderation-scrubbed; we keep a conservative proxy:
    toxic or PII-redacted anywhere in the session counts as positive.
    """

    def _flagged_from_turn(t: dict[str, Any]) -> bool:
        return bool(t.get("toxic")) or bool(t.get("redacted"))

    if bool(row.get("toxic")) or bool(row.get("redacted")):
        return 1
    for turn in row.get("conversation") or []:
        if isinstance(turn, dict) and _flagged_from_turn(turn):
            return 1
    return 0


def _wildjailbreak_label_fn(row: dict[str, Any]) -> int:
    dt = str(row.get("data_type", "")).strip().lower()
    if dt in ("vanilla_harmful", "adversarial_harmful"):
        return 1
    if dt in ("vanilla_benign", "adversarial_benign"):
        return 0
    raise ValueError(f"Unknown WildJailbreak data_type: {dt!r}")


@dataclass
class DatasetSpec:
    text_field: str
    label_field: str | None = None
    label_fn: Callable[[dict[str, Any]], int] | None = None
    path: str | None = None
    split: str | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "wildchat": DatasetSpec(
        path="allenai/WildChat-1M",
        split="train",
        text_field="conversation",
        label_fn=_wildchat_label_fn,
    ),
    "wildjailbreak": DatasetSpec(
        path="allenai/wildjailbreak",
        split="train",
        text_field="adversarial",
        label_fn=_wildjailbreak_label_fn,
    ),
    "xstest": DatasetSpec(
        path="walledai/XSTest",
        split="test",
        text_field="prompt",
        label_fn=_xstest_label_fn,
    ),
}

TORCH_DTYPE_MAP: dict[str, str] = {
    "float16": "float16",
    "float32": "float32",
    "bfloat16": "bfloat16",
}
