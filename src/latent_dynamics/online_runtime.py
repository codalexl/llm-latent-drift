from __future__ import annotations

import logging
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter

from latent_dynamics.contrastive_vectors import compute_contrastive_vector
from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.steering import (
    apply_contrastive_steering,
    steer_toward_reference,
    steer_with_nnsight,
)
from latent_dynamics.tda_metrics import (
    compute_risk_score,
    decompose_risk_components,
    topology_snapshot,
)

if TYPE_CHECKING:
    from nnsight import NNsight
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class DriftStep:
    token_id: int
    cosine_continuity: float | None
    lipschitz_proxy: float | None
    topology_diameter: float | None
    topology_beta0: int | None
    topology_beta1: int | None
    topology_persistence_l1: float | None
    risk_score: float
    continuity_risk: float
    lipschitz_risk: float
    topology_risk: float
    topology_diameter_risk: float
    topology_persistence_l1_risk: float
    topology_beta0_risk: float
    topology_beta1_risk: float
    tda_backend: str | None
    tda_approximate: bool
    alarm: bool
    steered: bool
    probe_risk: float | None = None
    dynamics_risk: float | None = None
    steering_delta_norm: float | None = None
    latency_ms: float | None = None


@dataclass
class DriftSessionResult:
    generated_text: str
    steps: list[DriftStep]
    alarms: int
    steered_steps: int
    first_alarm_token: int | None
    first_alarm_lead_time: int | None
    mean_step_latency_ms: float
    tda_attempted_steps: int
    tda_executed_steps: int
    tda_skipped_budget_steps: int
    tda_skipped_stride_steps: int


@dataclass
class SteeringIntervention:
    logits: torch.Tensor
    steered: bool
    delta_norm: float | None


@dataclass
class _TdaState:
    """Mutable state for TDA budget tracking across generation steps."""
    attempted: int = 0
    executed: int = 0
    skipped_budget: int = 0
    skipped_stride: int = 0
    estimated_ms: float | None = None


@dataclass
class _GpuHiddenRingBuffer:
    """GPU-resident ring buffer for hidden-state history."""
    data: torch.Tensor
    write_idx: int = 0
    count: int = 0

    @classmethod
    def create(
        cls,
        window: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "_GpuHiddenRingBuffer":
        return cls(
            data=torch.zeros((window, hidden_dim), device=device, dtype=dtype),
            write_idx=0,
            count=0,
        )

    def append(self, hidden: torch.Tensor) -> None:
        self.data[self.write_idx] = hidden.to(dtype=self.data.dtype)
        self.write_idx = (self.write_idx + 1) % self.data.shape[0]
        self.count = min(self.count + 1, self.data.shape[0])

    def window_numpy(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((0, self.data.shape[1]), dtype=np.float32)
        if self.count < self.data.shape[0]:
            ordered = self.data[: self.count]
        else:
            ordered = torch.cat(
                [self.data[self.write_idx :], self.data[: self.write_idx]],
                dim=0,
            )
        return ordered.detach().to(dtype=torch.float32).to("cpu").numpy()


class ModelAdapter:
    """Light adapter to resolve architecture-specific nnsight model internals."""

    def __init__(self, model_obj: object):
        self.model_obj = model_obj

    def layer_stack(self) -> object:
        roots = [getattr(self.model_obj, "model", None), self.model_obj]
        try:
            attrs = set(dir(self.model_obj))
        except Exception as exc:
            logger.debug("Unable to inspect nnsight model attrs: %s", exc)
            attrs = set()
        if "model" in attrs:
            try:
                nested_model = inspect.getattr_static(self.model_obj, "model")
                if nested_model is not None and hasattr(nested_model, "layers"):
                    return nested_model.layers
            except Exception as exc:
                logger.debug("Static model-layer inspection failed: %s", exc)
        for root in roots:
            if root is None:
                continue
            if hasattr(root, "layers"):
                return root.layers
            if hasattr(root, "model") and hasattr(root.model, "layers"):
                return root.model.layers
            if hasattr(root, "language_model") and hasattr(root.language_model, "layers"):
                return root.language_model.layers
            if hasattr(root, "transformer") and hasattr(root.transformer, "h"):
                return root.transformer.h
            if hasattr(root, "gpt_neox") and hasattr(root.gpt_neox, "layers"):
                return root.gpt_neox.layers
        raise AttributeError("Could not locate nnsight layer stack on wrapped model.")


def _resolve_nnsight_layer_stack(model_obj: object) -> object:
    """Backward-compatible wrapper around ModelAdapter layer resolution."""
    return ModelAdapter(model_obj).layer_stack()


def _materialize_proxy(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _resolve_contrastive_vector(
    cfg: DriftGuardConfig,
    layer_idx: int,
    contrastive_vectors: dict[str, list[float]] | None = None,
) -> torch.Tensor | None:
    if contrastive_vectors:
        key = f"layer_{layer_idx}"
        if key in contrastive_vectors:
            vec = np.asarray(contrastive_vectors[key], dtype=np.float32)
            if vec.ndim == 1 and vec.size > 0:
                return torch.as_tensor(vec)
    return None


def _project_probe_risk(hidden: torch.Tensor, contrastive_vector: torch.Tensor | None) -> float:
    if contrastive_vector is None:
        return 0.0
    h = hidden.squeeze(0).float()
    vec = contrastive_vector.to(h.device).float()
    if vec.ndim != 1 or vec.numel() != h.numel():
        return 0.0
    unit = vec / torch.clamp(torch.norm(vec, p=2), min=1e-8)
    proj = float(torch.dot(h, unit).item())
    return max(0.0, proj)


def _hybrid_risk_score(cfg: DriftGuardConfig, probe_risk: float, dynamics_risk: float) -> float:
    if not cfg.use_contrastive_probe:
        return float(np.clip(dynamics_risk, 0.0, 1.5))
    probe_w = float(np.clip(cfg.probe_weight, 0.0, 1.0))
    score = probe_w * float(probe_risk) + (1.0 - probe_w) * float(dynamics_risk)
    return float(np.clip(score, 0.0, 1.5))


def _maybe_clear_device_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.startswith("mps") and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def _cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    xn = torch.norm(x, p=2)
    yn = torch.norm(y, p=2)
    denom = torch.clamp(xn * yn, min=1e-8)
    return float(torch.sum(x * y).item() / float(denom.item()))


def _lipschitz_proxy(x: torch.Tensor, y: torch.Tensor) -> float:
    num = torch.norm(x - y, p=2)
    den = torch.clamp(torch.norm(y, p=2), min=1e-8)
    return float((num / den).item())


def _tda_within_budget(
    tda_latency_budget_ms: float,
    estimated_tda_ms: float | None,
) -> bool:
    """Gate TDA work using estimated TDA cost vs budget.

    Only the TDA computation cost is checked against the budget -- the model
    forward-pass time is unavoidable and should not consume the TDA budget.
    On the first eligible step (no estimate yet), TDA is allowed to run so
    the runtime can bootstrap a cost estimate for subsequent gating.
    """
    if estimated_tda_ms is None:
        return True
    return float(estimated_tda_ms) <= float(tda_latency_budget_ms)


# ---------------------------------------------------------------------------
# Shared step-level monitoring logic (DRY: used by both HF and nnsight paths)
# ---------------------------------------------------------------------------

@dataclass
class _StepMetrics:
    """Intermediate monitoring results for a single generation step."""
    cos: float | None
    lip: float | None
    topology_diam: float | None
    topology_beta0: int | None
    topology_beta1: int | None
    topology_persistence_l1: float | None
    tda_backend: str | None
    tda_approximate: bool
    risk_score: float
    probe_risk: float
    dynamics_risk: float
    continuity_risk: float
    lipschitz_risk: float
    topology_risk: float
    topology_diameter_risk: float
    topology_persistence_l1_risk: float
    topology_beta0_risk: float
    topology_beta1_risk: float
    alarm: bool


def _compute_step_metrics(
    hidden: torch.Tensor,
    prev_hidden: torch.Tensor | None,
    hidden_history: _GpuHiddenRingBuffer,
    last_topology: object | None,
    step_idx: int,
    cfg: DriftGuardConfig,
    tda_state: _TdaState,
    contrastive_vector: torch.Tensor | None,
) -> tuple[_StepMetrics, object | None]:
    """Compute all monitoring metrics for one generation step.

    Returns the metrics and the (possibly updated) last_topology snapshot.
    """
    cos = None
    lip = None
    if prev_hidden is not None:
        cos = _cosine(hidden, prev_hidden)
        lip = _lipschitz_proxy(hidden, prev_hidden)

    # TDA gating
    has_window = hidden_history.count >= cfg.topology_window
    within_budget = _tda_within_budget(
        tda_latency_budget_ms=cfg.tda_latency_budget_ms,
        estimated_tda_ms=tda_state.estimated_ms,
    )
    stride_hit = step_idx % max(cfg.topology_stride, 1) == 0 or last_topology is None
    should_run_tda = cfg.tda_enabled and has_window and within_budget and stride_hit
    if cfg.tda_enabled and has_window:
        tda_state.attempted += 1
        if not within_budget:
            tda_state.skipped_budget += 1
        elif not stride_hit:
            tda_state.skipped_stride += 1

    if should_run_tda:
        tda_t0 = perf_counter()
        window = hidden_history.window_numpy()
        last_topology = topology_snapshot(
            window,
            config=cfg,
            pca_components=cfg.pca_components,
            tda_enabled=cfg.tda_enabled,
        )
        tda_state.executed += 1
        tda_elapsed_ms = (perf_counter() - tda_t0) * 1000.0
        if tda_state.estimated_ms is None:
            tda_state.estimated_ms = float(tda_elapsed_ms)
        else:
            tda_state.estimated_ms = float(0.8 * tda_state.estimated_ms + 0.2 * tda_elapsed_ms)

    # Extract topology values
    topology_diam = None
    topology_beta0 = None
    topology_beta1 = None
    topology_persistence_l1 = None
    if last_topology is not None:
        topology_diam = last_topology.diameter
        topology_beta0 = last_topology.beta0
        topology_beta1 = last_topology.beta1
        topology_persistence_l1 = last_topology.persistence_l1
    tda_backend = last_topology.tda_backend if last_topology is not None else None
    tda_approximate = bool(last_topology.tda_approximate) if last_topology is not None else False

    # Risk computation
    risk_metrics = {
        "cosine_cont": cos,
        "lipschitz": lip,
        "cloud_diameter": topology_diam,
        "beta0": topology_beta0,
        "beta1": topology_beta1,
        "persistence_l1": topology_persistence_l1,
    }
    risk_parts = decompose_risk_components(risk_metrics, config=cfg)
    dynamics_risk = compute_risk_score(risk_metrics, config=cfg)
    probe_risk = _project_probe_risk(hidden, contrastive_vector)
    risk_score = _hybrid_risk_score(
        cfg=cfg,
        probe_risk=probe_risk,
        dynamics_risk=dynamics_risk,
    )
    alarm = risk_score >= cfg.risk_threshold

    metrics = _StepMetrics(
        cos=cos,
        lip=lip,
        topology_diam=topology_diam,
        topology_beta0=topology_beta0,
        topology_beta1=topology_beta1,
        topology_persistence_l1=topology_persistence_l1,
        tda_backend=tda_backend,
        tda_approximate=tda_approximate,
        risk_score=risk_score,
        probe_risk=probe_risk,
        dynamics_risk=dynamics_risk,
        continuity_risk=risk_parts.continuity,
        lipschitz_risk=risk_parts.lipschitz,
        topology_risk=risk_parts.topology,
        topology_diameter_risk=risk_parts.topology_diameter,
        topology_persistence_l1_risk=risk_parts.topology_persistence_l1,
        topology_beta0_risk=risk_parts.topology_beta0,
        topology_beta1_risk=risk_parts.topology_beta1,
        alarm=alarm,
    )
    return metrics, last_topology


def _build_drift_step(
    token_id: int,
    m: _StepMetrics,
    steered: bool,
    steering_delta_norm: float | None,
    contrastive_vector: torch.Tensor | None,
    latency_ms: float,
) -> DriftStep:
    """Build a DriftStep from pre-computed metrics."""
    return DriftStep(
        token_id=token_id,
        cosine_continuity=m.cos,
        lipschitz_proxy=m.lip,
        topology_diameter=m.topology_diam,
        topology_beta0=m.topology_beta0,
        topology_beta1=m.topology_beta1,
        topology_persistence_l1=m.topology_persistence_l1,
        risk_score=m.risk_score,
        continuity_risk=m.continuity_risk,
        lipschitz_risk=m.lipschitz_risk,
        topology_risk=m.topology_risk,
        topology_diameter_risk=m.topology_diameter_risk,
        topology_persistence_l1_risk=m.topology_persistence_l1_risk,
        topology_beta0_risk=m.topology_beta0_risk,
        topology_beta1_risk=m.topology_beta1_risk,
        tda_backend=m.tda_backend,
        tda_approximate=m.tda_approximate,
        alarm=m.alarm,
        steered=steered,
        probe_risk=m.probe_risk if contrastive_vector is not None else None,
        dynamics_risk=m.dynamics_risk,
        steering_delta_norm=steering_delta_norm,
        latency_ms=latency_ms,
    )


def _summarize_session(
    generated: list[int],
    steps: list[DriftStep],
    tokenizer: AutoTokenizer,
    tda_state: _TdaState,
) -> DriftSessionResult:
    """Build a DriftSessionResult from accumulated steps."""
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    alarms = sum(1 for s in steps if s.alarm)
    steered_steps = sum(1 for s in steps if s.steered)
    first_alarm = next((i for i, s in enumerate(steps) if s.alarm), None)
    first_alarm_lead = None
    if first_alarm is not None:
        first_alarm_lead = max(0, len(steps) - 1 - first_alarm)
    mean_latency = float(np.mean([s.latency_ms for s in steps if s.latency_ms is not None])) if steps else 0.0
    return DriftSessionResult(
        generated_text=generated_text,
        steps=steps,
        alarms=alarms,
        steered_steps=steered_steps,
        first_alarm_token=first_alarm,
        first_alarm_lead_time=first_alarm_lead,
        mean_step_latency_ms=mean_latency,
        tda_attempted_steps=tda_state.attempted,
        tda_executed_steps=tda_state.executed,
        tda_skipped_budget_steps=tda_state.skipped_budget,
        tda_skipped_stride_steps=tda_state.skipped_stride,
    )


# ---------------------------------------------------------------------------
# Steering helpers
# ---------------------------------------------------------------------------

def _steer_logits_hf(
    model: object,
    hidden: torch.Tensor,
    logits_last: torch.Tensor,
    safe_reference: torch.Tensor | None,
    contrastive_vector: torch.Tensor | None,
    cfg: DriftGuardConfig,
) -> SteeringIntervention:
    if contrastive_vector is not None:
        steered_hidden, steering_result = apply_contrastive_steering(
            hidden=hidden,
            contrastive_direction=contrastive_vector.to(hidden.device),
            strength=cfg.contrastive_steering_strength,
        )
    elif safe_reference is not None:
        steered_hidden, steering_result = steer_toward_reference(
            hidden=hidden,
            reference=safe_reference.to(hidden.device),
            strength=cfg.steering_strength,
        )
    else:
        return SteeringIntervention(logits=logits_last, steered=False, delta_norm=None)
    if not steering_result.applied:
        return SteeringIntervention(
            logits=logits_last,
            steered=False,
            delta_norm=steering_result.delta_norm,
        )
    output_head = model.get_output_embeddings()
    base_logits = output_head(hidden)
    steer_logits = output_head(steered_hidden)
    return SteeringIntervention(
        logits=logits_last + (steer_logits - base_logits),
        steered=True,
        delta_norm=steering_result.delta_norm,
    )


def _steer_logits_nnsight(
    nns_model: object,
    prompt: str,
    logits_last: torch.Tensor,
    safe_reference: torch.Tensor | None,
    contrastive_vector: torch.Tensor | None,
    cfg: DriftGuardConfig,
) -> SteeringIntervention:
    try:
        steer_out = steer_with_nnsight(
            nns_model=nns_model,
            prompt=prompt,
            safe_ref_hidden=safe_reference,
            layer_idx=cfg.layer_idx,
            alpha=(
                cfg.contrastive_steering_strength
                if contrastive_vector is not None
                else cfg.steering_strength
            ),
            contrastive_direction=contrastive_vector,
            project=True,
        )
        logits_for_sample = torch.as_tensor(steer_out["logits"]).to(logits_last.device)
        delta_norm = float(torch.norm(logits_for_sample - logits_last, p=2).item())
        if delta_norm <= 1e-8:
            raise RuntimeError("nnsight steering produced zero logit delta.")
        return SteeringIntervention(
            logits=logits_for_sample,
            steered=True,
            delta_norm=delta_norm,
        )
    except Exception:
        if not cfg.nnsight_fail_open:
            raise
        logger.warning("nnsight steering failed; falling back to unsteered logits.", exc_info=True)
        return SteeringIntervention(logits=logits_last, steered=False, delta_norm=None)


def _apply_steering_intervention(
    backend: str,
    cfg: DriftGuardConfig,
    device: str,
    logits_last: torch.Tensor,
    hidden: torch.Tensor,
    safe_reference: torch.Tensor | None,
    contrastive_vector: torch.Tensor | None = None,
    model: object | None = None,
    nns_model: object | None = None,
    prompt: str | None = None,
) -> SteeringIntervention:
    """Single steering abstraction used by both HF and nnsight runtimes."""
    if backend == "hf":
        if model is None:
            raise ValueError("HF steering backend requires model.")
        result = _steer_logits_hf(
            model=model,
            hidden=hidden,
            logits_last=logits_last,
            safe_reference=safe_reference,
            contrastive_vector=contrastive_vector,
            cfg=cfg,
        )
    elif backend == "nnsight":
        if nns_model is None or prompt is None:
            raise ValueError("nnsight steering backend requires model and prompt.")
        result = _steer_logits_nnsight(
            nns_model=nns_model,
            prompt=prompt,
            logits_last=logits_last,
            safe_reference=safe_reference,
            contrastive_vector=contrastive_vector,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown steering backend: {backend}")

    return result


def _next_token_id(logits: torch.Tensor, do_sample: bool, temperature: float) -> int:
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())
    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def estimate_safe_reference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: str,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Average last-token hidden states over safe prompts."""
    if not prompts:
        raise ValueError("prompts must be non-empty for safe reference estimation.")
    vectors: list[torch.Tensor] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        h = out.hidden_states[layer_idx][:, -1, :].detach()
        vectors.append(h)
    stacked = torch.cat(vectors, dim=0)
    return torch.mean(stacked, dim=0)


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def _set_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# HF backend session
# ---------------------------------------------------------------------------

def run_driftguard_session(
    model: Union["PreTrainedModel", "NNsight", object],
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: DriftGuardConfig,
    device: str,
    safe_reference: torch.Tensor | None = None,
    contrastive_vectors: dict[str, list[float]] | None = None,
) -> DriftSessionResult:
    """Run full online DriftGuard inference loop with optional steering."""
    _set_random_seed(cfg.random_seed)

    if cfg.use_nnsight and hasattr(model, "trace"):
        return run_driftguard_session_nnsight(
            nns_model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            safe_reference=safe_reference,
            contrastive_vectors=contrastive_vectors,
        )

    encoded = tokenizer(prompt, return_tensors="pt")
    full_input_ids = encoded["input_ids"].to(device)
    full_attention_mask = encoded["attention_mask"].to(device)
    input_ids = full_input_ids
    attention_mask = full_attention_mask

    past_key_values = None
    prev_hidden: torch.Tensor | None = None
    hidden_history: _GpuHiddenRingBuffer | None = None
    last_topology = None
    generated: list[int] = []
    steps: list[DriftStep] = []
    tda_state = _TdaState()

    contrastive_vector = _resolve_contrastive_vector(
        cfg=cfg,
        layer_idx=cfg.layer_idx,
        contrastive_vectors=contrastive_vectors,
    )
    if (
        contrastive_vector is None
        and cfg.safe_prompts
        and cfg.harmful_prompts
        and hasattr(model, "__call__")
    ):
        try:
            vec_np = compute_contrastive_vector(
                model=model,
                tokenizer=tokenizer,
                safe_prompts=cfg.safe_prompts,
                harmful_prompts=cfg.harmful_prompts,
                layer_idx=cfg.layer_idx,
                cfg=cfg,
                device=device,
            )
            contrastive_vector = torch.as_tensor(vec_np)
        except Exception:
            logger.warning("Contrastive vector computation failed; proceeding without.", exc_info=True)
            contrastive_vector = None

    for step_idx in range(cfg.max_new_tokens):
        t0 = perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )

        past_key_values = out.past_key_values
        hidden = out.hidden_states[cfg.layer_idx][:, -1, :].detach()
        if hidden_history is None:
            hidden_history = _GpuHiddenRingBuffer.create(
                window=cfg.topology_window,
                hidden_dim=int(hidden.shape[-1]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        hidden_history.append(hidden.squeeze(0))

        m, last_topology = _compute_step_metrics(
            hidden=hidden,
            prev_hidden=prev_hidden,
            hidden_history=hidden_history,
            last_topology=last_topology,
            step_idx=step_idx,
            cfg=cfg,
            tda_state=tda_state,
            contrastive_vector=contrastive_vector,
        )

        logits = out.logits[:, -1, :]
        steered = False
        steering_delta_norm = None
        cache_reset_needed = False
        if m.alarm and (contrastive_vector is not None or safe_reference is not None):
            steering_out = _apply_steering_intervention(
                backend="hf",
                cfg=cfg,
                device=device,
                logits_last=logits,
                hidden=hidden,
                safe_reference=safe_reference,
                contrastive_vector=contrastive_vector,
                model=model,
            )
            logits = steering_out.logits
            steered = steering_out.steered
            steering_delta_norm = steering_out.delta_norm
            cache_reset_needed = steered and cfg.clear_cache_after_steer

        token_id = _next_token_id(
            logits=logits,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        generated.append(token_id)
        latency_ms = (perf_counter() - t0) * 1000.0
        steps.append(_build_drift_step(
            token_id=token_id,
            m=m,
            steered=steered,
            steering_delta_norm=steering_delta_norm,
            contrastive_vector=contrastive_vector,
            latency_ms=latency_ms,
        ))
        prev_hidden = hidden

        if token_id == tokenizer.eos_token_id:
            break

        new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        full_input_ids = torch.cat([full_input_ids, new_token], dim=1)
        full_attention_mask = torch.cat(
            [full_attention_mask, torch.ones_like(new_token, device=device)],
            dim=1,
        )
        if cache_reset_needed:
            past_key_values = None
            input_ids = full_input_ids
            attention_mask = full_attention_mask
        else:
            input_ids = new_token
            attention_mask = torch.ones_like(new_token, device=device)

    result = _summarize_session(generated, steps, tokenizer, tda_state)
    if cfg.clear_cache_after_steer and any(step.steered for step in steps):
        _maybe_clear_device_cache(device)
    return result


# ---------------------------------------------------------------------------
# nnsight model loading
# ---------------------------------------------------------------------------

def load_nnsight_model(
    model_path: str,
    device_map: str = "auto",
) -> object:
    try:
        from nnsight import LanguageModel  # type: ignore[import]
        kwargs: dict[str, object] = {"device_map": device_map}
        return LanguageModel(model_path, **kwargs)
    except ImportError:
        logger.debug("nnsight.LanguageModel not available; trying NNsight wrapper.")
    except Exception:
        logger.warning("LanguageModel(%s) failed; trying NNsight fallback.", model_path, exc_info=True)

    try:
        from nnsight import NNsight  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("nnsight is required for --use-nnsight.") from exc

    if hasattr(NNsight, "from_pretrained"):
        return NNsight.from_pretrained(
            model_path,
            device_map=device_map,
        )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
    )
    return NNsight(hf_model)


# ---------------------------------------------------------------------------
# nnsight backend session
# ---------------------------------------------------------------------------

def run_driftguard_session_nnsight(
    nns_model: Union["NNsight", object],
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: DriftGuardConfig,
    device: str,
    safe_reference: torch.Tensor | None = None,
    contrastive_vectors: dict[str, list[float]] | None = None,
) -> DriftSessionResult:
    """nnsight-backed online drift loop with cache-aware token stepping.

    Fast path uses incremental token forwarding with `past_key_values` to avoid
    quadratic running-prefix retracing. If a backend does not support direct
    forward calls, this function gracefully falls back to `trace(...)`.
    """
    _set_random_seed(cfg.random_seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    full_input_ids = encoded["input_ids"].to(device)
    full_attention_mask = encoded["attention_mask"].to(device)
    input_ids = full_input_ids
    attention_mask = full_attention_mask
    past_key_values = None
    prev_hidden: torch.Tensor | None = None
    hidden_history: _GpuHiddenRingBuffer | None = None
    last_topology = None
    generated: list[int] = []
    steps: list[DriftStep] = []
    tda_state = _TdaState()

    contrastive_vector = _resolve_contrastive_vector(
        cfg=cfg,
        layer_idx=cfg.layer_idx,
        contrastive_vectors=contrastive_vectors,
    )
    adapter = ModelAdapter(nns_model)
    for step_idx in range(cfg.max_new_tokens):
        t0 = perf_counter()
        try:
            with torch.no_grad():
                out = nns_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
            past_key_values = out.past_key_values
            hidden = out.hidden_states[cfg.layer_idx][:, -1, :].detach()
            logits_last = out.logits[:, -1, :].detach()
        except (TypeError, ValueError, NotImplementedError):
            layers = adapter.layer_stack()
            prompt_for_trace = tokenizer.decode(full_input_ids[0], skip_special_tokens=False)
            with nns_model.trace(prompt_for_trace, scan=True):
                hidden_proxy = layers[cfg.layer_idx].output[0].save()
                logits_proxy = nns_model.lm_head.output.save()
            hidden_val = _materialize_proxy(hidden_proxy)
            logits_val = _materialize_proxy(logits_proxy)
            if isinstance(hidden_val, torch.Tensor):
                hidden_t = hidden_val.detach()
            else:
                hidden_t = torch.as_tensor(hidden_val)
            if hidden_t.ndim == 3:
                hidden = hidden_t[:, -1, :]
            elif hidden_t.ndim == 2:
                hidden = hidden_t[-1:, :]
            else:
                raise ValueError(f"Unexpected hidden tensor rank for nnsight: {hidden_t.ndim}")
            if isinstance(logits_val, torch.Tensor):
                logits_t = logits_val.detach()
            else:
                logits_t = torch.as_tensor(logits_val)
            if logits_t.ndim == 3:
                logits_last = logits_t[:, -1, :]
            elif logits_t.ndim == 2:
                logits_last = logits_t[-1:, :]
            else:
                raise ValueError(f"Unexpected logits tensor rank for nnsight: {logits_t.ndim}")
            past_key_values = None

        if hidden_history is None:
            hidden_history = _GpuHiddenRingBuffer.create(
                window=cfg.topology_window,
                hidden_dim=int(hidden.shape[-1]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        hidden_history.append(hidden.squeeze(0))

        m, last_topology = _compute_step_metrics(
            hidden=hidden,
            prev_hidden=prev_hidden,
            hidden_history=hidden_history,
            last_topology=last_topology,
            step_idx=step_idx,
            cfg=cfg,
            tda_state=tda_state,
            contrastive_vector=contrastive_vector,
        )

        steered = False
        steering_delta_norm = None
        cache_reset_needed = False
        logits_for_sample = logits_last
        if m.alarm and (contrastive_vector is not None or safe_reference is not None):
            # Prefer in-pass hidden-space steering to avoid an extra nnsight trace.
            if hasattr(nns_model, "get_output_embeddings"):
                steering_out = _steer_logits_hf(
                    model=nns_model,
                    hidden=hidden,
                    logits_last=logits_last,
                    safe_reference=safe_reference,
                    contrastive_vector=contrastive_vector,
                    cfg=cfg,
                )
            else:
                prompt_for_steer = tokenizer.decode(full_input_ids[0], skip_special_tokens=False)
                steering_out = _apply_steering_intervention(
                    backend="nnsight",
                    cfg=cfg,
                    device=device,
                    logits_last=logits_last,
                    hidden=hidden,
                    safe_reference=safe_reference,
                    contrastive_vector=contrastive_vector,
                    nns_model=nns_model,
                    prompt=prompt_for_steer,
                )
            logits_for_sample = steering_out.logits
            steered = steering_out.steered
            steering_delta_norm = steering_out.delta_norm
            cache_reset_needed = steered and cfg.clear_cache_after_steer

        token_id = _next_token_id(
            logits=logits_for_sample,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        generated.append(token_id)
        latency_ms = (perf_counter() - t0) * 1000.0
        steps.append(_build_drift_step(
            token_id=token_id,
            m=m,
            steered=steered,
            steering_delta_norm=steering_delta_norm,
            contrastive_vector=contrastive_vector,
            latency_ms=latency_ms,
        ))
        prev_hidden = hidden
        if token_id == tokenizer.eos_token_id:
            break
        new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        full_input_ids = torch.cat([full_input_ids, new_token], dim=1)
        full_attention_mask = torch.cat(
            [full_attention_mask, torch.ones_like(new_token, device=device)],
            dim=1,
        )
        if cache_reset_needed or past_key_values is None:
            past_key_values = None
            input_ids = full_input_ids
            attention_mask = full_attention_mask
        else:
            input_ids = new_token
            attention_mask = torch.ones_like(new_token, device=device)
    return _summarize_session(generated, steps, tokenizer, tda_state)
