from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter

from latent_dynamics.config import DriftGuardConfig
from latent_dynamics.steering import steer_toward_reference, steer_with_nnsight
from latent_dynamics.tda_metrics import compute_risk_score, topology_snapshot


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
    alarm: bool
    steered: bool
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


def _resolve_nnsight_layer_stack(model_obj: object) -> object:
    roots = [getattr(model_obj, "model", None), model_obj]
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


def _materialize_proxy(value: object) -> object:
    return value.value if hasattr(value, "value") else value


def _maybe_clear_device_cache(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.startswith("mps") and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
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


def run_driftguard_session(
    model: object,
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: DriftGuardConfig,
    device: str,
    safe_reference: torch.Tensor | None = None,
) -> DriftSessionResult:
    """Run KV-cache online monitoring with optional latent steering."""
    if cfg.use_nnsight and hasattr(model, "trace"):
        return run_driftguard_session_nnsight(
            nns_model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            cfg=cfg,
            device=device,
            safe_reference=safe_reference,
        )

    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

    encoded = tokenizer(prompt, return_tensors="pt")
    full_input_ids = encoded["input_ids"].to(device)
    full_attention_mask = encoded["attention_mask"].to(device)
    input_ids = full_input_ids
    attention_mask = full_attention_mask

    past_key_values = None
    prev_hidden: torch.Tensor | None = None
    hidden_history: deque[np.ndarray] = deque(maxlen=cfg.topology_window)
    last_topology = None
    generated: list[int] = []
    steps: list[DriftStep] = []
    risk_cfg = cfg

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
        hidden_history.append(hidden.squeeze(0).float().cpu().numpy())

        cos = None
        lip = None
        topology_diam = None
        topology_beta0 = None
        topology_beta1 = None
        topology_persistence_l1 = None
        if prev_hidden is not None:
            cos = _cosine(hidden, prev_hidden)
            lip = _lipschitz_proxy(hidden, prev_hidden)

        should_run_tda = (
            cfg.tda_enabled
            and len(hidden_history) >= cfg.topology_window
            and ((perf_counter() - t0) * 1000.0) <= cfg.tda_latency_budget_ms
            and (
                step_idx % max(cfg.topology_stride, 1) == 0
                or last_topology is None
            )
        )
        if should_run_tda:
            window = np.asarray(list(hidden_history), dtype=np.float32)
            last_topology = topology_snapshot(
                window,
                config=risk_cfg,
                pca_components=cfg.pca_components,
                tda_enabled=cfg.tda_enabled,
            )
        if last_topology is not None:
            topology_diam = last_topology.diameter
            topology_beta0 = last_topology.beta0
            topology_beta1 = last_topology.beta1
            topology_persistence_l1 = last_topology.persistence_l1

        risk_score = compute_risk_score(
            {
                "cosine_cont": cos,
                "lipschitz": lip,
                "cloud_diameter": topology_diam,
                "beta0": topology_beta0,
                "beta1": topology_beta1,
                "persistence_l1": topology_persistence_l1,
            },
            config=risk_cfg,
        )
        alarm = risk_score >= cfg.risk_threshold

        logits = out.logits[:, -1, :]
        steered = False
        steering_delta_norm = None
        cache_reset_needed = False
        if alarm and safe_reference is not None:
            steered_hidden, steering_result = steer_toward_reference(
                hidden=hidden,
                reference=safe_reference.to(hidden.device),
                strength=cfg.steering_strength,
            )
            base_logits = model.get_output_embeddings()(hidden)
            steer_logits = model.get_output_embeddings()(steered_hidden)
            logits = logits + (steer_logits - base_logits)
            steered = steering_result.applied
            steering_delta_norm = steering_result.delta_norm
            if steered and cfg.clear_cache_after_steer:
                _maybe_clear_device_cache(device)
                cache_reset_needed = True

        token_id = _next_token_id(
            logits=logits,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        generated.append(token_id)
        latency_ms = (perf_counter() - t0) * 1000.0
        steps.append(
            DriftStep(
                token_id=token_id,
                cosine_continuity=cos,
                lipschitz_proxy=lip,
                topology_diameter=topology_diam,
                topology_beta0=topology_beta0,
                topology_beta1=topology_beta1,
                topology_persistence_l1=topology_persistence_l1,
                risk_score=risk_score,
                alarm=alarm,
                steered=steered,
                steering_delta_norm=steering_delta_norm,
                latency_ms=latency_ms,
            )
        )
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
            # Recompute from full prefix after steering to avoid stale cache state.
            past_key_values = None
            input_ids = full_input_ids
            attention_mask = full_attention_mask
        else:
            input_ids = new_token
            attention_mask = torch.ones_like(new_token, device=device)

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
    )


def load_nnsight_model(
    model_path: str,
    device_map: str = "auto",
    load_in_4bit: bool = False,
) -> object:
    try:
        from nnsight import LanguageModel  # type: ignore[import]
        kwargs: dict[str, object] = {"device_map": device_map}
        if load_in_4bit:
            # Not all nnsight backends expose quantized constructor kwargs.
            kwargs["dispatch"] = True
        return LanguageModel(model_path, **kwargs)
    except Exception:
        pass

    try:
        from nnsight import NNsight  # type: ignore[import]
    except Exception as exc:
        raise ImportError("nnsight is required for --use-nnsight.") from exc

    if hasattr(NNsight, "from_pretrained"):
        return NNsight.from_pretrained(
            model_path,
            device_map=device_map,
            load_in_4bit=load_in_4bit,
        )

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
    )
    return NNsight(hf_model)


def run_driftguard_session_nnsight(
    nns_model: object,
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: DriftGuardConfig,
    device: str,
    safe_reference: torch.Tensor | None = None,
) -> DriftSessionResult:
    """
    nnsight-backed online drift loop.

    By default this path traces the full running prefix each step, which favors
    tracing fidelity over speed.
    """
    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

    running_text = prompt
    prev_hidden: torch.Tensor | None = None
    hidden_history: deque[np.ndarray] = deque(maxlen=cfg.topology_window)
    last_topology = None
    generated: list[int] = []
    steps: list[DriftStep] = []
    risk_cfg = cfg

    for step_idx in range(cfg.max_new_tokens):
        t0 = perf_counter()
        layers = _resolve_nnsight_layer_stack(nns_model)
        with nns_model.trace(running_text):
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
        hidden_history.append(hidden.squeeze(0).float().cpu().numpy())

        cos = None
        lip = None
        topology_diam = None
        topology_beta0 = None
        topology_beta1 = None
        topology_persistence_l1 = None
        if prev_hidden is not None:
            cos = _cosine(hidden, prev_hidden)
            lip = _lipschitz_proxy(hidden, prev_hidden)

        should_run_tda = (
            cfg.tda_enabled
            and len(hidden_history) >= cfg.topology_window
            and ((perf_counter() - t0) * 1000.0) <= cfg.tda_latency_budget_ms
            and (
                step_idx % max(cfg.topology_stride, 1) == 0
                or last_topology is None
            )
        )
        if should_run_tda:
            window = np.asarray(list(hidden_history), dtype=np.float32)
            last_topology = topology_snapshot(
                window,
                config=risk_cfg,
                pca_components=cfg.pca_components,
                tda_enabled=cfg.tda_enabled,
            )
        if last_topology is not None:
            topology_diam = last_topology.diameter
            topology_beta0 = last_topology.beta0
            topology_beta1 = last_topology.beta1
            topology_persistence_l1 = last_topology.persistence_l1

        risk_score = compute_risk_score(
            {
                "cosine_cont": cos,
                "lipschitz": lip,
                "cloud_diameter": topology_diam,
                "beta0": topology_beta0,
                "beta1": topology_beta1,
                "persistence_l1": topology_persistence_l1,
            },
            config=risk_cfg,
        )
        alarm = risk_score >= cfg.risk_threshold
        steered = False
        steering_delta_norm = None

        logits_for_sample = logits_last
        if alarm and safe_reference is not None:
            if cfg.nnsight_full_prefix_trace:
                try:
                    steer_out = steer_with_nnsight(
                        nns_model=nns_model,
                        prompt=running_text,
                        safe_ref_hidden=safe_reference,
                        layer_idx=cfg.layer_idx,
                        alpha=cfg.steering_strength,
                    )
                    logits_for_sample = torch.as_tensor(steer_out["logits"]).to(logits_last.device)
                    steering_delta_norm = float(torch.norm(logits_for_sample - logits_last, p=2).item())
                    if steering_delta_norm <= 1e-8:
                        raise RuntimeError("nnsight steering produced zero logit delta.")
                    steered = True
                except Exception:
                    if not cfg.nnsight_fail_open:
                        raise
                    steered_hidden, steering_result = steer_toward_reference(
                        hidden=hidden,
                        reference=safe_reference.to(hidden.device),
                        strength=cfg.steering_strength,
                    )
                    base_logits = nns_model.lm_head(hidden)
                    steer_logits = nns_model.lm_head(steered_hidden)
                    logits_for_sample = logits_last + (steer_logits - base_logits)
                    steered = steering_result.applied
                    steering_delta_norm = steering_result.delta_norm
            else:
                steered_hidden, steering_result = steer_toward_reference(
                    hidden=hidden,
                    reference=safe_reference.to(hidden.device),
                    strength=cfg.steering_strength,
                )
                base_logits = nns_model.lm_head(hidden)
                steer_logits = nns_model.lm_head(steered_hidden)
                logits_for_sample = logits_last + (steer_logits - base_logits)
                steered = steering_result.applied
                steering_delta_norm = steering_result.delta_norm

            if steered and cfg.clear_cache_after_steer:
                _maybe_clear_device_cache(device)

        token_id = _next_token_id(
            logits=logits_for_sample,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        generated.append(token_id)
        latency_ms = (perf_counter() - t0) * 1000.0
        steps.append(
            DriftStep(
                token_id=token_id,
                cosine_continuity=cos,
                lipschitz_proxy=lip,
                topology_diameter=topology_diam,
                topology_beta0=topology_beta0,
                topology_beta1=topology_beta1,
                topology_persistence_l1=topology_persistence_l1,
                risk_score=risk_score,
                alarm=alarm,
                steered=steered,
                steering_delta_norm=steering_delta_norm,
                latency_ms=latency_ms,
            )
        )
        prev_hidden = hidden

        if token_id == tokenizer.eos_token_id:
            break
        running_text = running_text + tokenizer.decode([token_id], skip_special_tokens=False)

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
    )
