from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.steering import steer_toward_reference
from latent_dynamics.tda_metrics import topology_snapshot


@dataclass
class DriftGuardConfig:
    layer_idx: int = -1
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    cosine_floor: float = 0.96
    lipschitz_ceiling: float = 0.20
    risk_threshold: float = 0.5
    topology_window: int = 24
    topology_stride: int = 4
    steering_strength: float = 0.20


@dataclass
class DriftStep:
    token_id: int
    cosine_continuity: float | None
    lipschitz_proxy: float | None
    topology_diameter: float | None
    topology_beta0: int | None
    topology_beta1: int | None
    risk_score: float
    alarm: bool
    steered: bool


@dataclass
class DriftSessionResult:
    generated_text: str
    steps: list[DriftStep]
    alarms: int
    steered_steps: int


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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    cfg: DriftGuardConfig,
    device: str,
    safe_reference: torch.Tensor | None = None,
) -> DriftSessionResult:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    past_key_values = None
    prev_hidden: torch.Tensor | None = None
    hidden_history: list[np.ndarray] = []
    generated: list[int] = []
    steps: list[DriftStep] = []

    for step_idx in range(cfg.max_new_tokens):
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
        if prev_hidden is not None:
            cos = _cosine(hidden, prev_hidden)
            lip = _lipschitz_proxy(hidden, prev_hidden)

        if (
            len(hidden_history) >= cfg.topology_window
            and step_idx % max(cfg.topology_stride, 1) == 0
        ):
            window = np.asarray(hidden_history[-cfg.topology_window :], dtype=np.float32)
            topo = topology_snapshot(window)
            topology_diam = topo.diameter
            topology_beta0 = topo.beta0
            topology_beta1 = topo.beta1

        continuity_risk = 0.0
        if cos is not None:
            continuity_risk += max(0.0, cfg.cosine_floor - cos) / max(
                1.0 - cfg.cosine_floor, 1e-6
            )
        lipschitz_risk = 0.0
        if lip is not None:
            lipschitz_risk = max(0.0, lip - cfg.lipschitz_ceiling) / max(
                cfg.lipschitz_ceiling, 1e-6
            )
        risk_score = float(np.clip(0.5 * continuity_risk + 0.5 * lipschitz_risk, 0.0, 1.5))
        alarm = risk_score >= cfg.risk_threshold

        logits = out.logits[:, -1, :]
        steered = False
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

        token_id = _next_token_id(
            logits=logits,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
        )
        generated.append(token_id)
        steps.append(
            DriftStep(
                token_id=token_id,
                cosine_continuity=cos,
                lipschitz_proxy=lip,
                topology_diameter=topology_diam,
                topology_beta0=topology_beta0,
                topology_beta1=topology_beta1,
                risk_score=risk_score,
                alarm=alarm,
                steered=steered,
            )
        )
        prev_hidden = hidden

        if token_id == tokenizer.eos_token_id:
            break

        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    alarms = sum(1 for s in steps if s.alarm)
    steered_steps = sum(1 for s in steps if s.steered)
    return DriftSessionResult(
        generated_text=generated_text,
        steps=steps,
        alarms=alarms,
        steered_steps=steered_steps,
    )
