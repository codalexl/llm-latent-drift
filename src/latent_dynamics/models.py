from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_dynamics.config import MODEL_REGISTRY

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_key: str,
    device: str,
    load_in_4bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, optionally in 4-bit if CUDA + bitsandbytes are available."""
    spec = MODEL_REGISTRY[model_key]
    hf_id = spec["hf_id"]
    dtype = _TORCH_DTYPES.get(spec.get("dtype", "bfloat16"), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    use_4bit = bool(load_in_4bit and device == "cuda")

    if use_4bit:
        try:
            from bitsandbytes import BitsAndBytesConfig  # type: ignore[import]

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
        except Exception:
            # Fallback to standard loading if bitsandbytes is unavailable.
            quantization_config = None
            use_4bit = False

    if use_4bit and quantization_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    model.eval()
    return model, tokenizer
