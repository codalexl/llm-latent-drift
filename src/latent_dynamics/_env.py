"""Load repo `.env` and default project-local Hugging Face cache when unset."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"

# Append to NO_PROXY / no_proxy so Hub traffic skips broken localhost proxies.
_HF_NO_PROXY_HOSTS = ".huggingface.co,huggingface.co,hf.co"

_loaded = False


def _maybe_bypass_proxy_for_hf() -> None:
    """If LATENT_DYNAMICS_BYPASS_PROXY_FOR_HF is set, route HF hosts around HTTP(S)_PROXY."""
    flag = os.environ.get("LATENT_DYNAMICS_BYPASS_PROXY_FOR_HF", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    cur = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").strip()
    if "huggingface" in cur.lower():
        return
    merged = f"{cur},{_HF_NO_PROXY_HOSTS}".strip(",") if cur else _HF_NO_PROXY_HOSTS
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def load_repo_environment() -> None:
    """Load `.env` (no override of existing env), then set HF cache paths if unset."""
    global _loaded
    if _loaded:
        return
    _loaded = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None and _ENV_FILE.is_file():
        load_dotenv(_ENV_FILE, override=False)

    _maybe_bypass_proxy_for_hf()

    if not os.environ.get("HF_HOME"):
        cache_root = _REPO_ROOT / ".cache" / "huggingface"
        os.environ["HF_HOME"] = str(cache_root)
    hf_home = os.environ["HF_HOME"]
    hub = str(Path(hf_home) / "hub")
    if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
        os.environ["HUGGINGFACE_HUB_CACHE"] = hub
    if not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
