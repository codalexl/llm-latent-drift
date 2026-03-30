#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a clean Apple-Silicon-friendly env for MPS usage.
# Usage:
#   bash scripts/setup_mps_env.sh
# Optional flags:
#   USE_NIGHTLY=1 bash scripts/setup_mps_env.sh
#   PYTHON_VERSION=3.12 VENV_NAME=.venv-mps bash scripts/setup_mps_env.sh

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_NAME="${VENV_NAME:-.venv-mps}"
USE_NIGHTLY="${USE_NIGHTLY:-0}"

echo "==> Repo: $(pwd)"
echo "==> Python version: ${PYTHON_VERSION}"
echo "==> Virtual environment: ${VENV_NAME}"
echo "==> USE_NIGHTLY=${USE_NIGHTLY}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but not found in PATH."
  exit 1
fi

echo "==> Installing Python ${PYTHON_VERSION} via uv (if missing)"
uv python install "${PYTHON_VERSION}"

if [[ -d "${VENV_NAME}" ]]; then
  echo "==> Removing existing ${VENV_NAME}"
  rm -rf "${VENV_NAME}"
fi

echo "==> Creating venv ${VENV_NAME}"
uv venv --python "${PYTHON_VERSION}" "${VENV_NAME}"

echo "==> Installing PyTorch"
# Some fresh uv environments may not have pip bootstrapped yet.
if ! "${VENV_NAME}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> Bootstrapping pip with ensurepip"
  "${VENV_NAME}/bin/python" -m ensurepip --upgrade
fi
uv pip install --python "${VENV_NAME}/bin/python" -U pip
if [[ "${USE_NIGHTLY}" == "1" ]]; then
  uv pip install --python "${VENV_NAME}/bin/python" --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
else
  uv pip install --python "${VENV_NAME}/bin/python" torch torchvision torchaudio
fi

echo "==> Verifying MPS runtime"
"${VENV_NAME}/bin/python" - <<'PY'
import platform
import sys
import torch

print("python:", sys.version.split()[0])
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("mps_built:", torch.backends.mps.is_built())
print("mps_available:", torch.backends.mps.is_available())
PY

echo
echo "==> Done."
echo "Activate with:"
echo "  source ${VENV_NAME}/bin/activate"
echo
echo "Recommended for runtime stability:"
echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
