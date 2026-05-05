#!/usr/bin/env bash
# Setup variant for Blackwell GPUs (RTX 5090, 5080 — sm_120).
# Differs from setup_cloud.sh only in CUDA wheel version + bitsandbytes pin.
# Use this if you rented a 5090 / 5080 instance on Vast.ai or similar.

set -euo pipefail

echo "[1/7] System info"
nvidia-smi || { echo "ERROR: nvidia-smi not found"; exit 1; }
python3 --version

echo "[2/7] Python venv"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "[3/7] PyTorch (CUDA 12.4 — required for Blackwell sm_120)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "[4/7] Project deps + Blackwell-compatible bitsandbytes"
pip install -e ".[dev,train]"
pip install --upgrade "bitsandbytes>=0.45.0"

echo "[5/7] HuggingFace auth"
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "WARNING: HF_TOKEN not set."
fi

echo "[6/7] Pull ViT checkpoint"
mkdir -p ../clip_model
if [ -n "${HF_CKPT_REPO:-}" ]; then
  hf download "$HF_CKPT_REPO" best_model.pth --local-dir ../clip_model
fi

echo "[7/7] Sanity"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"}  capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"\"}')"
python -c "import bitsandbytes; print(f'bnb: {bitsandbytes.__version__}')"
python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
pytest -q

echo ""
echo "Setup complete. tmux + make data + make stage1 + make stage2 + make eval"
