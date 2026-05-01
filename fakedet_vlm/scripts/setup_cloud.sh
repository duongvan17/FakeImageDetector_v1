#!/usr/bin/env bash
# One-shot setup for cloud GPU instances (RunPod / Vast.ai / Lambda / etc.)
# Assumes a fresh Ubuntu 22.04 box with NVIDIA driver + CUDA already installed
# (most "PyTorch" templates on these providers come with that).
#
# Required env vars:
#   HF_TOKEN          - HuggingFace token (read scope is enough)
#   HF_CKPT_REPO      - HF repo id holding best_model.pth, e.g. "user/fakedet-vit"
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export HF_CKPT_REPO=duongvan17/fakedet-vit-b16-fakeclue
#   bash scripts/setup_cloud.sh

set -euo pipefail

echo "[1/7] System info"
nvidia-smi || { echo "ERROR: nvidia-smi not found — wrong template"; exit 1; }
python3 --version

echo "[2/7] Python venv"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "[3/7] PyTorch (CUDA 12.1)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[4/7] Project deps"
pip install -e ".[dev,train]"

echo "[5/7] HuggingFace auth"
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "WARNING: HF_TOKEN not set. Call 'hf auth login' manually."
fi

echo "[6/7] Pull ViT checkpoint from HF Hub"
mkdir -p ../clip_model
if [ -n "${HF_CKPT_REPO:-}" ]; then
  hf download "$HF_CKPT_REPO" best_model.pth --local-dir ../clip_model
else
  echo "WARNING: HF_CKPT_REPO not set. Upload best_model.pth manually to ../clip_model/."
fi

echo "[7/7] Sanity check"
if [ -f ../clip_model/best_model.pth ]; then
  python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
else
  echo "ERROR: ../clip_model/best_model.pth still missing — cannot continue."
  exit 1
fi
pytest -q

echo ""
echo "Setup complete. Next steps:"
echo "  tmux                      # avoid disconnect kills"
echo "  make data                 # ~30-60 min, ~30 GB"
echo "  make stage1               # ~2-3 h on RTX 4090"
echo "  make stage2               # ~2-3 h on RTX 4090"
echo "  make eval                 # ~10-30 min"
echo ""
echo "When done, copy back to local:"
echo "  scp -r runs/stage2_sft/final user@local:./fakedet_vlm/runs/stage2_sft/"
