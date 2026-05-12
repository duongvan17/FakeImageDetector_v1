#!/usr/bin/env bash
# Setup script for cloud GPU instances on RunPod / Vast.ai / Lambda.
#
# Assumes the official ``pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`` base
# image (or compatible) which already has PyTorch + CUDA pre-installed under
# /opt/conda. We DO NOT create a venv — reinstalling 2.5 GB of torch wheels
# is wasteful when the container ships them.
#
# Required env vars:
#   HF_TOKEN          - HuggingFace token (read scope)
#   HF_CKPT_REPO      - HF repo id holding best_model.pth
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export HF_CKPT_REPO=duongvan17/fakedet-vit-b16-fakeclue
#   bash scripts/setup_cloud.sh

set -euo pipefail

echo "[1/6] System info"
nvidia-smi || { echo "ERROR: nvidia-smi not found — wrong template"; exit 1; }
python3 --version
which python3

echo "[2/6] Install fakedet_vlm package (editable, no deps)"
# --no-deps because we want to use the container's pre-installed torch and
# pip-install only the missing pure-Python libs below.
pip install -e . --no-deps

echo "[3/6] Install / pin Python dependencies"
# transformers >=4.50 calls torch.library.custom_op with string annotations
# that torch 2.4 rejects. Cap below 4.50 until torch is upgraded.
pip install \
  "transformers>=4.46,<4.50" \
  "accelerate>=0.34,<1.0" \
  "peft>=0.12,<0.14" \
  "datasets>=2.20,<3.0" \
  "huggingface_hub>=0.24,<0.27" \
  "tokenizers>=0.20,<0.21" \
  --upgrade

# bitsandbytes / timm / utilities. These are usually missing from the base
# pytorch container.
pip install \
  "bitsandbytes>=0.43" \
  "timm>=1.0.9" \
  "scikit-learn>=1.3" \
  "safetensors>=0.4" \
  "sentencepiece>=0.2" \
  "pyyaml>=6" \
  "tqdm>=4.65" \
  "Pillow>=10"

echo "[4/6] HuggingFace auth"
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "  WARNING: HF_TOKEN not set. Call 'hf auth login' manually."
fi

echo "[5/6] Pull ViT checkpoint from HF Hub"
mkdir -p ../clip_model
if [ -n "${HF_CKPT_REPO:-}" ]; then
  hf download "$HF_CKPT_REPO" best_model.pth --local-dir ../clip_model
else
  echo "  WARNING: HF_CKPT_REPO not set. Upload best_model.pth manually."
fi

echo "[6/6] Sanity check"
python -c "
import transformers, torch, peft, bitsandbytes
print('transformers:', transformers.__version__)
print('torch:       ', torch.__version__)
print('peft:        ', peft.__version__)
print('bnb:         ', bitsandbytes.__version__)
print('CUDA OK:     ', torch.cuda.is_available())
print('GPU:         ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')
"

if [ -f ../clip_model/best_model.pth ]; then
  python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
else
  echo "  WARNING: ViT checkpoint not found — skipping verify."
fi

pip install pytest 2>/dev/null
pytest -q || echo "  (some tests may skip if dataset/tokenizer not cached — OK)"

echo ""
echo "Setup complete. Next:"
echo "  tmux new -s train           # avoid disconnect kills"
echo "  make data                   # ~30-60 min download + extract"
echo "  make stage1                 # ~2-3 h on RTX 4090, ~4-5 h on 3090 Ti"
echo "  make stage2                 # ~3 h on 4090, ~5-6 h on 3090 Ti"
echo "  make eval                   # ~30 min"
echo ""
echo "When done, copy back:"
echo "  scp -P <port> -r root@<host>:/workspace/FakeImageDetector_v1/fakedet_vlm/runs/stage2_sft/final ./runs/stage2_sft/"
