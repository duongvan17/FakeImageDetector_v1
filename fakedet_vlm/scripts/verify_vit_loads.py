"""Standalone sanity check: load the user's ViT-B/16 checkpoint and dump the
shape of the patch features for one dummy image.

Run this BEFORE attempting any training. Requires only torch + timm + Pillow.

    python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running without ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fakedet_vlm.models import build_vision_tower  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="../clip_model/best_model.pth")
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    print(f"[1/3] Loading ViT-B/16 from {args.ckpt} ...")
    tower = build_vision_tower(args.ckpt, image_size=args.image_size, strict=True)
    n_params = sum(p.numel() for p in tower.parameters())
    print(f"      OK — {n_params/1e6:.1f}M params, frozen={all(not p.requires_grad for p in tower.parameters())}")

    print("[2/3] Forward on dummy (2, 3, 224, 224) ...")
    x = torch.randn(2, 3, args.image_size, args.image_size)
    feats = tower(x)
    print(f"      output shape = {tuple(feats.shape)}  (expected: (2, 196, 768))")
    assert feats.shape == (2, tower.num_patches, tower.hidden_size), "shape mismatch"

    print("[3/3] Range / norm sanity:")
    print(f"      min={feats.min().item():.3f} max={feats.max().item():.3f} "
          f"mean={feats.mean().item():.3f} std={feats.std().item():.3f}")
    print("\nOK — vision tower loads and runs correctly.")


if __name__ == "__main__":
    main()
