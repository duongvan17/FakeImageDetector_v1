"""Unit tests for VisionTowerViTB16.

Skipped automatically if the actual checkpoint is not on disk so the suite
runs in CI without needing the 1GB weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from fakedet_vlm.models import build_vision_tower

CKPT = Path(__file__).resolve().parents[2] / "clip_model" / "best_model.pth"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason=f"missing {CKPT}")


def test_loads_and_freezes():
    tower = build_vision_tower(CKPT)
    assert tower.hidden_size == 768
    assert tower.num_patches == 196
    assert all(not p.requires_grad for p in tower.parameters())


def test_forward_shape():
    tower = build_vision_tower(CKPT)
    x = torch.randn(3, 3, 224, 224)
    out = tower(x)
    assert out.shape == (3, 196, 768)
    assert out.dtype == torch.float32


def test_train_mode_keeps_eval():
    tower = build_vision_tower(CKPT)
    tower.train()  # composed model goes to train; backbone must stay eval
    assert not tower.backbone.training
