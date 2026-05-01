from __future__ import annotations

import torch

from fakedet_vlm.models import ProjectorMLP


def test_shape_and_grad():
    proj = ProjectorMLP(vision_dim=768, llm_dim=1536)
    x = torch.randn(2, 196, 768, requires_grad=True)
    y = proj(x)
    assert y.shape == (2, 196, 1536)
    y.sum().backward()
    assert x.grad is not None
    # Projector params receive gradient.
    for p in proj.parameters():
        assert p.grad is not None
