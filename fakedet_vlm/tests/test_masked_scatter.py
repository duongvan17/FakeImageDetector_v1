"""Verify that ``torch.Tensor.masked_scatter`` populates rows in scan order
exactly the way ``FakeDetVLM._merge_visual`` relies on.

This test catches the most common source of silent bugs in VLM forward
implementations: visual features ending up at the wrong sequence positions.
"""
from __future__ import annotations

import torch


def test_masked_scatter_row_order_matches_image_token_positions():
    B, L, D, N = 2, 10, 4, 3
    text_embeds = torch.zeros(B, L, D)

    # Image tokens at positions [2,3,4] in row 0 and [5,6,7] in row 1.
    image_mask = torch.zeros(B, L, dtype=torch.bool)
    image_mask[0, 2:5] = True
    image_mask[1, 5:8] = True

    # Distinct values per (batch, slot) so we can verify positional mapping.
    proj = torch.arange(B * N * D, dtype=torch.float).view(B, N, D)

    merged = text_embeds.masked_scatter(image_mask.unsqueeze(-1), proj.reshape(-1, D))

    # Row 0 slots 2..4 should equal proj[0, 0..2]
    torch.testing.assert_close(merged[0, 2:5], proj[0])
    # Row 1 slots 5..7 should equal proj[1, 0..2]
    torch.testing.assert_close(merged[1, 5:8], proj[1])
    # Untouched positions stay zero.
    assert (merged[0, :2] == 0).all() and (merged[0, 5:] == 0).all()
    assert (merged[1, :5] == 0).all() and (merged[1, 8:] == 0).all()


def test_masked_scatter_is_differentiable():
    B, L, D, N = 1, 6, 4, 2
    text_embeds = torch.zeros(B, L, D)
    image_mask = torch.zeros(B, L, dtype=torch.bool)
    image_mask[0, 1:3] = True

    proj = torch.randn(B, N, D, requires_grad=True)
    merged = text_embeds.masked_scatter(image_mask.unsqueeze(-1), proj.reshape(-1, D))
    merged.sum().backward()

    # Gradient flows back to projector output (non-zero on selected rows).
    assert proj.grad is not None
    assert (proj.grad != 0).any()
