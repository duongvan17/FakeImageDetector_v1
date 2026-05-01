"""Integration test that exercises the *full* visual-injection path on CPU.

This test does NOT need transformers or pretrained LLM weights — it stitches
together the real :class:`VisionTowerViTB16` (loaded from your checkpoint),
the real :class:`ProjectorMLP`, and a *tiny* dummy LLM that mirrors the
HuggingFace ``CausalLM.forward(inputs_embeds=..., labels=...)`` contract. It
catches dtype mismatches, gradient-flow regressions, and shape errors that
``test_masked_scatter`` is too synthetic to surface.

Heavy imports (transformers, peft, bitsandbytes) are intentionally avoided —
the suite runs in a few seconds on CPU.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fakedet_vlm.models import ProjectorMLP, build_vision_tower

CKPT = Path(__file__).resolve().parents[2] / "clip_model" / "best_model.pth"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason=f"missing {CKPT}")


class _StubCausalLM(nn.Module):
    """Drop-in for the LLM that exposes the embedding interface and a tiny
    decoder so ``inputs_embeds → logits`` is well-defined."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.decoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, batch_first=True, dim_feedforward=128
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor | None = None):
        # Use attention_mask to build a key-padding mask (True = pad).
        kpm = attention_mask == 0
        h = self.decoder(inputs_embeds, src_key_padding_mask=kpm)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        class _Out:
            pass
        out = _Out()
        out.loss = loss
        out.logits = logits
        return out


def _merge_visual(text_embeds, proj, image_mask, num_visual_tokens):
    """Replica of FakeDetVLM._merge_visual without the LLM dependency."""
    flat_proj = proj.reshape(-1, proj.size(-1))
    return text_embeds.masked_scatter(image_mask.unsqueeze(-1), flat_proj)


def _build_pieces(num_visual_tokens=196, hidden=64, vocab=300):
    tower = build_vision_tower(CKPT)
    projector = ProjectorMLP(vision_dim=768, llm_dim=hidden)
    llm = _StubCausalLM(vocab_size=vocab, hidden_size=hidden)
    image_token_id = vocab - 1  # reserved
    return tower, projector, llm, image_token_id


def test_full_forward_with_loss_backward():
    torch.manual_seed(0)
    tower, projector, llm, image_token_id = _build_pieces(num_visual_tokens=196)
    B, L = 2, 220
    # Build an input where 196 consecutive positions per row are image tokens.
    input_ids = torch.randint(low=0, high=image_token_id, size=(B, L))
    input_ids[0, 5:201] = image_token_id
    input_ids[1, 10:206] = image_token_id

    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == image_token_id] = -100  # never predict on visual slots

    pixel_values = torch.randn(B, 3, 224, 224)

    # Forward
    text_embeds = llm.get_input_embeddings()(input_ids)
    vision_feats = tower(pixel_values)
    assert vision_feats.shape == (B, 196, 768)
    proj = projector(vision_feats)
    assert proj.shape == (B, 196, 64)
    merged = _merge_visual(text_embeds, proj, input_ids == image_token_id, 196)
    assert merged.shape == text_embeds.shape

    out = llm(inputs_embeds=merged, attention_mask=attention_mask, labels=labels)
    assert out.loss is not None
    assert out.logits.shape == (B, L, 300)
    assert torch.isfinite(out.loss)

    # Backward — gradients must reach the projector.
    out.loss.backward()
    grads = [p.grad for p in projector.parameters()]
    assert all(g is not None for g in grads)
    assert any((g.abs().sum() > 0).item() for g in grads), "projector got no signal"

    # ViT must remain frozen.
    assert all(p.grad is None for p in tower.parameters())


def test_dtype_promotion_to_llm_dtype():
    """Projector outputs in fp32 must be cast cleanly to fp16 LLM embeds."""
    torch.manual_seed(0)
    tower, projector, llm, image_token_id = _build_pieces(num_visual_tokens=196)
    # Switch stub LLM embeddings to fp16 to mimic bf16/fp16 LLM weights.
    llm.embed_tokens = llm.embed_tokens.to(torch.float16)

    # 198 total: 2 real + 196 image tokens — must match num_visual_tokens=196.
    input_ids = torch.full((1, 198), image_token_id, dtype=torch.long)
    input_ids[0, :2] = 1
    pixel_values = torch.randn(1, 3, 224, 224)

    text_embeds = llm.get_input_embeddings()(input_ids)
    assert text_embeds.dtype == torch.float16

    vision_feats = tower(pixel_values)
    proj = projector(vision_feats).to(text_embeds.dtype)
    merged = _merge_visual(text_embeds, proj, input_ids == image_token_id, 196)
    assert merged.dtype == torch.float16
    assert torch.isfinite(merged).all()
