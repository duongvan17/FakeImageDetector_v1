"""Load the user's fine-tuned ViT-B/16 checkpoint as a frozen feature extractor.

The checkpoint at ``clip_model/best_model.pth`` is a dict:
    {epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict,
     acc, auc, category_names}

with ``model_state_dict`` keys matching the timm ``vit_base_patch16_224`` layout
(cls_token, pos_embed, patch_embed.proj, blocks.{0..11}, norm, head). The head
is binary (1×768) and is dropped — we expose the 196 patch tokens for the VLM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required. Install with: pip install timm") from e


# CLIP-style normalization (matches the original ViT-B/16 CLIP pretraining stats
# the FakeClue classifier was likely fine-tuned from). If your encoder was
# trained with ImageNet stats instead, override via ``image_mean`` / ``image_std``.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VisionTowerViTB16(nn.Module):
    """Frozen ViT-B/16 returning the 196 patch tokens (CLS dropped).

    Output shape: ``(B, 196, 768)``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        image_size: int = 224,
        drop_cls: bool = True,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.drop_cls = drop_cls
        self.hidden_size = 768
        self.num_patches = (image_size // 16) ** 2  # 196 for 224

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = self._extract_state_dict(ckpt)
        # The fine-tuned checkpoint includes the binary ``head.{weight,bias}`` —
        # drop it because we build the model with ``num_classes=0``.
        state = {k: v for k, v in state.items() if not k.startswith("head.")}

        # Auto-detect the timm variant from checkpoint keys. The original
        # FakeClue classifier was fine-tuned from OpenAI CLIP-ViT-B/16, which
        # in timm corresponds to ``vit_base_patch16_clip_224``: it uses a
        # pre-norm (``norm_pre``) and a bias-less patch projection.
        has_norm_pre = "norm_pre.weight" in state
        has_patch_bias = "patch_embed.proj.bias" in state
        timm_name = (
            "vit_base_patch16_clip_224"
            if (has_norm_pre and not has_patch_bias)
            else "vit_base_patch16_224"
        )

        # ``num_classes=0`` removes the classification head entirely; we rely on
        # ``forward_features`` for tokens. ``global_pool=""`` keeps spatial dim.
        self.backbone = timm.create_model(
            timm_name,
            pretrained=False,
            num_classes=0,
            global_pool="",
            img_size=image_size,
        )

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Vision checkpoint mismatch (timm_name={timm_name}).\n"
                f"  missing: {missing}\n"
                f"  unexpected: {unexpected}"
            )

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> dict:
        if not isinstance(ckpt, dict):
            raise ValueError(f"Expected dict checkpoint, got {type(ckpt)}")
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Last resort: assume the dict itself is a state_dict (all values are tensors).
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise KeyError(
            f"No state_dict found in checkpoint. Top-level keys: {list(ckpt.keys())}"
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch tokens of shape (B, num_patches, 768)."""
        # ``forward_features`` returns the full token sequence (B, 1+N, D) for
        # a CLS-prefixed timm ViT.
        feats = self.backbone.forward_features(pixel_values)
        if feats.dim() != 3:
            raise RuntimeError(f"Unexpected feature shape {tuple(feats.shape)}")
        if self.drop_cls and feats.size(1) == self.num_patches + 1:
            feats = feats[:, 1:, :]
        return feats

    def train(self, mode: bool = True):  # noqa: D401
        # Always keep frozen backbone in eval mode (BN/dropout off).
        super().train(mode)
        self.backbone.eval()
        return self


def build_vision_tower(
    checkpoint_path: str | Path,
    image_size: int = 224,
    drop_cls: bool = True,
    strict: bool = True,
) -> VisionTowerViTB16:
    return VisionTowerViTB16(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        drop_cls=drop_cls,
        strict=strict,
    )


class VisionClassifier(nn.Module):
    """The ORIGINAL binary deepfake classifier from ``best_model.pth``.

    The VLM uses the ViT only as a frozen feature extractor (patch tokens)
    and *drops* this head. But the real/fake decision must come from this
    head — it is the 98.8%-acc / 0.999-AUC model the user trained — not
    from parsing the small LLM's free text. ``head`` is a single logit
    (768→1); ``sigmoid(logit)`` is the model's confidence for the positive
    class. Which class is "fake" was fixed at the original training time
    and is NOT recorded in the checkpoint, so it is left configurable
    (``fake_is_positive``) and must be sanity-checked on a known
    real + known fake image.
    """

    def __init__(self, checkpoint_path: str | Path, image_size: int = 224) -> None:
        super().__init__()
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = VisionTowerViTB16._extract_state_dict(ckpt)

        has_norm_pre = "norm_pre.weight" in state
        has_patch_bias = "patch_embed.proj.bias" in state
        timm_name = (
            "vit_base_patch16_clip_224"
            if (has_norm_pre and not has_patch_bias)
            else "vit_base_patch16_224"
        )
        # num_classes=1 keeps timm's exact train-time forward (token pool +
        # head) so the logit reproduces the original classifier.
        self.backbone = timm.create_model(
            timm_name, pretrained=False, num_classes=1, img_size=image_size
        )
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        # head.* MUST be present; anything else missing/unexpected is a bug.
        if any(not k.startswith("head.") for k in missing) or unexpected:
            raise RuntimeError(
                f"Classifier checkpoint mismatch (timm={timm_name}).\n"
                f"  missing: {missing}\n  unexpected: {unexpected}"
            )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def fake_prob(self, pixel_values: torch.Tensor, fake_is_positive: bool = True) -> float:
        logit = self.backbone(pixel_values).reshape(-1)[0]
        p_pos = torch.sigmoid(logit).item()
        return p_pos if fake_is_positive else (1.0 - p_pos)

    @torch.no_grad()
    def embed_image(self, pixel_values: torch.Tensor) -> list[float]:
        """Extract the 768-d CLS token representation for similarity search (RAG)."""
        feats = self.backbone.forward_features(pixel_values)
        # If it returns the full sequence (B, N, D), take the CLS token (index 0).
        # If timm already pooled it to (B, D), it will be 2D.
        if feats.dim() == 3:
            feats = feats[:, 0, :]
        return feats[0].cpu().tolist()

    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        self.backbone.eval()
        return self


def load_vision_classifier(
    checkpoint_path: str | Path, image_size: int = 224
) -> VisionClassifier:
    return VisionClassifier(checkpoint_path, image_size)
