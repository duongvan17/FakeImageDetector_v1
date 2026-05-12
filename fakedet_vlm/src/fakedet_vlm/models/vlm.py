"""Composed VLM: frozen ViT-B/16 + trainable projector + Qwen2.5 LLM (4-bit + LoRA).

Visual injection contract
-------------------------
Each training/inference sample contains exactly ``N = num_visual_tokens``
consecutive ``image_token_id``\\ s in ``input_ids`` (inserted by the dataset).
The forward pass:

  1. Computes patch features from the vision tower → ``(B, N, vision_dim)``.
  2. Projects to LLM space → ``(B, N, llm_dim)``.
  3. Embeds ``input_ids`` via ``embed_tokens`` → ``(B, L, llm_dim)``.
  4. Replaces the embeddings at the ``N`` image-token positions with the
     projected visual embeddings using ``masked_scatter`` (gradient-safe).
  5. Feeds the merged sequence to the LLM with the original attention mask.

The replacement is done with ``masked_scatter`` (not in-place index assignment)
so gradients flow back through the projector during training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .projector import ProjectorMLP
from .vit_loader import build_vision_tower


IMAGE_TOKEN = "<image>"


def build_tokenizer(llm_name: str) -> AutoTokenizer:
    """Load Qwen2.5 tokenizer and ensure ``<image>`` is a special token."""
    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if IMAGE_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    return tok


class FakeDetVLM(nn.Module):
    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        vision_checkpoint: str | Path = "../clip_model/best_model.pth",
        vision_dim: int = 768,
        num_visual_tokens: int = 196,
        image_size: int = 224,
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.llm_name = llm_name
        self.vision_dim = vision_dim
        self.num_visual_tokens = num_visual_tokens

        self.tokenizer = build_tokenizer(llm_name)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # 1) Vision tower (frozen, fp32 on CPU initially — moved by accelerate later).
        self.vision_tower = build_vision_tower(
            checkpoint_path=vision_checkpoint,
            image_size=image_size,
        )

        # 2) LLM (4-bit by default for 12GB VRAM).
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map={"": 0} if torch.cuda.is_available() else None,
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # Resize embeddings for the new <image> special token. The embedding
        # table itself is fp16/bf16 (not 4-bit) under bitsandbytes, so this is
        # safe; ``mean_resizing=True`` initialises new rows from the mean of
        # existing embeddings rather than random noise.
        new_vocab = len(self.tokenizer)
        if self.llm.get_input_embeddings().weight.size(0) != new_vocab:
            self.llm.resize_token_embeddings(new_vocab, mean_resizing=True)

        self.llm_dim = self.llm.config.hidden_size

        # 3) Projector — keep trainable in fp32 for stable gradients, then cast
        #    its outputs to the LLM compute dtype before merging.
        self.projector = ProjectorMLP(vision_dim=vision_dim, llm_dim=self.llm_dim)
        self._compute_dtype = torch_dtype

    # ------------------------------------------------------------------ utils
    def freeze_vision(self) -> None:
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()

    def freeze_llm(self) -> None:
        for p in self.llm.parameters():
            p.requires_grad = False

    def unfreeze_projector(self) -> None:
        for p in self.projector.parameters():
            p.requires_grad = True

    # ---------------------------------------------------- HF Trainer hooks
    # HF Trainer calls these on ``self.model`` directly. Because FakeDetVLM
    # is a plain nn.Module (not a PreTrainedModel), we forward to the wrapped
    # LLM where the real implementation lives.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.llm.gradient_checkpointing_enable()
            else:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
        # Also ensure gradients flow through the (frozen) embedding to the
        # projector inputs — required when the LLM is quantised + checkpointed.
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.llm, "is_gradient_checkpointing", False))

    def enable_input_require_grads(self):
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def trainable_param_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {"projector": [], "llm_lora": [], "other": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("projector."):
                groups["projector"].append(p)
            elif "lora_" in n:
                groups["llm_lora"].append(p)
            else:
                groups["other"].append(p)
        return groups

    # ----------------------------------------------------------------- forward
    def _merge_visual(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``inputs_embeds`` of shape (B, L, llm_dim) with image tokens
        replaced by projected vision features."""
        # Vision features: (B, N, vision_dim). Cast to compute dtype before
        # projecting so the projector returns a tensor compatible with the LLM.
        with torch.no_grad():
            vision_feats = self.vision_tower(pixel_values)
        vision_feats = vision_feats.to(self.projector.fc1.weight.dtype)
        proj = self.projector(vision_feats)  # (B, N, llm_dim) in fp32

        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, D)
        proj = proj.to(text_embeds.dtype)

        image_mask = (input_ids == self.image_token_id)  # (B, L) bool
        n_expected = image_mask.sum(dim=1)
        if not torch.all(n_expected == self.num_visual_tokens):
            raise RuntimeError(
                f"Each sample must contain exactly {self.num_visual_tokens} "
                f"image tokens; got counts {n_expected.tolist()}"
            )

        # masked_scatter is differentiable w.r.t. the source tensor. Source must
        # be flat-or-broadcastable; we flatten the (B, N, D) projector output to
        # (B*N, D) and let masked_scatter pull rows in scan order — which lines
        # up with the row-major positions of True entries in image_mask.
        flat_proj = proj.reshape(-1, proj.size(-1))
        merged = text_embeds.masked_scatter(
            image_mask.unsqueeze(-1),
            flat_proj,
        )
        return merged

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        inputs_embeds = self._merge_visual(input_ids, pixel_values)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 128,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        inputs_embeds = self._merge_visual(input_ids, pixel_values)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
