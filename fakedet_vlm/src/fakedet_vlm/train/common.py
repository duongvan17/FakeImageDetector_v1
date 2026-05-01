"""Shared training helpers."""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import Trainer


def load_config(stage_yaml: str | Path, base_yaml: str | Path) -> dict:
    with open(base_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(stage_yaml, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f))
    return cfg


def torch_dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


class VLMTrainer(Trainer):
    """Trainer wrapper that forwards our custom kwargs to ``model.forward``.

    The HF Trainer normally calls ``model(**inputs)``; our forward already
    accepts (input_ids, attention_mask, pixel_values, labels) so we just need
    to override ``compute_loss`` to make sure the return type is consistent.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict=None):
        # Override to also persist the projector regardless of LoRA wrapping.
        super()._save(output_dir, state_dict)
        if output_dir is None:
            return
        proj = getattr(self.model, "projector", None)
        if proj is not None:
            torch.save(proj.state_dict(), Path(output_dir) / "projector.pt")
