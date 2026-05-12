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
        """Custom checkpoint that bypasses HF Trainer's default model dump.

        Why bypass?  HF's default tries to serialise the whole VLM
        (LLM + vision tower + projector) with ``safetensors``.  Qwen2.5 ties
        ``embed_tokens`` and ``lm_head`` weights, and safetensors refuses to
        save shared-memory tensors:

            RuntimeError: Some tensors share memory, this will lead to
            duplicate memory on disk ...

        We only need to persist what is trainable anyway:
          - projector.pt (always)
          - LoRA adapter via ``llm.save_pretrained`` (stage 2 only — present
            when the LLM is wrapped by PEFT)
          - tokenizer (lightweight, useful for downstream inference)

        The base LLM weights and the frozen ViT are never touched by training,
        so re-loading them at inference time from their original sources is
        fine and saves disk + dodges the tied-weights bug entirely.
        """
        if output_dir is None:
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        proj = getattr(self.model, "projector", None)
        if proj is not None:
            torch.save(proj.state_dict(), out / "projector.pt")

        llm = getattr(self.model, "llm", None)
        # PEFT-wrapped models expose ``peft_config`` and ``save_pretrained``.
        if llm is not None and hasattr(llm, "peft_config") and hasattr(llm, "save_pretrained"):
            llm.save_pretrained(str(out))

        tok = getattr(self.model, "tokenizer", None)
        if tok is not None:
            try:
                tok.save_pretrained(str(out))
            except Exception:  # noqa: BLE001
                pass  # tokenizer save is best-effort, not critical

        # Persist the training arguments so resume picks them up.
        torch.save(self.args, out / "training_args.bin")
