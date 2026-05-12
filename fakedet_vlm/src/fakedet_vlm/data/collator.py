"""Right-padding collator for ``FakeClueDataset`` items."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VLMCollator:
    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].size(0) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_len), self.label_pad_id, dtype=torch.long
        )
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)

        for i, f in enumerate(features):
            n = f["input_ids"].size(0)
            input_ids[i, :n] = f["input_ids"]
            attention_mask[i, :n] = 1
            labels[i, :n] = f["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
