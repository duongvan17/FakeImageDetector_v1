"""FakeClue dataset adapter producing samples ready for ``VLMCollator``.

Each item returns a *pre-tokenized* dict with prompt-vs-response split kept
separate so the collator can construct ``labels`` with -100 over the prompt
without doing brittle substring matching.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .prompts import IMAGE_PLACEHOLDER, build_chat_prompt, format_assistant_response


class FakeClueDataset(Dataset):
    """Loads either:
      - a HuggingFace ``datasets`` arrow split (call ``from_hf`` factory), or
      - a JSON file in LLaVA-style format produced by ``scripts/prepare_fakeclue.py``.

    Image preprocessing matches the ViT-B/16 vision tower (224×224, CLIP stats).
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        images_dir: str | Path | None,
        tokenizer,
        num_visual_tokens: int = 196,
        image_token: str = IMAGE_PLACEHOLDER,
        image_size: int = 224,
        image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        max_length: int = 1024,
        augment: Any | None = None,
    ) -> None:
        self.records = records
        self.images_dir = Path(images_dir) if images_dir else None
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens
        self.image_token = image_token
        self.max_length = max_length
        # ``augment`` is any PIL→PIL callable (e.g. ``DeepfakeAugment``). Train
        # splits should pass one; eval/inference must leave it ``None``.
        self.augment = augment

        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer does not know special token {image_token!r}. "
                "Add it via tokenizer.add_special_tokens() before constructing the dataset."
            )
        self.image_token_id = image_token_id

        self.image_processor = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(image_mean), std=list(image_std)),
            ]
        )

    @classmethod
    def from_json(cls, json_path: str | Path, images_dir: str | Path, **kwargs):
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return cls(records=records, images_dir=images_dir, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec: dict) -> Image.Image:
        if "image" in rec and isinstance(rec["image"], str):
            if self.images_dir is None:
                raise ValueError("images_dir must be set for JSON records with string paths")
            img = Image.open(self.images_dir / rec["image"])
        elif "image_pil" in rec:
            img = rec["image_pil"]
        else:
            raise KeyError(f"No image found in record keys: {list(rec.keys())}")
        return img.convert("RGB")

    def _build_response(self, rec: dict) -> str:
        # Prefer pre-formatted conversation if present (LLaVA JSON format).
        convs = rec.get("conversations")
        if convs and isinstance(convs, list):
            for turn in convs:
                if turn.get("from") == "gpt":
                    return turn["value"]
        # Otherwise build from raw FakeClue fields.
        return format_assistant_response(rec.get("label", 1), rec.get("clue"))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]

        image = self._load_image(rec)
        if self.augment is not None:
            image = self.augment(image)
        pixel_values = self.image_processor(image)

        response_text = self._build_response(rec)
        prefix, suffix = build_chat_prompt(assistant_response=response_text)

        # Expand the single ``<image>`` placeholder into N copies of the
        # special token. This produces N consecutive image_token_ids in
        # input_ids — the VLM forward replaces each with one patch embedding.
        expanded_image = self.image_token * self.num_visual_tokens
        prefix = prefix.replace(self.image_token, expanded_image, 1)

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]

        input_ids = prefix_ids + suffix_ids
        labels = [-100] * len(prefix_ids) + list(suffix_ids)

        # Truncate from the right (preserve prefix + image tokens); training
        # samples that overflow lose the tail of the response — fine for SFT.
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,  # (3, H, W)
        }
