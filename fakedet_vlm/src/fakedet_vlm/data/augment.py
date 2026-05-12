"""Deepfake-aware augmentation.

Detection models that learn texture / GAN artefacts are notoriously fragile
under JPEG compression and resampling that occur in the wild. This module
exposes a small, opinionated augmentation pipeline applied to the PIL image
*before* the standard ToTensor + Normalize stack.

Defaults are intentionally mild — strong geometric distortion can erase the
pixel-level forensic signal we want the model to use.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image
from torchvision import transforms


@dataclass
class DeepfakeAugment:
    """Callable applied to PIL → PIL."""

    jpeg_prob: float = 0.5
    jpeg_quality_range: tuple[int, int] = (60, 95)
    resize_jitter_prob: float = 0.3
    resize_jitter_range: tuple[float, float] = (0.85, 1.15)
    color_jitter_prob: float = 0.3
    horizontal_flip_prob: float = 0.0   # off by default — face-deepfake parity
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._color = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self._rng.random() < self.resize_jitter_prob:
            scale = self._rng.uniform(*self.resize_jitter_range)
            w, h = image.size
            new_size = (max(32, int(w * scale)), max(32, int(h * scale)))
            image = image.resize(new_size, Image.BICUBIC)

        if self._rng.random() < self.color_jitter_prob:
            image = self._color(image)

        if self._rng.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self._rng.random() < self.jpeg_prob:
            quality = self._rng.randint(*self.jpeg_quality_range)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        return image
