from __future__ import annotations

from PIL import Image

from fakedet_vlm.data import DeepfakeAugment


def test_augment_returns_rgb_pil():
    aug = DeepfakeAugment(seed=0,
                           jpeg_prob=1.0, resize_jitter_prob=1.0, color_jitter_prob=1.0)
    img = Image.new("RGB", (256, 200), (200, 100, 50))
    out = aug(img)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size[0] >= 32 and out.size[1] >= 32


def test_augment_deterministic_with_seed():
    a = DeepfakeAugment(seed=123)
    b = DeepfakeAugment(seed=123)
    img = Image.new("RGB", (224, 224), (10, 20, 30))
    pa = a(img.copy()).tobytes()
    pb = b(img.copy()).tobytes()
    assert pa == pb
