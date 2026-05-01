"""Lazy re-exports — keep imports cheap.

Importing :mod:`vlm` pulls in transformers, peft and bitsandbytes. The
sanity-check scripts (``verify_vit_loads.py``) only need the vision tower and
projector, so we expose those eagerly and load :class:`FakeDetVLM` on demand.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .vit_loader import VisionTowerViTB16, build_vision_tower
from .projector import ProjectorMLP

if TYPE_CHECKING:  # pragma: no cover
    from .vlm import FakeDetVLM


def __getattr__(name: str):
    if name == "FakeDetVLM":
        from .vlm import FakeDetVLM as _FakeDetVLM
        return _FakeDetVLM
    raise AttributeError(f"module 'fakedet_vlm.models' has no attribute {name!r}")


__all__ = ["VisionTowerViTB16", "build_vision_tower", "ProjectorMLP", "FakeDetVLM"]
