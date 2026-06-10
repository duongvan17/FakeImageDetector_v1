"""Loader for ``prompts.yaml``.

All LLM-facing prompts and intent heuristic keywords live in YAML so they
can be tuned without touching code. The file is loaded once per process
(``lru_cache``); override the path with ``FAKEDET_PROMPTS_FILE`` for tests
or A/B experiments.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).parent / "prompts.yaml"

# Crude language detector — enough to pick a value for the {language}
# placeholder. Any Vietnamese-tone character → Vietnamese; otherwise
# English (the model is multilingual anyway, this is a hint).
_VI_PATTERN = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    return "Vietnamese" if text and _VI_PATTERN.search(text) else "English"


@lru_cache(maxsize=1)
def prompts() -> dict[str, Any]:
    path = Path(os.environ.get("FAKEDET_PROMPTS_FILE") or _DEFAULT_PATH)
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def reload_prompts() -> None:
    """Drop the cached prompts (useful in tests that point at a temp file)."""
    prompts.cache_clear()
