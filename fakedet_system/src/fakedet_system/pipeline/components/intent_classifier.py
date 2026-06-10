"""Stage 2 — intent classification via the EXTERNAL big model.

Rules:

* image attached  -> ``image_deepfake`` (decided here, no API call).
* otherwise        -> the external model picks one of
  ``text_to_detect`` | ``user_guide`` | ``misleading``.
* no API key / API fails -> transparent keyword heuristic, defaulting to
  ``misleading`` (out of scope) when nothing in-scope is detected — so a
  random "what's the weather" is NOT mistaken for a detection request.
"""
from __future__ import annotations

from haystack import component

from ...config import (
    INTENT_IMAGE_DEEPFAKE,
    INTENT_MISLEADING,
    INTENT_TEXT_TO_DETECT,
    INTENT_USER_GUIDE,
    INTENT_USER_FEEDBACK,
    TEXT_INTENTS,
    Settings,
)
from ...prompts import prompts
from ..external import ExternalLLM


def _normalise(raw: str) -> str | None:
    low = raw.strip().lower()
    for label in TEXT_INTENTS:
        if label in low:
            return label
    return None


@component
class IntentClassifier:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._llm = ExternalLLM(settings)

    @component.output_types(intent=str, intent_source=str)
    def run(self, user_text: str, has_image: bool) -> dict:
        if has_image:
            return {"intent": INTENT_IMAGE_DEEPFAKE, "intent_source": "rule"}

        cfg = prompts().get("intent", {})
        if self._llm.enabled:
            raw = self._llm.classify(cfg.get("classifier_system", ""), user_text)
            label = _normalise(raw) if raw else None
            if label is not None:
                return {"intent": label, "intent_source": self._s.intent_provider}

        return {"intent": self._heuristic(user_text, cfg), "intent_source": "heuristic"}

    def _heuristic(self, text: str, cfg: dict) -> str:
        low = text.lower()
        if any(k in low for k in cfg.get("feedback_keywords", [])):
            return INTENT_USER_FEEDBACK
        if any(k in low for k in cfg.get("guide_keywords", [])):
            return INTENT_USER_GUIDE
        if any(k in low for k in cfg.get("detect_keywords", [])):
            return INTENT_TEXT_TO_DETECT
        # Nothing in-scope detected -> treat as out of scope.
        return INTENT_MISLEADING
