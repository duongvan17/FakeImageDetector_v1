from __future__ import annotations

from haystack import component

from ...config import (
    INTENT_IMAGE_DEEPFAKE,
    INTENT_MISLEADING,
    INTENT_USER_GUIDE,
)
from ...prompts import detect_language, prompts


def _history_messages(history: list) -> list[dict]:
    out: list[dict] = []
    for t in history:
        if t.get("role") in ("user", "assistant") and t.get("content"):
            out.append({"role": t["role"], "content": t["content"]})
    return out


@component
class PromptBuilder:
    @component.output_types(plan=dict)
    def run(
        self,
        user_text: str,
        intent: str,
        has_image: bool,
        image_data_uri: str,
        history: list,
        core_memory: str,
        last_image: str,
    ) -> dict:
        resp = prompts().get("response", {})
        language = detect_language(user_text)

        if intent == INTENT_IMAGE_DEEPFAKE or has_image:
            return {"plan": {
                "route": "model",
                "intent": INTENT_IMAGE_DEEPFAKE,
                "language": language,
                "image": image_data_uri,   # carried for the Gemini polish step
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text or resp.get("image_instruction", "")},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }],
            }}

        if intent == INTENT_MISLEADING:
            return {"plan": {
                "route": "static",
                "intent": intent,
                "language": language,
                "static_reply": resp.get("scope_notice", ""),
            }}

        if intent == INTENT_USER_GUIDE:
            system = resp.get("guide_system", "").format(language=language)
            image = ""
        else:  # text_to_detect
            system = resp.get("analyst_system", "").format(language=language)
            image = last_image  # let the big model re-look at the prev image
        if core_memory:
            system += f"\n\nKnown context about this session:\n{core_memory}"

        messages = _history_messages(history)
        messages.append({"role": "user", "content": user_text})
        return {"plan": {
            "route": "gemini",
            "intent": intent,
            "language": language,
            "system": system,
            "messages": messages,
            "image": image,
        }}
