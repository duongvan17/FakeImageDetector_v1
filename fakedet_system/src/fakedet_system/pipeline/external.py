"""Thin client for the external big model (Gemini by default, or any
OpenAI-compatible endpoint). Used in two places:

* intent classification (short, ``classify``)
* answering the ``text_to_detect`` / ``user_guide`` intents with context
  and optionally a re-look at the previous image (``generate``)

Every method returns ``None`` on any failure (no key, HTTP error,
unexpected JSON) so callers can fall back gracefully — the system never
crashes because the external API is unavailable.
"""
from __future__ import annotations

import httpx

from ..config import Settings


def _split_data_uri(uri: str) -> tuple[str, str] | None:
    """``data:image/png;base64,AAAA`` -> ``("image/png", "AAAA")``."""
    if not uri.startswith("data:") or ";base64," not in uri:
        return None
    head, b64 = uri.split(";base64,", 1)
    mime = head[len("data:"):] or "image/png"
    return mime, b64


class ExternalLLM:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    @property
    def enabled(self) -> bool:
        return self._s.external_enabled

    # ------------------------------------------------------------- classify
    def classify(self, system: str, text: str) -> str | None:
        if not self.enabled:
            return None
        try:
            if self._s.intent_provider == "gemini":
                return self._gemini(system, [("user", text)], None, 16)
            return self._openai(system, [("user", text)], None, 16)
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return None

    # ------------------------------------------------------------- generate
    def generate(
        self,
        system: str,
        messages: list[dict],
        image_data_uri: str | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        turns = [
            (m.get("role", "user"), str(m.get("content", "")))
            for m in messages
            if m.get("content")
        ]
        try:
            if self._s.intent_provider == "gemini":
                return self._gemini(
                    system, turns, image_data_uri, self._s.external_max_tokens
                )
            return self._openai(
                system, turns, image_data_uri, self._s.external_max_tokens
            )
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return None

    # --------------------------------------------------------------- gemini
    def _gemini(
        self,
        system: str,
        turns: list[tuple[str, str]],
        image_data_uri: str | None,
        max_tokens: int,
    ) -> str:
        contents: list[dict] = []
        for role, text in turns:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            })
        if image_data_uri and contents:
            parsed = _split_data_uri(image_data_uri)
            if parsed:
                mime, b64 = parsed
                contents[-1]["parts"].append(
                    {"inline_data": {"mime_type": mime, "data": b64}}
                )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._s.intent_model}:generateContent"
        )
        gen_cfg: dict = {"temperature": 0.0, "maxOutputTokens": max_tokens}
        # Gemini 2.5 models "think" by default and would burn the whole
        # token budget on hidden reasoning, returning a candidate with no
        # text parts. Disable thinking so short answers actually come back.
        if "2.5" in self._s.intent_model:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        r = httpx.post(
            url,
            params={"key": self._s.intent_api_key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": gen_cfg,
            },
            timeout=self._s.intent_timeout_s,
        )
        r.raise_for_status()
        cand = (r.json().get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    # --------------------------------------------------------------- openai
    def _openai(
        self,
        system: str,
        turns: list[tuple[str, str]],
        image_data_uri: str | None,
        max_tokens: int,
    ) -> str:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for i, (role, text) in enumerate(turns):
            last = i == len(turns) - 1
            if last and image_data_uri and role == "user":
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ]})
            else:
                msgs.append({"role": role, "content": text})
        r = httpx.post(
            f"{self._s.intent_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self._s.intent_api_key}"},
            json={
                "model": self._s.intent_model,
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
            timeout=self._s.intent_timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
