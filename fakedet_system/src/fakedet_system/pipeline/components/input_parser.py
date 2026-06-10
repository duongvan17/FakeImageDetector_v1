"""Stage 1 — parse the inbound OpenAI chat request.

Pulls the last user turn's text and (if any) the last attached image out of
the OpenAI ``messages`` array. The image is kept as its original
``data:image/...;base64,...`` URI so it can be forwarded to the model
backend untouched (the backend does its own preprocessing).
"""
from __future__ import annotations

from haystack import component


@component
class InputParser:
    @component.output_types(
        user_text=str,
        image_data_uri=str,
        has_image=bool,
        session_id=str,
    )
    def run(self, messages: list[dict], session_id: str = "default") -> dict:
        user_text = ""
        image_data_uri = ""

        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                user_text = content.strip()
            elif isinstance(content, list):
                texts: list[str] = []
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        texts.append(part.get("text", ""))
                    elif ptype == "image_url" and not image_data_uri:
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            image_data_uri = url
                user_text = " ".join(t.strip() for t in texts if t).strip()
            break  # only the most recent user turn

        return {
            "user_text": user_text,
            "image_data_uri": image_data_uri,
            "has_image": bool(image_data_uri),
            "session_id": session_id or "default",
        }
