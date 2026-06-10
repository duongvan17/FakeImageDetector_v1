
from __future__ import annotations

import httpx
from haystack import component

from ...config import Settings
from ...prompts import prompts
from ..external import ExternalLLM


from .rag_store import FeedbackRAGStore

def _image_uri_from_messages(messages: list) -> str:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return part.get("image_url", {}).get("url", "")
    return ""


def _split_verdict(reply: str) -> tuple[str, str]:
    
    r = (reply or "").strip()
    if r.startswith("**") and "\n\n" in r:
        head, body = r.split("\n\n", 1)
        if head.endswith("**"):
            return head, body.strip()
    return "", r


@component
class Responder:
    def __init__(self, settings: Settings, rag_store: FeedbackRAGStore | None = None) -> None:
        self._s = settings
        self._llm = ExternalLLM(settings)
        self._rag_store = rag_store

    @component.output_types(reply=str)
    def run(self, plan: dict, backend_url: str = "", backend_model: str = "") -> dict:
        """``backend_url`` and ``backend_model`` are optional per-request
        overrides (open-webui passes the chosen model's URL and backend
        model name via ``run_chat``). Empty → use defaults from settings."""
        route = plan.get("route")
        if route == "static":
            return {"reply": plan.get("static_reply", "")}
        if route == "model":
            return {"reply": self._image_verdict(plan, backend_url, backend_model)}

        # route == "gemini"
        answer = self._llm.generate(
            plan.get("system", ""),
            plan.get("messages", []),
            plan.get("image") or None,
        )
        if answer:
            return {"reply": answer}
        
        sys = plan.get("system", "")
        msgs = list(plan.get("messages", []))
        if sys:
            msgs = [{"role": "system", "content": sys}, *msgs]
        return {"reply": self._call_backend(msgs, backend_url, backend_model)}

    def _image_verdict(self, plan: dict, backend_url: str, backend_model: str) -> str:
       
        messages = plan["messages"]
        raw = self._call_backend(messages, backend_url, backend_model)
        header, body = _split_verdict(raw)

        
        image = plan.get("image") or _image_uri_from_messages(messages)
        
        # --- RAG ---
        rag_context = ""
        if self._rag_store and image:
            b64 = image.split(",", 1)[1] if "," in image else image
            b_url = backend_url or self._s.model_backend_url
            b_model = backend_model or self._s.model_name
            docs = self._rag_store.search_similar(b64, b_url, b_model, k=2)
            if docs:
                rag_context = "PAST HUMAN FEEDBACK ON VISUALLY SIMILAR IMAGES:\n- " + "\n- ".join(docs) + "\n\nUse this feedback to adjust or improve the explanation if relevant to THIS image."

        polish_system = prompts().get("polish", {}).get("system", "").format(
            verdict=header.strip("* ") or "the verdict above",
            language=plan.get("language") or "Vietnamese",
            rag_context=rag_context
        )
        polished = self._llm.generate(
            polish_system,
            [{"role": "user", "content": f"Draft from the captioner:\n{body}"}],
            image or None,
        )
        explanation = polished.strip() if polished else body
        return f"{header}\n\n{explanation}" if header else explanation

    def _call_backend(self, messages: list, backend_url: str = "", backend_model: str = "") -> str:
        base = backend_url or self._s.model_backend_url
        model_name = backend_model or self._s.model_name
        url = f"{base.rstrip('/')}/chat/completions"
        try:
            r = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self._s.model_backend_key}"},
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                },
                timeout=self._s.model_timeout_s,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            return (
                "⚠️ Không gọi được model backend "
                f"({type(e).__name__}). Kiểm tra model server đã chạy chưa và "
                f"FAKEDET_BACKEND_URL đang trỏ đúng "
                f"(hiện tại: {base})."
            )
        except (KeyError, IndexError, ValueError):
            return "⚠️ Model backend trả về response không hợp lệ."
