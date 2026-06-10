"""OpenAI-compatible API for the FakeDet *system* (the Haystack pipeline).

Topology::

    open-webui ──► THIS (fakedet_system, local, no GPU)
                       │  intent  ─► external LLM API (Gemini/…)
                       │  memory  ─► SQLite (local)
                       └─ answer  ─► fakedet_vlm.serve (cloud GPU) ─► model

open-webui should point its OpenAI connection here (``/v1``), NOT directly
at the model backend. ``FAKEDET_BACKEND_URL`` (see config) is where THIS
service finds the model.

Run::

    uvicorn fakedet_system.serve.openai_api:app --host 0.0.0.0 --port 9000

Session id (so memory works per conversation): taken from the
``X-Session-Id`` header if the client sends one, else the OpenAI ``user``
field, else a hash of the first message — which keeps one open-webui
thread on one memory track without any client changes.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import get_settings
from ..pipeline import run_chat


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    user: str | None = None
    # accepted and ignored — behaviour is fixed by the pipeline
    temperature: float | None = None
    max_tokens: int | None = None


app = FastAPI(title="FakeDet System (OpenAI-compatible)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_id(req: ChatRequest, header_sid: str | None) -> str:
    if header_sid:
        return header_sid
    if req.user:
        return req.user
    for msg in req.messages:
        if msg.get("role") == "user":
            basis = json.dumps(msg.get("content"), sort_keys=True, default=str)
            return "auto-" + hashlib.sha1(basis.encode()).hexdigest()[:16]
    return "default"


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "intent_provider": s.intent_provider if s.external_enabled else "heuristic",
        "model_backend": s.model_backend_url,
    }


def _model_ids(s) -> list[str]:
    """Model ids surfaced to clients.

    v1 entries use the base ``served_model_id`` (e.g. ``fakedet-v1.2``).
    ``-colab`` variants only appear when their URL is set — open-webui hides
    options that don't resolve. We only expose v2 on Colab."""
    ids = [s.served_model_id]                       # fakedet-v1.2
    if s.colab_backend_url:
        ids.append(f"{s.served_model_id}-colab")    # fakedet-v1.2-colab
        ids.append(f"{s.served_model_id}-v2-colab") # fakedet-v1.2-v2-colab
    return ids


def _resolve_backend(model_id: str, s) -> tuple[str, str]:
    """Map the requested model id to (backend_url, backend_model_name).

    Suffix ``-colab`` routes to the Colab tunnel when one is configured;
    suffix ``-v2`` tells the backend to use the v2 vision classifier."""
    mid = (model_id or "").lower()
    use_colab = "colab" in mid and s.colab_backend_url
    use_v2 = "v2" in mid
    url = s.colab_backend_url if use_colab else s.model_backend_url
    backend_model = s.model_name_v2 if use_v2 else s.model_name
    return url, backend_model


@app.get("/v1/models")
def list_models() -> dict:
    s = get_settings()
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "fakedet"}
            for mid in _model_ids(s)
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatRequest,
    x_session_id: str | None = Header(default=None),
) -> dict:
    s = get_settings()
    session_id = _session_id(req, x_session_id)
    backend_url, backend_model = _resolve_backend(req.model or "", s)
    result = run_chat(
        req.messages, session_id,
        backend_url=backend_url,
        backend_model=backend_model,
    )

    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": now,
        "model": req.model or get_settings().served_model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["reply"]},
            "finish_reason": "stop",
        }],
        # surfaced for debugging / the thesis write-up; clients ignore it
        "fakedet_intent": result["intent"],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
