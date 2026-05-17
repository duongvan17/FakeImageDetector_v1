"""OpenAI-compatible API for the FakeDet VLM.

Exposes ``/v1/models`` and ``/v1/chat/completions`` so any ChatGPT-style
front-end that speaks the OpenAI protocol (open-webui, LibreChat, Lobe Chat,
HuggingFace chat-ui, the openai python client, ...) can talk to the model.

The model only does one thing: given an image, classify real/fake and
explain the artifacts. So the handler ignores most chat parameters, pulls
the first image out of the message content, runs the detector, and returns
the verdict + evidence as the assistant message.

Run:
    uvicorn fakedet_vlm.serve.openai_api:app --host 0.0.0.0 --port 8000

Then point open-webui at  http://<host>:8000/v1  (no API key needed, but the
clients usually require *some* string — use "sk-fakedet" or anything).

Env vars (same as serve/api.py):
    FAKEDET_LLM_NAME, FAKEDET_VISION_CHECKPOINT, FAKEDET_ADAPTER_DIR,
    FAKEDET_PROJECTOR, FAKEDET_DEVICE, FAKEDET_MAX_NEW_TOKENS
"""
from __future__ import annotations

import base64
import binascii
import io
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from .api import _load_model, _STATE

MODEL_ID = "fakedet-vlm"


def _ensure_loaded() -> dict[str, Any]:
    if "model" not in _STATE:
        _STATE.update(_load_model())
    return _STATE


def _extract_image(messages: list[dict]) -> Image.Image | None:
    """Pull the last image from OpenAI-style message content.

    OpenAI multimodal format:
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]}
    """
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in reversed(content):
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                try:
                    b64 = url.split(",", 1)[1]
                    raw = base64.b64decode(b64)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except (IndexError, binascii.Error, OSError):
                    return None
    return None


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    stream: bool = False
    # accepted but ignored — the model behaviour is fixed
    temperature: float | None = None
    max_tokens: int | None = None


app = FastAPI(title="FakeDet VLM (OpenAI-compatible)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": "model" in _STATE}


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "fakedet",
        }],
    }


def _run(messages: list[dict]) -> str:
    image = _extract_image(messages)
    if image is None:
        return ("Please attach an image. This assistant detects whether an "
                "image is a deepfake and explains the visual artifacts.")

    st = _ensure_loaded()
    model = st["model"]
    device = st["device"]
    tmp = "/tmp/fakedet_openai_input.jpg"
    image.save(tmp, quality=95)

    pixel_values = st["transform"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=st["input_ids"],
            attention_mask=st["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=st["max_new_tokens"],
            do_sample=False,
        )
    text = model.tokenizer.decode(out[0], skip_special_tokens=True).strip()
    is_fake = ("deepfake" in text.lower()) or (
        "fake" in text.lower() and "authentic" not in text.lower()
    )
    verdict = "🟥 FAKE" if is_fake else "🟩 REAL"
    return f"**{verdict}**\n\n{text}"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict:
    answer = _run(req.messages)
    now = int(time.time())
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # Non-streaming response is enough for open-webui / LibreChat.
    return {
        "id": cid,
        "object": "chat.completion",
        "created": now,
        "model": req.model or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
