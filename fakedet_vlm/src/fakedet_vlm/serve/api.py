"""FastAPI deepfake detection service.

Configuration via environment variables:
  FAKEDET_LLM_NAME          (default: Qwen/Qwen2.5-1.5B-Instruct)
  FAKEDET_VISION_CHECKPOINT (default: ./clip_model/best_model.pth)
  FAKEDET_ADAPTER_DIR       (default: ./runs/stage2_sft/final)
  FAKEDET_PROJECTOR         (default: ./runs/stage2_sft/final/projector.pt)
  FAKEDET_DEVICE            (default: cuda if available else cpu)
  FAKEDET_MAX_NEW_TOKENS    (default: 96)

Run:
  uvicorn fakedet_vlm.serve.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from fakedet_vlm.data.prompts import IMAGE_PLACEHOLDER, build_chat_prompt
from fakedet_vlm.models import FakeDetVLM
from fakedet_vlm.models.vit_loader import CLIP_MEAN, CLIP_STD


_STATE: dict[str, Any] = {}


def _load_model() -> dict[str, Any]:
    from torchvision import transforms

    llm_name = os.environ.get("FAKEDET_LLM_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    vision_ckpt = os.environ.get("FAKEDET_VISION_CHECKPOINT", "./clip_model/best_model.pth")
    adapter_dir = os.environ.get("FAKEDET_ADAPTER_DIR", "./runs/stage2_sft/final")
    projector_path = os.environ.get("FAKEDET_PROJECTOR", "./runs/stage2_sft/final/projector.pt")
    device = os.environ.get("FAKEDET_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(os.environ.get("FAKEDET_IMAGE_SIZE", "224"))
    num_visual_tokens = int(os.environ.get("FAKEDET_NUM_VISUAL_TOKENS", "196"))
    max_new_tokens = int(os.environ.get("FAKEDET_MAX_NEW_TOKENS", "96"))

    model = FakeDetVLM(
        llm_name=llm_name,
        vision_checkpoint=vision_ckpt,
        num_visual_tokens=num_visual_tokens,
        image_size=image_size,
        load_in_4bit=(device == "cuda"),
    )
    if Path(adapter_dir).exists():
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, adapter_dir)
    if Path(projector_path).exists():
        sd = torch.load(projector_path, map_location="cpu")
        model.projector.load_state_dict(sd)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])

    # Pre-tokenize the prompt prefix (image tokens already expanded).
    prefix, _ = build_chat_prompt(assistant_response=None)
    prefix = prefix.replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER * num_visual_tokens, 1)
    enc = model.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)

    return {
        "model": model,
        "transform": transform,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("FAKEDET_LAZY_LOAD") != "1":
        _STATE.update(_load_model())
    yield
    _STATE.clear()


class DetectResponse(BaseModel):
    classification: str
    response: str
    confidence: float


def build_app() -> FastAPI:
    app = FastAPI(title="FakeDet VLM", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "loaded": "model" in _STATE}

    @app.post("/detect", response_model=DetectResponse)
    @torch.no_grad()
    async def detect(file: UploadFile = File(...)) -> DetectResponse:
        if "model" not in _STATE:
            _STATE.update(_load_model())
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Bad image: {e}")

        model = _STATE["model"]
        device = _STATE["device"]
        pixel_values = _STATE["transform"](image).unsqueeze(0).to(device)
        out = model.generate(
            input_ids=_STATE["input_ids"],
            attention_mask=_STATE["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=_STATE["max_new_tokens"],
            do_sample=False,
        )
        text = model.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        is_fake = ("deepfake" in text.lower()) or (
            "fake" in text.lower() and "authentic" not in text.lower()
        )
        return DetectResponse(
            classification="Fake" if is_fake else "Real",
            response=text,
            confidence=0.9 if is_fake else 0.85,  # placeholder until calibrated
        )

    return app


app = build_app()
