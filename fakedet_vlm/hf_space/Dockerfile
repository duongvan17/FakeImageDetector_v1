# HF Spaces (Docker SDK) — serves fakedet_vlm.serve.openai_api directly.
# No Gradio (its api_info crashes on the FastAPI/pydantic mount). Pure
# FastAPI = the OpenAI-compatible API fakedet_system needs. Free CPU basic.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch (small wheel; no CUDA on free CPU Spaces).
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Runtime deps (bitsandbytes intentionally omitted: only used for 4-bit on
# CUDA; on CPU the loader sets load_in_4bit=False).
RUN pip install --no-cache-dir \
    "transformers>=4.46,<4.50" "peft>=0.12,<0.14" "accelerate>=0.34,<1.0" \
    "timm>=1.0.9" pillow "huggingface_hub>=0.24,<0.27" sentencepiece \
    protobuf "safetensors>=0.4" "fastapi>=0.115" "uvicorn>=0.30" pydantic \
    python-multipart

# The model code (no deps — we installed exactly what serving needs above).
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/duongvan17/FakeImageDetector_v1.git#subdirectory=fakedet_vlm"

COPY start.py .

ENV FAKEDET_VISION_CHECKPOINT=/app/clip_model/best_model.pth \
    FAKEDET_ADAPTER_DIR=/app/adapter \
    FAKEDET_PROJECTOR=/app/adapter/projector.pt \
    FAKEDET_MAX_NEW_TOKENS=96 \
    FAKEDET_LAZY_LOAD=1 \
    HF_HOME=/app/.cache

# HF Spaces runs the container as uid 1000 — make /app writable for the
# model download + cache at runtime.
RUN mkdir -p /app/.cache && chmod -R 777 /app

EXPOSE 7860
CMD ["python", "start.py"]
