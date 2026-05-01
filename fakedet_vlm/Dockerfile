### Stage 1: build env
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Install torch first so we can pin the CUDA index URL.
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[serve]"

# Model artefacts must be mounted at /app/{clip_model,runs} at runtime — we do
# not bake the 1 GB ViT checkpoint into the image.
ENV FAKEDET_VISION_CHECKPOINT=/app/clip_model/best_model.pth \
    FAKEDET_ADAPTER_DIR=/app/runs/stage2_sft/final \
    FAKEDET_PROJECTOR=/app/runs/stage2_sft/final/projector.pt \
    FAKEDET_LAZY_LOAD=0

EXPOSE 8000

CMD ["uvicorn", "fakedet_vlm.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
