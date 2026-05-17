"""Space entrypoint: fetch the (private) model artifacts, then serve the
OpenAI-compatible FastAPI app.

``FAKEDET_LAZY_LOAD=1`` (set in the Dockerfile) means the model is loaded
on the first ``/v1/chat/completions`` request, so the port opens
immediately and the Space goes "Running" fast. ``/v1/models`` and
``/health`` respond instantly; the first chat request is slow (CPU load).
"""
import os

from huggingface_hub import hf_hub_download, snapshot_download

_TOKEN = os.environ.get("HF_TOKEN")  # Space secret; repos are private

hf_hub_download(
    "duongvan17/fakedet-vit-b16-fakeclue", "best_model.pth",
    local_dir="/app/clip_model", token=_TOKEN,
)
snapshot_download(
    "duongvan17/fakedet-vlm-stage2", local_dir="/app/adapter", token=_TOKEN,
    allow_patterns=["adapter_*", "projector.pt", "tokenizer*",
                    "special_*", "added_*", "vocab.json", "merges.txt"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fakedet_vlm.serve.openai_api:app",
        host="0.0.0.0",
        port=7860,
    )
