"""Gradio demo for the FakeDet VLM.

Runs anywhere with a GPU: Google Colab (free T4), HuggingFace Spaces
(free ZeroGPU), or a local CUDA machine. Downloads both checkpoints from
the Hub at startup so no local model files are needed.

HF Hub repos used (set them private→public or pass a read token via the
HF_TOKEN env var / `huggingface-cli login`):
  - duongvan17/fakedet-vit-b16-fakeclue   (best_model.pth — vision encoder)
  - duongvan17/fakedet-vlm-stage2          (LoRA adapter + projector)

Local run:
    pip install -e ".[serve]" gradio
    python app.py

HF Spaces: put this file as app.py at the Space root with a requirements.txt.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch

VIS_REPO = os.environ.get("FAKEDET_VIS_REPO", "duongvan17/fakedet-vit-b16-fakeclue")
VLM_REPO = os.environ.get("FAKEDET_VLM_REPO", "duongvan17/fakedet-vlm-stage2")
LLM_NAME = os.environ.get("FAKEDET_LLM_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

_DETECTOR = None


def _bootstrap():
    """Download checkpoints + build the detector once."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading vision checkpoint ...")
    clip_dir = Path("clip_model")
    clip_dir.mkdir(exist_ok=True)
    hf_hub_download(
        repo_id=VIS_REPO,
        filename="best_model.pth",
        local_dir=str(clip_dir),
        repo_type="model",
    )

    print("Downloading VLM adapter + projector ...")
    vlm_dir = Path("vlm_final")
    snapshot_download(
        repo_id=VLM_REPO,
        local_dir=str(vlm_dir),
        repo_type="model",
        allow_patterns=[
            "adapter_model.safetensors",
            "adapter_config.json",
            "projector.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        ],
    )

    from fakedet_vlm.infer import DeepfakeDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
    _DETECTOR = DeepfakeDetector(
        llm_name=LLM_NAME,
        vision_checkpoint=str(clip_dir / "best_model.pth"),
        adapter_dir=str(vlm_dir),
        projector_path=str(vlm_dir / "projector.pt"),
        device=device,
    )
    print("Ready.")
    return _DETECTOR


def analyze(image):
    if image is None:
        return "—", "Upload an image first."
    det = _bootstrap()
    tmp = "/tmp/fakedet_input.jpg"
    image.convert("RGB").save(tmp, quality=95)
    result = det.detect(tmp, max_new_tokens=64)
    label = result["classification"]
    badge = "🟥 FAKE" if label == "Fake" else "🟩 REAL"
    return badge, result["response"]


with gr.Blocks(title="FakeDet VLM — Deepfake Detection") as demo:
    gr.Markdown(
        "# FakeDet VLM — Explainable Deepfake Detection\n"
        "ViT-B/16 (frozen) + Qwen2.5-1.5B + LoRA. Upload an image; the model "
        "classifies real/fake and explains the visual artifacts it relied on."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            verdict = gr.Textbox(label="Verdict", interactive=False)
            evidence = gr.Textbox(label="Evidence", lines=6, interactive=False)
    btn.click(analyze, inputs=inp, outputs=[verdict, evidence])
    gr.Markdown(
        "_Trained on FakeClue. Results are most reliable on in-distribution "
        "images; out-of-distribution accuracy may be lower._"
    )


if __name__ == "__main__":
    demo.launch(share=True)
