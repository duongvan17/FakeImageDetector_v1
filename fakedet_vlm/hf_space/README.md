---
title: FakeDet VLM Backend
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenAI-compatible backend for the FakeDet VLM
---

# FakeDet VLM — backend (HF Spaces, Docker)

Serves the trained FakeDet VLM (ViT-B/16 frozen + Qwen2.5-1.5B + LoRA)
as a plain OpenAI-compatible FastAPI service — no Gradio.

- `GET /health`, `GET /v1/models`
- `POST /v1/chat/completions` — send an image → REAL/FAKE + evidence;
  send text → trained LLM answers.

Point `fakedet_system`'s `FAKEDET_BACKEND_URL` at
`https://<user>-<space>.hf.space/v1`.

**Required Space secret:** `HF_TOKEN` (a *Read* token) — the two model
repos are private. **Hardware:** `CPU basic` (free) is enough (slow).

Setup (tiếng Việt): `HUONG_DAN_HF.md` in the code repo
(`fakedet_vlm/hf_space/`).
