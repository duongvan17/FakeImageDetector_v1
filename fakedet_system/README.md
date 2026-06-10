# fakedet_system

Linear Haystack pipeline (NOT agentic) wrapping the trained FakeDet VLM:
`InputParser → IntentClassifier → MemoryReader → PromptBuilder →
Responder → MemoryWriter`.

Separate from `../fakedet_vlm` (train + model) on purpose — this runs on
the local box with no GPU and reaches the model over HTTP. 4 intents:
`image_deepfake` (trained model decides), `text_to_detect` (context
follow-up answered by the big external model + the previous image),
`user_guide` (external model), `misleading` (out-of-scope notice). The
external API (Gemini default) does intent + the contextual answers; the
trained model owns every image verdict.

Full run guide (tiếng Việt): [README_HE_THONG.md](README_HE_THONG.md).

```bash
pip install -e ".[serve]"
cp .env.example .env          # set Gemini key + model backend URL
uvicorn fakedet_system.serve.openai_api:app --port 9000
```
