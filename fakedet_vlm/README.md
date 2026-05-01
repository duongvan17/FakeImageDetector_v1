# fakedet_vlm

Refactor of the FakeImageDetector VLM stack. Built around the user's
fine-tuned **ViT-Base/16** (768-dim, 224×224, acc 98.82% on FakeClue) as a
frozen vision tower, projected into **Qwen2.5-1.5B-Instruct** via a 2-layer
MLP, and fine-tuned with LoRA. Designed to fit on a single 12 GB VRAM GPU.

This replaces `../vlm_system/`. Key bugs fixed:

- Vision tower now matches the actual checkpoint (ViT-B/16 768-dim, 224×224,
  loaded from `model_state_dict`) instead of the assumed CLIP-L 1024-dim.
- All **196 patch tokens** are passed to the LLM (one LLM token per patch),
  not a single pooled vector.
- Visual injection uses `masked_scatter` so gradients flow to the projector
  during training.
- Label masking is built by tokenizing prompt and response **separately** —
  no fragile substring matching.
- Stage-2 LR uses two parameter groups (`projector_lr=2e-4`, `lora_lr=2e-5`)
  per LLaVA recipe.
- `eval_strategy` (transformers ≥4.46), checkpoints persist projector +
  LoRA adapter together.

## Layout

```
fakedet_vlm/
├── pyproject.toml
├── configs/{base,stage1,stage2}.yaml
├── src/fakedet_vlm/
│   ├── models/{vit_loader,projector,vlm}.py
│   ├── data/{prompts,dataset,collator}.py
│   ├── train/{common,stage1,stage2}.py
│   ├── infer/pipeline.py
│   └── utils/memory.py
├── scripts/{verify_vit_loads,prepare_fakeclue}.py
└── tests/{test_vit_loader,test_projector,test_masked_scatter,test_dataset}.py
```

## Setup

```bash
# 1) Create env (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install torch with CUDA 11.8 (or 12.1 — match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) Install package
pip install -e ".[dev,train]"
```

## Sanity check first

Before any training, verify the vision tower loads correctly:

```bash
python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
# expect: output shape = (2, 196, 768)
```

Run unit tests:

```bash
pytest -q
```

## Data prep

```bash
python scripts/prepare_fakeclue.py --out data --train-ratio 0.9
# Smoke test with 200 samples first:
python scripts/prepare_fakeclue.py --out data --max-samples 200
```

## Training

### Stage 1 — projector alignment (1 epoch)

```bash
python -m fakedet_vlm.train.stage1 \
  --base configs/base.yaml --stage configs/stage1.yaml
```

Trainable: projector only (~3M params). LR 1e-3, cosine, batch effective 16.
Output: `runs/stage1_align/projector.pt`.

### Stage 2 — LoRA SFT (2 epochs)

```bash
python -m fakedet_vlm.train.stage2 \
  --base configs/base.yaml --stage configs/stage2.yaml
```

Trainable: projector (LR 2e-4) + LoRA r=16 (LR 2e-5). 4-bit NF4 + paged
adamw 8-bit. Output: `runs/stage2_sft/final/{adapter_model.safetensors,
projector.pt}`.

## Evaluation

After stage 2 finishes, score the model on the held-out val set:

```bash
make eval
# or manually:
python scripts/eval.py \
  --val-json data/val.json --images-dir data/images \
  --adapter-dir runs/stage2_sft/final \
  --projector runs/stage2_sft/final/projector.pt \
  --out runs/stage2_sft/eval
```

Outputs:

- `runs/stage2_sft/eval/eval_metrics.json` — overall + per-category accuracy /
  precision / recall / F1 / AUC.
- `runs/stage2_sft/eval/eval_predictions.jsonl` — one row per sample with the
  raw response, prediction, target, and category for offline error analysis.

## Inference (Python)

```python
from fakedet_vlm.infer import DeepfakeDetector

det = DeepfakeDetector(
    llm_name="Qwen/Qwen2.5-1.5B-Instruct",
    vision_checkpoint="../clip_model/best_model.pth",
    adapter_dir="runs/stage2_sft/final",
    projector_path="runs/stage2_sft/final/projector.pt",
)
print(det.detect("path/to/image.jpg"))
```

## Serving (FastAPI)

```bash
# Local
make serve   # uvicorn on :8000

# Test
curl -F file=@test.jpg http://localhost:8000/detect
# {"classification":"Fake","response":"...","confidence":0.9}
```

Endpoints:

| Method | Path     | Purpose |
|--------|----------|---------|
| GET    | /health  | Liveness + model-loaded flag |
| POST   | /detect  | multipart `file` upload → JSON |

The model loads at startup unless `FAKEDET_LAZY_LOAD=1`. Configure via env:
`FAKEDET_LLM_NAME`, `FAKEDET_VISION_CHECKPOINT`, `FAKEDET_ADAPTER_DIR`,
`FAKEDET_PROJECTOR`, `FAKEDET_DEVICE`, `FAKEDET_MAX_NEW_TOKENS`.

## Docker

```bash
make docker
# At runtime, mount the vision checkpoint and trained adapter:
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/clip_model:/app/clip_model:ro \
  -v $(pwd)/runs:/app/runs:ro \
  fakedet-vlm:0.1
```

The image base is `nvidia/cuda:12.1.0-runtime-ubuntu22.04`; the 1 GB ViT
checkpoint is *not* baked in — mount it.

## VRAM budget (RTX 3060 12 GB, bs=1, accum=16)

| Component                 | Size   |
|---------------------------|--------|
| Qwen2.5-1.5B (4-bit NF4)  | ~1.0 G |
| ViT-B/16 (fp32 frozen)    | ~0.4 G |
| Embeds + activations      | ~3 G   |
| Optimizer state (8-bit)   | ~0.8 G |
| Gradients (LoRA + proj)   | ~0.2 G |
| KV cache + slack          | ~3 G   |
| **Peak**                  | **~8.5 G** |

Reduce `data.max_length` (default 1024 → 768) or set `lora.r=8` if you OOM.

## Configuration knobs

- `model.num_visual_tokens` — keep at 196 unless you compress (LLaVA-Mini
  style). Setting to 1 reproduces the old behaviour and silently degrades
  quality.
- `model.image_size` — must equal what the ViT was trained at (224). The
  loader rebuilds positional embeddings only at the right size.
- `data.max_length` — must accommodate 196 image tokens + system + user +
  assistant tokens. 1024 is safe.

## Roadmap

- [ ] Stage 3 DPO on hard negatives from val misclassifications
- [ ] vLLM serving wrapper (currently bottlenecked by HF generate)
- [ ] FastAPI + Docker deployment template
- [ ] Distillation to a 0.5B LLM for edge deployment
