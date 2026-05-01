# Hướng dẫn chạy FakeDet VLM từ đầu

Hướng dẫn end-to-end bằng tiếng Việt cho người mới: từ cài máy → tải FakeClue
trên HuggingFace → train → eval → deploy.

> **TL;DR**: 1 GPU NVIDIA 12 GB VRAM + ~50 GB ổ cứng + Python 3.10–3.12.
> Toàn bộ pipeline ~6–10 tiếng tùy GPU.

---

## Mục lục

1. [Yêu cầu phần cứng](#1-yêu-cầu-phần-cứng)
2. [Cài Python + CUDA + PyTorch](#2-cài-python--cuda--pytorch)
3. [Clone + cài project](#3-clone--cài-project)
4. [Đăng nhập HuggingFace](#4-đăng-nhập-huggingface)
5. [Sanity check vision tower](#5-sanity-check-vision-tower)
6. [Tải dataset FakeClue](#6-tải-dataset-fakeclue)
7. [Smoke test với 200 mẫu](#7-smoke-test-với-200-mẫu-trước)
8. [Train Stage 1: Projector alignment](#8-train-stage-1-projector-alignment)
9. [Train Stage 2: LoRA SFT](#9-train-stage-2-lora-sft)
10. [Đánh giá model](#10-đánh-giá-model)
11. [Inference 1 ảnh](#11-inference-1-ảnh)
12. [Deploy bằng FastAPI](#12-deploy-bằng-fastapi)
13. [Đóng gói Docker](#13-đóng-gói-docker)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Yêu cầu phần cứng

| | Tối thiểu | Khuyên dùng |
|---|---|---|
| GPU VRAM | 12 GB (RTX 3060 / 4060) | 16 GB+ (RTX 4060 Ti / 4070 / 3090) |
| RAM | 16 GB | 32 GB |
| Ổ cứng trống | 50 GB | 100 GB (cho dataset full + checkpoints) |
| Driver NVIDIA | ≥ 535 | ≥ 545 |
| OS | Windows 11 / Ubuntu 22.04 | Ubuntu 22.04 |

Kiểm tra GPU có sẵn:

```bash
nvidia-smi
```

Nếu không có GPU NVIDIA → vẫn dùng được CPU mode để **inference**, nhưng
**không train** được (4-bit quantization của bitsandbytes cần CUDA).

---

## 2. Cài Python + CUDA + PyTorch

### Windows

1. Cài Python 3.12 từ <https://www.python.org/downloads/> (tick "Add to PATH").
2. Kiểm tra CUDA driver: `nvidia-smi` (cột "CUDA Version" cho biết driver
   support tới đâu, ví dụ 12.4 = driver chạy được CUDA toolkit ≤ 12.4).
3. Tạo virtualenv:
   ```powershell
   cd d:\FakeImageDetector_v1\fakedet_vlm
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
4. Cài PyTorch với CUDA 12.1 (chọn CUDA version match driver):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Linux

```bash
cd ~/FakeImageDetector_v1/fakedet_vlm
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify CUDA hoạt động

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# CUDA: True NVIDIA GeForce RTX 3060
```

> **Lưu ý**: nếu PyTorch in `False` → driver/toolkit mismatch. Cài lại
> đúng index URL (`cu118` cho driver cũ hơn).

---

## 3. Clone + cài project

```bash
# Project đã có ở d:\FakeImageDetector_v1, vào fakedet_vlm:
cd fakedet_vlm

# Cài package + dev/train extras
pip install -e ".[dev,train]"
```

`pip install -e ".[dev,train]"` sẽ kéo về:
`transformers, accelerate, peft, bitsandbytes, datasets, timm, scikit-learn,
pytest, wandb, trl`. Lần đầu mất 5-10 phút.

> **Windows note**: `bitsandbytes` cần CUDA. Nếu cài lỗi, thử
> `pip install bitsandbytes --upgrade --index-url https://pypi.org/simple/`.

Verify import:

```bash
python -c "import transformers, peft, bitsandbytes, timm; print('OK')"
```

---

## 4. Đăng nhập HuggingFace

Một số dataset/model trên HF cần login (FakeClue không cần, nhưng Qwen2.5
download nhanh hơn nếu có token).

1. Tạo token: <https://huggingface.co/settings/tokens> → "Read" scope là đủ.
2. Login:
   ```bash
   huggingface-cli login
   # Paste token khi được hỏi
   ```
   Hoặc set env: `HF_TOKEN=hf_xxx...` trong shell.

Test:
```bash
python -c "from huggingface_hub import whoami; print(whoami())"
```

---

## 5. Sanity check vision tower

**LÀM TRƯỚC KHI TẢI DATASET** — verify checkpoint ViT-B của bạn load đúng:

```bash
python scripts/verify_vit_loads.py --ckpt ../clip_model/best_model.pth
```

Output mong đợi:

```
[1/3] Loading ViT-B/16 from ../clip_model/best_model.pth ...
      OK — 85.8M params, frozen=True
[2/3] Forward on dummy (2, 3, 224, 224) ...
      output shape = (2, 196, 768)  (expected: (2, 196, 768))
[3/3] Range / norm sanity:
      min=-5.976 max=9.489 mean=0.176 std=0.824

OK — vision tower loads and runs correctly.
```

Nếu báo lỗi `Vision checkpoint mismatch` → checkpoint của bạn khác kiến trúc
mình giả định. Mở [vit_loader.py:65-79](src/fakedet_vlm/models/vit_loader.py#L65-L79)
và adjust `timm_name` cho đúng variant.

Chạy unit tests:

```bash
pytest -q
# 10/10 passed
```

---

## 6. Tải dataset FakeClue

Dataset trên HuggingFace: <https://huggingface.co/datasets/lingcco/FakeClue>

- ~100,000 ảnh, 7 categories (`animal, human, object, scenery, satellite,
  document, deepfake`).
- Mỗi sample có `image, label (0=real, 1=fake), clue (text artifacts)`.
- Tổng dung lượng raw: ~30-40 GB sau khi save thành JPG.

### Tải full dataset

```bash
python scripts/prepare_fakeclue.py --out data --train-ratio 0.9
```

Script này sẽ:
1. Stream dataset từ HuggingFace (`load_dataset("lingcco/FakeClue")`).
2. Shuffle với seed=42.
3. Save từng ảnh thành `data/images/{train,val}_NNNNNN.jpg`.
4. Build LLaVA-style JSON ở `data/train.json` và `data/val.json`.
5. In stats per-category + balance real/fake.

Output mẫu sau khi xong:

```
[3/3] Train=89421  Val=9936

[train] total=89421  real=44103 (49.3%)  fake=45318 (50.7%)
  per-category:
    deepfake          24532 (27.4%)
    human             18901 (21.1%)
    object            12044 (13.5%)
    scenery           11830 (13.2%)
    animal             9221 (10.3%)
    satellite          7452  (8.3%)
    document           5441  (6.1%)

[val] total=9936  real=4892 (49.2%)  fake=5044 (50.8%)
  per-category: ...
```

> **Thời gian**: 30-60 phút tùy băng thông. Lần đầu chạy `datasets` sẽ cache
> về `~/.cache/huggingface/datasets/` (~25 GB).

### Lưu ý ổ cứng

- HF cache: `~/.cache/huggingface/` ~25 GB
- `data/images/`: ~15-30 GB
- `data/{train,val}.json`: ~50 MB

→ chuẩn bị **>50 GB free**.

Nếu thiếu chỗ, giới hạn số mẫu (xem mục 7).

---

## 7. Smoke test với 200 mẫu trước

**Khuyên làm trước khi train full** — verify pipeline hoạt động end-to-end:

```bash
# Reset thư mục data nếu đã có
rm -rf data
python scripts/prepare_fakeclue.py --out data --max-samples 200
```

Sẽ tạo ~180 mẫu train + ~20 val. Đủ để check:
- Train loop chạy không OOM
- Loss giảm bình thường
- Eval script chạy được

Sau smoke test thành công → xóa và prep full dataset.

---

## 8. Train Stage 1: Projector alignment

**Mục tiêu**: Train projector MLP (3M params) để align không gian visual ↔
LLM. ViT và LLM frozen.

```bash
python -m fakedet_vlm.train.stage1 \
  --base configs/base.yaml --stage configs/stage1.yaml
```

Hoặc dùng Makefile:

```bash
make stage1
```

### Cấu hình mặc định ([configs/stage1.yaml](configs/stage1.yaml))

| Param | Giá trị | Lý do |
|---|---|---|
| Trainable | projector only | ~3M params, train nhanh |
| Epochs | 1 | Chỉ cần align, không cần fit task |
| Batch | 1 × accum 16 = 16 | Giữ effective batch ổn |
| LR | 1e-3 | Cao vì projector init từ random |
| Warmup | 3% | Standard |
| Scheduler | cosine | |
| bf16 | true | Stable hơn fp16 trên Ampere+ |

### Theo dõi

Trong terminal, mỗi `logging_steps=10` sẽ in:

```
{'loss': 4.3215, 'grad_norm': 8.2, 'learning_rate': 8.5e-4, 'epoch': 0.12}
{'loss': 3.8901, ...}
```

Loss expected sau 1 epoch: **~2.5-3.5** (chưa phải task loss, chỉ alignment).

### Thời gian

| GPU | 100K mẫu × 1 epoch |
|---|---|
| RTX 3060 12 GB | ~3 giờ |
| RTX 4060 Ti 16 GB | ~2 giờ |
| RTX 4090 | ~45 phút |
| A100 40 GB | ~30 phút |

### Output

```
runs/stage1_align/
├── checkpoint-500/
├── checkpoint-1000/
├── ...
└── projector.pt          ← file quan trọng nhất, dùng cho Stage 2
```

### Bật wandb (tùy chọn)

```bash
wandb login   # 1 lần
# Trong configs/stage1.yaml đổi report_to: "wandb"
```

---

## 9. Train Stage 2: LoRA SFT

**Mục tiêu**: Fine-tune LLM bằng LoRA (r=16) + projector tiếp tục train, để
model học task deepfake detection + generate evidence.

```bash
python -m fakedet_vlm.train.stage2 \
  --base configs/base.yaml --stage configs/stage2.yaml
```

Hoặc:

```bash
make stage2
```

### Cấu hình ([configs/stage2.yaml](configs/stage2.yaml))

| Param | Giá trị |
|---|---|
| Trainable | projector + LoRA on q/k/v/o/gate/up/down |
| LoRA r | 16 |
| LoRA alpha | 32 |
| Epochs | 2 |
| projector_lr | 2e-4 |
| lora_lr | 2e-5 (10× thấp hơn) |
| Optim | paged_adamw_8bit |
| Augment | DeepfakeAugment (JPEG, color, resize) |
| Early stopping | patience=3 |

### Loss expected

- Đầu epoch 1: ~3.0
- Cuối epoch 2: **0.4 - 0.8** (task loss thực sự)

Nếu loss không giảm dưới 1.5 → debug:
- Check Stage 1 projector có load đúng không (log "loaded projector from ...")
- Verify augmentation không quá mạnh

### Thời gian

| GPU | 100K mẫu × 2 epoch |
|---|---|
| RTX 3060 12 GB | ~8 giờ |
| RTX 4060 Ti 16 GB | ~6 giờ |
| RTX 4090 | ~2.5 giờ |
| A100 40 GB | ~1.5 giờ |

### Output

```
runs/stage2_sft/
├── checkpoint-500/
├── ...
└── final/
    ├── adapter_model.safetensors   ← LoRA weights
    ├── adapter_config.json
    ├── projector.pt                 ← Projector
    ├── tokenizer.json
    └── ...
```

---

## 10. Đánh giá model

```bash
python scripts/eval.py \
  --val-json data/val.json \
  --images-dir data/images \
  --adapter-dir runs/stage2_sft/final \
  --projector runs/stage2_sft/final/projector.pt \
  --out runs/stage2_sft/eval
```

Hoặc:

```bash
make eval
```

### Kết quả

In ra terminal + lưu vào `runs/stage2_sft/eval/`:

```
[Overall] n=9936  acc=0.912  P=0.918 R=0.905 F1=0.911 AUC=0.962
[Per-category]
  animal           n=  920  acc=0.945  F1=0.943
  deepfake         n= 2453  acc=0.881  F1=0.882
  document         n=  544  acc=0.971  F1=0.965
  human            n= 1890  acc=0.902  F1=0.901
  object           n= 1204  acc=0.931  F1=0.929
  satellite        n=  745  acc=0.952  F1=0.950
  scenery          n= 1183  acc=0.918  F1=0.916
```

Files:
- `eval_metrics.json` — overall + per-category metrics
- `eval_predictions.jsonl` — 1 dòng/sample với `target, pred, score, response`

### Mục tiêu

Theo paper FakeVLM (NeurIPS 2025): accuracy ≥ 0.85, F1 ≥ 0.80 là ổn cho
production. Nếu < 0.80 → cân nhắc:
- Train thêm 1 epoch
- Tăng LoRA r=32
- Tăng dataset (đã dùng full chưa?)
- Stage 3 DPO (xem mục P3 README)

### Error analysis

```bash
# Xem các ảnh model đoán sai
python -c "
import json
with open('runs/stage2_sft/eval/eval_predictions.jsonl') as f:
    wrong = [json.loads(l) for l in f if json.loads(l)['target'] != json.loads(l)['pred']]
print(f'Wrong: {len(wrong)}')
for w in wrong[:5]:
    print(f'  {w[\"category\"]:<12s} target={w[\"target\"]} pred={w[\"pred\"]} | {w[\"response\"][:80]}')
"
```

---

## 11. Inference 1 ảnh

### Python API

```python
from fakedet_vlm.infer import DeepfakeDetector

det = DeepfakeDetector(
    llm_name="Qwen/Qwen2.5-1.5B-Instruct",
    vision_checkpoint="../clip_model/best_model.pth",
    adapter_dir="runs/stage2_sft/final",
    projector_path="runs/stage2_sft/final/projector.pt",
)

result = det.detect("path/to/test.jpg")
print(result)
# {
#   'image_path': 'path/to/test.jpg',
#   'classification': 'Fake',
#   'response': 'This image is a deepfake. Evidence: Unnatural skin texture...'
# }
```

### Tốc độ

| GPU | Latency / ảnh |
|---|---|
| RTX 3060 | ~3-5s |
| RTX 4090 | ~1s |

---

## 12. Deploy bằng FastAPI

### Chạy local

```bash
make serve
# hoặc:
uvicorn fakedet_vlm.serve.api:app --host 0.0.0.0 --port 8000
```

Server load model lúc startup (~30s đầu), sau đó nhận request:

```bash
curl -F file=@test.jpg http://localhost:8000/detect
```

Response:

```json
{
  "classification": "Fake",
  "response": "This image is a deepfake. Evidence: Unnatural texture...",
  "confidence": 0.9
}
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok","loaded":true}
```

Swagger UI: <http://localhost:8000/docs>

### Configure qua env vars

```bash
export FAKEDET_LLM_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export FAKEDET_VISION_CHECKPOINT="./clip_model/best_model.pth"
export FAKEDET_ADAPTER_DIR="./runs/stage2_sft/final"
export FAKEDET_PROJECTOR="./runs/stage2_sft/final/projector.pt"
export FAKEDET_DEVICE="cuda"
export FAKEDET_MAX_NEW_TOKENS="96"
export FAKEDET_LAZY_LOAD="0"   # 0 = load ở startup, 1 = load khi request đầu
```

---

## 13. Đóng gói Docker

```bash
make docker
# hoặc
docker build -t fakedet-vlm:0.1 .
```

Run với GPU + mount checkpoint:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/clip_model:/app/clip_model:ro \
  -v $(pwd)/runs:/app/runs:ro \
  fakedet-vlm:0.1
```

> **Image base**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`. Image size sau
> build ~6 GB (bao gồm PyTorch CUDA). 1 GB ViT checkpoint **không** bake vào
> image — phải mount.

Test:

```bash
curl -F file=@test.jpg http://localhost:8000/detect
```

---

## 14. Troubleshooting

### CUDA out of memory (OOM)

Trong `configs/stage2.yaml`:

```yaml
data:
  max_length: 768          # giảm từ 1024

train:
  gradient_accumulation_steps: 8   # giảm từ 16

lora:
  r: 8                     # giảm từ 16
```

Hoặc thêm flags khi chạy: tắt eval thường xuyên (`eval_steps: 1000`).

### Lỗi `bitsandbytes` không tìm thấy CUDA

```bash
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
# Nếu lỗi:
pip uninstall bitsandbytes -y
pip install bitsandbytes --upgrade
```

Trên Windows: dùng `bitsandbytes-windows` fork nếu version chính thức không
chạy:
```bash
pip install bitsandbytes-windows
```

### `lingcco/FakeClue` download chậm / fail

```bash
# Set HF mirror (Trung Quốc / châu Á)
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_fakeclue.py --out data
```

### `<image>` token id is None

Tokenizer Qwen2.5 chưa có `<image>`. Fix tự động bởi `build_tokenizer` trong
[vlm.py](src/fakedet_vlm/models/vlm.py). Nếu vẫn lỗi → kiểm tra
transformers version ≥ 4.46:

```bash
pip install -U transformers
```

### Stage 1 loss không giảm

- Verify projector có trainable: log "trainable params = 3.0M (projector only)"
- Verify ViT output shape `(B, 196, 768)` qua `verify_vit_loads.py`
- Kiểm tra dataset: open 1 ảnh trong `data/images/` xem đúng không corrupt

### Stage 2 generate ra text rỗng / lặp

Có thể do tokenizer không decode đúng. Test:

```python
from fakedet_vlm.models.vlm import build_tokenizer
tok = build_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
print(tok.decode(tok("Hello <image> world")["input_ids"]))
```

Nếu output không có `<image>` → token chưa add đúng. Re-train Stage 1.

### Model dự đoán toàn "Real"

Augmentation quá mạnh hoặc dataset imbalance. Check:

```bash
grep -c '"label": 1' data/train.json
grep -c '"label": 0' data/train.json
```

Nếu lệch > 70/30 → dùng class weights hoặc oversample minority class.

### Eval AUC = 0.5

Parsing classification từ text fail. Mở 1-2 dòng `eval_predictions.jsonl`
xem `response` thực tế có chữ "deepfake" / "fake" / "authentic" không. Nếu
model output dạng khác (vd "yes" / "no") → adjust `_parse_classification` trong
[scripts/eval.py:54-78](scripts/eval.py).

---

## Tóm tắt commands theo thứ tự

```bash
# 1. Setup (1 lần)
cd fakedet_vlm
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev,train]"
huggingface-cli login

# 2. Verify (1 lần)
make verify
make test

# 3. Data (1 lần, 30-60 phút)
make data

# 4. Train (chia 2 stage)
make stage1   # 2-3 giờ
make stage2   # 6-8 giờ

# 5. Eval
make eval

# 6. Serve
make serve

# 7. Docker (optional)
make docker
```

Total time đầu tiên (RTX 3060): **~12 giờ** (chủ yếu là Stage 2).

---

## Liên hệ / Issues

- Bug trong code: mở issue trên repo
- Câu hỏi về dataset FakeClue: <https://huggingface.co/datasets/lingcco/FakeClue>
- Paper FakeVLM (kiến trúc tham khảo): <https://arxiv.org/abs/2503.14905>
