# ChatGPT-style UI cho FakeDet VLM

Dùng **open-webui** (ChatGPT clone open-source, 50k+ sao GitHub) làm
front-end, model của mình làm backend qua OpenAI-compatible API.

```
Browser → open-webui (Docker :3000) → openai_api.py (:8000) → FakeDet VLM
```

---

## Cách 1: Docker Compose (1 lệnh, khuyên cho local có GPU)

Cần: Docker + NVIDIA Container Toolkit (GPU), model files local.

### Chuẩn bị model files

```bash
cd fakedet_vlm

# Tải vision checkpoint
hf download duongvan17/fakedet-vit-b16-fakeclue best_model.pth \
    --local-dir ../clip_model

# Tải VLM adapter + projector
hf download duongvan17/fakedet-vlm-stage2 \
    --local-dir runs/stage2_sft/final \
    --include "adapter_*" "projector.pt" "tokenizer*" "special_*" "added_*" "vocab.json" "merges.txt"
```

### Chạy

```bash
docker compose up -d --build
```

Lần đầu build ~5-10 phút. Xong:
- ChatGPT UI: <http://localhost:3000>
- Backend API: <http://localhost:8000/health>

### Dùng

1. Mở <http://localhost:3000>
2. Bỏ qua login (`WEBUI_AUTH=false`)
3. Chọn model `fakedet-vlm` ở dropdown trên cùng
4. Bấm icon 📎 (attach) → upload ảnh
5. Gửi → model trả về verdict 🟥FAKE/🟩REAL + evidence

### Dừng

```bash
docker compose down
```

---

## Cách 2: Backend local + open-webui Docker riêng

Nếu không build được Dockerfile backend (vd thiếu CUDA trong Docker):

### Bước 1: Chạy backend trực tiếp (Python local có GPU)

```bash
cd fakedet_vlm
pip install -e ".[serve]"

export FAKEDET_VISION_CHECKPOINT=../clip_model/best_model.pth
export FAKEDET_ADAPTER_DIR=runs/stage2_sft/final
export FAKEDET_PROJECTOR=runs/stage2_sft/final/projector.pt
export FAKEDET_MAX_NEW_TOKENS=64

uvicorn fakedet_vlm.serve.openai_api:app --host 0.0.0.0 --port 8000
```

Test API:

```bash
curl http://localhost:8000/v1/models
```

### Bước 2: Chạy open-webui Docker, trỏ vào backend

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=sk-fakedet \
  -e WEBUI_AUTH=false \
  -v openwebui:/app/backend/data \
  --name open-webui \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

→ <http://localhost:3000>

---

## Cách 3: Front-end khác (LibreChat / Lobe Chat)

Backend giống nhau (`openai_api.py`), chỉ đổi UI.

### LibreChat

```yaml
# librechat.yaml
endpoints:
  custom:
    - name: "FakeDet VLM"
      apiKey: "sk-fakedet"
      baseURL: "http://localhost:8000/v1"
      models:
        default: ["fakedet-vlm"]
```

### Lobe Chat

```bash
docker run -d -p 3210:3210 \
  -e OPENAI_API_KEY=sk-fakedet \
  -e OPENAI_PROXY_URL=http://host.docker.internal:8000/v1 \
  lobehub/lobe-chat
```

---

## Cách 4: Cloud (không có GPU local)

Backend cần GPU. Chạy backend trên Colab (free T4), expose ra ngoài bằng
`cloudflared` / `ngrok`, rồi open-webui local trỏ vào.

### Trên Colab:

```python
!pip install -q "transformers>=4.46,<4.50" "peft>=0.12,<0.14" bitsandbytes \
    accelerate timm pillow huggingface_hub fastapi uvicorn pydantic
!git clone https://github.com/duongvan17/FakeImageDetector_v1.git
%cd FakeImageDetector_v1/fakedet_vlm
!pip install -q -e . --no-deps

from huggingface_hub import login, hf_hub_download, snapshot_download
login()  # Read token
hf_hub_download("duongvan17/fakedet-vit-b16-fakeclue", "best_model.pth",
                local_dir="../clip_model")
snapshot_download("duongvan17/fakedet-vlm-stage2", local_dir="runs/stage2_sft/final",
                  allow_patterns=["adapter_*","projector.pt","tokenizer*",
                                  "special_*","added_*","vocab.json","merges.txt"])

import os
os.environ["FAKEDET_VISION_CHECKPOINT"] = "../clip_model/best_model.pth"
os.environ["FAKEDET_ADAPTER_DIR"] = "runs/stage2_sft/final"
os.environ["FAKEDET_PROJECTOR"] = "runs/stage2_sft/final/projector.pt"

# Expose qua cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
import subprocess, threading
threading.Thread(target=lambda: subprocess.run(
    ["uvicorn","fakedet_vlm.serve.openai_api:app","--host","0.0.0.0","--port","8000"]
), daemon=True).start()
import time; time.sleep(40)
!./cloudflared tunnel --url http://localhost:8000
# In ra https://xxxx.trycloudflare.com — dùng url đó + /v1 cho open-webui
```

Local: chạy open-webui Docker, set `OPENAI_API_BASE_URL=https://xxxx.trycloudflare.com/v1`.

---

## So sánh

| Cách | GPU | Setup | Bền | Khuyên |
|---|---|---|---|---|
| 1. Docker Compose | Local | 1 lệnh | Local | ✅ Nếu có GPU |
| 2. Backend local + UI Docker | Local | 2 bước | Local | Nếu Dockerfile fail |
| 3. LibreChat/Lobe | Local | Tùy UI | Local | Thích UI khác |
| 4. Colab + cloudflared | Free | Phức tạp | ~12h | Không có GPU |

---

## Lưu ý

- **API không cần key thật** — open-webui bắt buộc nhập 1 string, dùng `sk-fakedet`
- Model **chỉ làm 1 việc**: nhận ảnh → verdict + evidence. Hỏi text không kèm ảnh → nó nhắc gửi ảnh
- `openai_api.py` non-streaming (đủ cho open-webui/LibreChat). Muốn streaming thì thêm SSE sau
- Backend reuse `serve/api.py` loader → cùng env vars, cùng cách load model

## Troubleshooting

### open-webui không thấy model

- Check `curl http://localhost:8000/v1/models` trả về `fakedet-vlm`
- Trong open-webui: Settings → Connections → verify OpenAI URL = `http://fakedet-backend:8000/v1` (Docker Compose) hoặc `http://host.docker.internal:8000/v1` (UI Docker riêng)

### `host.docker.internal` không resolve (Linux)

Thêm `--add-host=host.docker.internal:host-gateway` vào `docker run`
(đã có trong lệnh Cách 2).

### Backend OOM

`FAKEDET_MAX_NEW_TOKENS=48` + đảm bảo dùng GPU (4-bit). CPU thì rất chậm
(~30-60s/ảnh) nhưng vẫn chạy.

### Build Dockerfile lỗi CUDA

Dùng Cách 2 (backend chạy Python local, chỉ UI trong Docker).
