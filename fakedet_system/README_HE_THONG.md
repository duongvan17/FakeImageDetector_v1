# FakeDet System — pipeline luận văn (Haystack, KHÔNG agentic)

Đây là **phần "system"** của đồ án, **tách bạch hoàn toàn** với phần
"train" (`../fakedet_vlm`). Package này **không import torch, không nạp
model** — nó chạy trên máy 4 GB của bạn. Model đã train chạy trên **GPU
cloud** (Colab) và được gọi qua HTTP.

```
open-webui (local)
   │  OpenAI API
   ▼
fakedet_system  (local, KHÔNG cần GPU)        ← package này
   ├─ intent     →  API ngoài (Gemini / OpenAI-compat)
   ├─ memory     →  SQLite (local, nhớ cả ảnh cuối)
   └─ answer ─┬─ ảnh    →  fakedet_vlm.serve (GPU cloud) → model đã train
              ├─ ngữ cảnh→  Gemini (+ ảnh trước) — "ăn gian"
              └─ ngoài scope → câu tĩnh
```

Pipeline là một **đường thẳng** (Haystack `Pipeline`, là DAG có kiểu nhưng
luồng đi tuyến tính — **không có vòng lặp chọn tool, không agentic**):

```
InputParser → IntentClassifier → MemoryReader → PromptBuilder
            → FakeDetClient → MemoryWriter → (reply)
```

- **InputParser** — tách text + ảnh từ request OpenAI.
- **IntentClassifier** — có ảnh ⇒ `image_deepfake` (không gọi API). Không
  ảnh ⇒ **API ngoài** chọn 1 trong: `text_to_detect` / `user_guide` /
  `misleading`. Không key/API lỗi ⇒ heuristic từ khoá (mặc định
  `misleading` nếu không thấy gì trong phạm vi → hỏi linh tinh không bị
  nhận nhầm).
- **MemoryReader / MemoryWriter** — lịch sử hội thoại + core memory +
  **ảnh cuối của session** (SQLite). Ảnh được nhớ để câu hỏi sau "ảnh
  trước sao lại giả" còn re-feed lại được.
- **PromptBuilder** — biến (intent + ngữ cảnh) thành 1 `plan` (route cố
  định theo intent — không agentic).
- **Responder** — chạy `plan`:

| Intent | Nghĩa | Ai trả lời |
|---|---|---|
| `image_deepfake` | Có ảnh → thật/giả + bằng chứng | **Model đã train** (vision+LLM, qua HTTP) |
| `text_to_detect` | Hỏi tiếp theo ngữ cảnh ("ảnh trước sao giả", "phân tích kĩ hơn", "tôi hỏi mấy ảnh rồi") | **Gemini** + memory + ảnh trước |
| `user_guide` | Hỏi về hệ thống | Gemini (theo ngữ cảnh) |
| `misleading` | **Ngoài phạm vi** (thời tiết, code, linh tinh) | Câu thông báo tĩnh "ngoài phạm vi" |

> Ảnh deepfake **luôn do model đã train** quyết định. Các câu hỏi sâu /
> theo ngữ cảnh được "ăn gian" bằng VLM ngoài to (Gemini). Nếu chưa cấu
> hình key Gemini, route gemini tự fallback sang nhánh sinh text của model
> đã train (`fakedet_vlm.serve._run_text`) — hệ thống luôn trả lời được.

---

## Chạy nhanh nhất — Docker (1 lệnh)

Yêu cầu: Docker Desktop. Backend model đã chạy ở HF Space (hoặc Colab).

1. Sửa `fakedet_system/.env` cho đầy đủ (key Gemini + `FAKEDET_BACKEND_URL`
   trỏ Space, ví dụ `https://duongvan17-fakedet-vlm-backend.hf.space/v1`).
2. Build + chạy:
   ```powershell
   cd fakedet_system
   docker compose up -d --build
   ```
3. Mở <http://localhost:8088> — không cần login, chọn model
   `fakedet-v1.2`, chat luôn.

Sau khi dùng xong:
```powershell
docker compose down            # giữ data (memory, chat history)
docker compose down -v         # xoá hẳn (cả memory)
```

Compose dựng 2 service: `fakedet-system` (pipeline, port nội bộ 9000) và
`open-webui` (UI, ánh xạ ra 8088). Đã **tắt sẵn** các request ngầm của
open-webui (title/tags/follow-up...) cho khỏi tốn lượt gọi backend.

---

## Chạy đầy đủ (4 bước, không Docker)

### 1. Bật backend model trên GPU cloud

Theo `../fakedet_vlm/CHAY_FREE.md` (Colab + cloudflared). Kết quả là một
URL kiểu `https://xxxx.trycloudflare.com`. Backend nói giao thức OpenAI ở
`/v1`.

### 2. Cài system ở máy local

```powershell
cd fakedet_system
pip install -e ".[serve]"
copy .env.example .env
```

Sửa `.env`:

```
FAKEDET_INTENT_API_KEY=<API key Gemini của bạn>
FAKEDET_INTENT_MODEL=gemini-2.0-flash
FAKEDET_BACKEND_URL=https://xxxx.trycloudflare.com/v1   # URL ở bước 1
```

Dùng provider khác (Groq/OpenAI-compat) thì đặt
`FAKEDET_INTENT_PROVIDER=openai`, `FAKEDET_INTENT_BASE_URL=...`.
Để trống key ⇒ intent chạy bằng heuristic (vẫn hoạt động).

### 3. Chạy system

```powershell
uvicorn fakedet_system.serve.openai_api:app --host 0.0.0.0 --port 9000
```

Kiểm tra: `http://localhost:9000/health` →
`{"status":"ok","intent_provider":"gemini","model_backend":"https://.../v1"}`

### 4. open-webui trỏ vào system (KHÔNG trỏ thẳng vào model)

```powershell
docker run -d -p 3000:8080 `
  -e OPENAI_API_BASE_URL=http://host.docker.internal:9000/v1 `
  -e OPENAI_API_KEY=sk-fakedet `
  -e WEBUI_AUTH=false `
  --name open-webui `
  --add-host=host.docker.internal:host-gateway `
  ghcr.io/open-webui/open-webui:main
```

Mở <http://localhost:3000>, chọn model `fakedet-system`:

- Gửi **ảnh** → `image_deepfake` → verdict 🟥FAKE/🟩REAL + bằng chứng
  (model đã train, vision + LLM).
- Hỏi tiếp **"ảnh vừa rồi sao lại giả, phân tích kĩ hơn"** →
  `text_to_detect` → Gemini xem lại ảnh trước + ngữ cảnh, giải thích sâu.
- Hỏi **"hệ thống này dùng sao"** → `user_guide` → Gemini trả lời.
- Hỏi **"thời tiết hôm nay"** / linh tinh → `misleading` → báo ngoài
  phạm vi.

Mỗi luồng hội thoại open-webui được tự gán 1 session (hash message đầu),
nên memory hoạt động mà không cần sửa client. Muốn chỉ định session thì
gửi header `X-Session-Id`.

---

## Vì sao tách 2 package

| | `fakedet_vlm` | `fakedet_system` (đây) |
|---|---|---|
| Vai trò | TRAIN + MODEL + serve model | SYSTEM: pipeline, intent, memory |
| Phụ thuộc | torch, transformers, peft, bnb | haystack-ai, httpx (nhẹ) |
| Chạy ở | GPU cloud | máy local 4 GB |
| Đổi cái này không ảnh hưởng cái kia | ✔ | ✔ |

`fakedet_system` chỉ "biết" model qua HTTP (`FAKEDET_BACKEND_URL`).

---

## Dev / test

```powershell
pip install -e ".[dev]"
python -m pytest -q          # 18 test, mock cả API ngoài lẫn backend
python -m ruff check src tests
```

Test không cần GPU, không cần mạng (mock bằng `respx`, SQLite `:memory:`).

## Cấu hình (env)

Xem `.env.example`. Tóm tắt:

- `FAKEDET_INTENT_PROVIDER` `gemini`|`openai` · `FAKEDET_INTENT_API_KEY`
  · `FAKEDET_INTENT_MODEL` · `FAKEDET_INTENT_BASE_URL`
- `FAKEDET_BACKEND_URL` (model đã train, có `/v1`) · `FAKEDET_BACKEND_KEY`
  · `FAKEDET_MODEL_NAME`
- `FAKEDET_MEMORY_DB` · `FAKEDET_MEMORY_TURNS`
- `FAKEDET_SYSTEM_MODEL_ID` (tên hiện trong open-webui)
