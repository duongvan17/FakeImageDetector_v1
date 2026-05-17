# Chạy model trên HF Spaces — Docker, free CPU basic

Backend = FastAPI OpenAI-compatible (`fakedet_vlm.serve.openai_api`) chạy
trong **Docker Space**. **Không dùng Gradio** (Gradio 4.44 + mount FastAPI
bị bug `TypeError: 'bool' is not iterable`). URL Space cố định →
`fakedet_system` trỏ 1 lần.

> Free thật: **CPU basic** (không cần PRO). Chậm ~30–60s/ảnh nhưng 24/7,
> URL vĩnh viễn, không cần bật máy. Cần nhanh lúc bảo vệ → Colab T4
> (`../CHATGPT_UI.md` Cách 4), đổi `FAKEDET_BACKEND_URL` tạm thời.

## 3 file của Space

Trong `fakedet_vlm/hf_space/`: **`Dockerfile`**, **`start.py`**,
**`README.md`** (frontmatter `sdk: docker`). Không có `app.py`/
`requirements.txt` nữa.

## Bước 1 — Space đã có sẵn (đã tạo)

Space: `https://huggingface.co/spaces/duongvan17/fakedet-vlm-backend`
SDK sẽ tự thành **docker** theo frontmatter `README.md`.

## Bước 2 — Secret HF_TOKEN (BẮT BUỘC)

Space → **Settings → Variables and secrets → New secret**
- Name `HF_TOKEN` · Value = **Read** token
  (<https://huggingface.co/settings/tokens>)

2 repo model private; `start.py` tải bằng token này. Thiếu → crash 401.

## Bước 3 — Hardware

Space → **Settings → Hardware** = **CPU basic** (free). Đúng rồi để yên.

## Bước 4 — Đẩy 3 file (đã script sẵn)

Bản chuẩn nằm trong repo code `fakedet_vlm/hf_space/`. Khi sửa thì sửa ở
đó rồi copy sang clone Space (`D:\fakedet-vlm-backend`), `git add -A` →
commit → push. Space tự build lại (~8–12 phút: cài torch CPU +
transformers + clone code).

## Bước 5 — Test

Khi Space xanh **Running**:

```powershell
curl https://duongvan17-fakedet-vlm-backend.hf.space/v1/models
# → {"object":"list","data":[{"id":"fakedet-vlm",...}]}
curl https://duongvan17-fakedet-vlm-backend.hf.space/health
# → {"status":"ok","loaded":false}   (false = chưa nạp, đúng vì LAZY_LOAD)
```

`/v1/models` + `/health` trả ngay (model nạp lười). Lần
`/v1/chat/completions` đầu **rất chậm** (CPU nạp + suy luận model) — bình
thường.

## Bước 6 — Trỏ fakedet_system vào

`fakedet_system/.env`:

```
FAKEDET_BACKEND_URL=https://duongvan17-fakedet-vlm-backend.hf.space/v1
FAKEDET_BACKEND_TIMEOUT_S=300
```

Rồi chạy hệ thống local theo
[../../fakedet_system/README_HE_THONG.md](../../fakedet_system/README_HE_THONG.md).

## Lỗi thường gặp

- **Build fail – torch quá nặng / hết dung lượng:** Dockerfile đã dùng
  wheel CPU (`download.pytorch.org/whl/cpu`) cho nhẹ. Đừng đổi sang bản
  CUDA.
- **401 / Repository Not Found:** thiếu/sai secret `HF_TOKEN` (cần *Read*).
- **Space "Running" nhưng `/v1/chat/completions` timeout lần đầu:** CPU
  nạp model 1.5B lâu — đã để `FAKEDET_BACKEND_TIMEOUT_S=300`; gọi
  `/v1/models` trước cho Space tỉnh.
- **OOM:** CPU basic 16GB đủ cho Qwen2.5-1.5B fp32 (~6GB) + ViT. Nếu vẫn
  OOM, đặt thêm Space variable `FAKEDET_MAX_NEW_TOKENS=48`.
- **Đổi hardware/SDK xong Space lỗi:** Settings → **Factory reboot**.
