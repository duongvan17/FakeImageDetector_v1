# Chạy model free trên cloud

Máy yếu không chạy được model 1.5B. Dùng 1 trong 2 cách dưới — đều **miễn phí**.

---

## Cách 1: Google Colab (nhanh nhất, test/dev)

Free T4 GPU 16GB, ~12h/ngày. Phù hợp test nhanh + demo tạm.

### Tạo notebook mới: https://colab.research.google.com

**Quan trọng**: Runtime → Change runtime type → **T4 GPU** → Save.

### Cell 1 — Cài deps + clone code

```python
!pip install -q "transformers>=4.46,<4.50" "peft>=0.12,<0.14" \
    bitsandbytes accelerate "timm>=1.0.9" pillow huggingface_hub gradio
!git clone https://github.com/duongvan17/FakeImageDetector_v1.git
%cd FakeImageDetector_v1/fakedet_vlm
!pip install -q -e . --no-deps
```

### Cell 2 — Login HF (cần Read token)

```python
from huggingface_hub import login
login()   # paste Read token: https://huggingface.co/settings/tokens
```

### Cell 3 — Chạy Gradio app

```python
%cd /content/FakeImageDetector_v1/fakedet_vlm
!python app.py
```

→ App load model (~2 phút lần đầu), in ra link công khai dạng
`https://xxxxx.gradio.live` — mở link đó, upload ảnh, xem kết quả.

Link sống ~72h. Demo cho thầy/hội đồng được luôn.

---

## Cách 2: HuggingFace Spaces (demo cố định cho thesis)

URL vĩnh viễn, không cần bật máy. Free ZeroGPU.

### Bước 1: Tạo Space

1. https://huggingface.co/new-space
2. Tên: `fakedet-vlm-demo`
3. SDK: **Gradio**
4. Hardware: **CPU basic (free)** hoặc **ZeroGPU (free, nhanh hơn)**
5. Create

### Bước 2: Upload 2 file vào Space

Trong Space → Files → Add file:

**`app.py`** — copy từ `fakedet_vlm/app.py` repo.

**`requirements.txt`**:

```
torch>=2.2
torchvision>=0.17
transformers>=4.46,<4.50
peft>=0.12,<0.14
bitsandbytes>=0.43
accelerate>=0.34
timm>=1.0.9
pillow>=10
huggingface_hub>=0.24,<0.27
gradio>=4
git+https://github.com/duongvan17/FakeImageDetector_v1.git#subdirectory=fakedet_vlm
```

### Bước 3: Set HF token secret

Space → Settings → Secrets → New secret:
- Name: `HF_TOKEN`
- Value: Read token của bạn

(Cần vì 2 model repo đang private. Hoặc đổi 2 repo sang public thì khỏi cần.)

### Bước 4: Đợi build

Space tự build (~5-10 phút). Xong có URL:
`https://huggingface.co/spaces/duongvan17/fakedet-vlm-demo`

→ Demo vĩnh viễn, paste vào thesis/slide.

---

## So sánh

| | Colab | HF Spaces |
|---|---|---|
| Free | ✅ | ✅ |
| GPU | T4 16GB | ZeroGPU (chia sẻ) |
| Tốc độ inference | ~2-3s/ảnh | ~3-5s/ảnh |
| URL bền | ~72h | Vĩnh viễn |
| Cần bật máy | Có (Colab session) | Không |
| Phù hợp | Dev, test, demo nhanh | Demo defense, share |

→ **Khuyên**: Colab để test trước. Nếu OK thì deploy HF Spaces cho demo cố định.

---

## CPU-only (nếu không có GPU nào, rất chậm)

Inference trên CPU được nhưng ~30-60s/ảnh (Qwen2.5-1.5B fp16):

```python
det = DeepfakeDetector(
    llm_name="Qwen/Qwen2.5-1.5B-Instruct",
    vision_checkpoint="clip_model/best_model.pth",
    adapter_dir="vlm_final",
    projector_path="vlm_final/projector.pt",
    device="cpu",
)
```

`load_in_4bit` tự tắt khi device=cpu (bitsandbytes cần CUDA). Dùng fp16 →
chậm nhưng vẫn ra kết quả. Chỉ dùng khi không có lựa chọn nào khác.

---

## Lỗi thường gặp

### `OutOfMemoryError` trên Colab

T4 16GB đủ cho 4-bit Qwen2.5-1.5B (~5GB peak). Nếu vẫn OOM:
- Restart runtime (Runtime → Restart)
- Đảm bảo chọn T4, không phải CPU

### `401 Unauthorized` khi download model

Token Read chưa set hoặc sai. Chạy lại `login()` với token đúng từ
https://huggingface.co/settings/tokens

### Space build fail

Xem tab "Logs" trên Space. Thường do version conflict — đảm bảo
`requirements.txt` pin `transformers<4.50` (tránh bug torch.library).

### Gradio link không mở được

Colab `share=True` link sống 72h. Hết hạn → chạy lại Cell 3.
