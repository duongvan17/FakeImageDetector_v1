# Hướng dẫn thuê + chạy GPU Vast.ai (RTX 4090)

Hướng dẫn step-by-step thuê GPU lần đầu trên Vast.ai và train xong VLM.
Targeting pod #30709378 (Hungary, 1x RTX 4090, $0.340/hr) — nhưng quy trình
áp dụng cho mọi pod 4090 24GB.

> **Tổng chi phí dự kiến**: ~$3-4 cho lần train đầu
> (gồm $2 GPU + $1.5 bandwidth + $0.5 buffer cho lỗi).

---

## Mục lục

1. [Chuẩn bị local (5 phút)](#1-chuẩn-bị-local-5-phút)
2. [Add credits vào Vast.ai](#2-add-credits-vào-vastai)
3. [Setup SSH key](#3-setup-ssh-key)
4. [Rent pod](#4-rent-pod)
5. [Kết nối vào pod](#5-kết-nối-vào-pod)
6. [Setup môi trường](#6-setup-môi-trường-trên-pod)
7. [Train (cách tốt nhất ⭐)](#7-train)
8. [Theo dõi tiến trình](#8-theo-dõi-tiến-trình)
9. [Download model về local](#9-download-model-về-local)
10. [Stop pod](#10-stop-pod)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Chuẩn bị local (5 phút)

Đảm bảo trên máy local đã có:

- ✅ Code đã push lên GitHub (`git push origin main`)
- ✅ `best_model.pth` đã upload HF Hub (đã làm xong)
- ✅ HF token (Read scope) sẵn sàng

Verify nhanh:

```bash
cd d:/FakeImageDetector_v1
git status                                 # working tree clean
git log --oneline -1                       # commit refactor đã có
```

Nếu chưa push, làm:

```bash
git push origin main
```

---

## 2. Add credits vào Vast.ai

Vast.ai prepaid — phải nạp credit trước khi rent.

1. Vào <https://cloud.vast.ai/billing/>
2. Bấm **"Add Credit"**
3. Nạp **$10** (đủ cho 2-3 lần train + buffer)
4. Thanh toán bằng credit card hoặc crypto
5. Đợi 1-2 phút credit về account

> **Lưu ý**: Vast charge theo giây. Pod chạy 6h tiêu ~$2.04. Stopped pod
> không tiêu GPU money nhưng vẫn tiêu disk storage rất nhỏ (~$0.003/GB/h).

---

## 3. Setup SSH key

Vast.ai dùng SSH key để vào pod (an toàn hơn password).

### Kiểm tra key đã có chưa

```bash
ls ~/.ssh/id_*.pub 2>/dev/null
# Nếu thấy id_rsa.pub hoặc id_ed25519.pub → đã có, skip bước tạo
```

### Tạo key mới (nếu chưa có)

Trong Git Bash:

```bash
ssh-keygen -t ed25519 -C "vast-ai" -f ~/.ssh/id_ed25519 -N ""
# Enter để bỏ qua passphrase
```

### Copy public key

```bash
cat ~/.ssh/id_ed25519.pub
# In ra: ssh-ed25519 AAAAC3Nza... vast-ai
# Copy toàn bộ dòng này
```

### Paste vào Vast.ai

1. Vào <https://cloud.vast.ai/account/>
2. Tab **"SSH Keys"**
3. Bấm **"+ NEW"**
4. Paste public key vào ô
5. Bấm **Save**

Done.

---

## 4. Rent pod

### Tìm pod #30709378

1. Vào <https://cloud.vast.ai/create/>
2. Trong panel **Filter Options** bên trái:
   - **GPU**: chọn `RTX 4090`
   - **Per GPU RAM**: 24 GB
   - **Reliability**: 99%+
3. Trong list pods, **Ctrl+F** tìm `30709378` (hoặc scroll tìm Hungary 4090
   $0.340/hr)
4. Click vào row đó

### Configure pod

Khi click vào pod, panel cấu hình hiện ra phía trên cùng:

| Setting | Giá trị | Ghi chú |
|---|---|---|
| **Template** (Image) | Bấm **"Edit Image & Config"** | |
| → trong dialog, search `pytorch` | chọn `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` | Image chuẩn có Python + PyTorch + CUDA sẵn |
| **Container Disk** | Kéo lên **70 GB** | Default 16GB không đủ |
| **Open Ports** | Thêm `8000` (TCP) | Optional, nếu muốn test FastAPI |
| **Launch Mode** | SSH | Không cần Jupyter |

### Bấm "Rent"

Pod sẽ status `creating` → `loading` → `running` (mất 1-3 phút).

Vào tab **Instances** ở thanh trái để xem pod.

---

## 5. Kết nối vào pod

Khi pod đã `running`:

1. Trong **Instances**, tìm pod vừa rent
2. Bấm icon **>_** (Connect) hoặc **"OPEN"**
3. Hiện dialog có command kiểu:
   ```
   ssh -p 12345 root@ssh4.vast.ai -L 8080:localhost:8080
   ```
4. **Copy command đó**

### Chạy SSH command trong Git Bash

```bash
ssh -p 12345 root@ssh4.vast.ai
# Lần đầu sẽ hỏi: "Are you sure you want to continue connecting (yes/no/[fingerprint])?"
# Gõ: yes
```

Nếu thấy prompt `root@C.vastai.io:~#` → vào được pod ✅

---

## 6. Setup môi trường trên pod

Trên pod (đang là root):

### 6.1 Clone code từ GitHub

```bash
cd /workspace                              # thư mục mặc định trên Vast
git clone https://github.com/duongvan17/FakeImageDetector_v1.git
cd FakeImageDetector_v1/fakedet_vlm
```

> Nếu repo private → cần Personal Access Token GitHub:
> ```bash
> git clone https://<token>@github.com/duongvan17/FakeImageDetector_v1.git
> ```

### 6.2 Tạo HF read token

Cần token Read scope (khác token Write đã dùng để upload):

1. Mở browser local: <https://huggingface.co/settings/tokens>
2. Bấm **"Create new token"** → type **Read** → tên `cloud-read`
3. Copy token (`hf_xxx...`)

### 6.3 Set environment variables

Trên pod:

```bash
export HF_TOKEN=hf_xxx_paste_token_read_o_day
export HF_CKPT_REPO=duongvan17/fakedet-vit-b16-fakeclue
```

### 6.4 Chạy setup script

```bash
bash scripts/setup_cloud.sh
```

Script này sẽ tự động:

| Bước | Thời gian |
|---|---|
| Print system info (verify GPU) | 1s |
| Tạo virtualenv | 5s |
| Cài PyTorch CUDA 12.1 | 1-2 phút |
| Cài project deps (transformers, peft, bnb, ...) | 2-3 phút |
| Login HF | 5s |
| Download `best_model.pth` từ HF Hub | 30-60s |
| Verify ViT load | 5s |
| Run pytest 10 tests | 30s |

**Tổng: ~5 phút**.

Output cuối khi thành công:

```
[7/7] Sanity check
      OK — 85.8M params, frozen=True
      output shape = (2, 196, 768)  (expected: (2, 196, 768))
OK — vision tower loads and runs correctly.
.......... [100%]
10 passed in 12.5s

Setup complete. Next steps:
  tmux                      # avoid disconnect kills
  make data                 # ~30-60 min, ~30 GB
  make stage1               # ~2-3 h on RTX 4090
  ...
```

Nếu lỗi gì → xem [Troubleshooting](#11-troubleshooting).

---

## 7. Train

### 7.0 Hiểu flow trước khi chạy — RẤT QUAN TRỌNG

Training là **2-stage tuần tự, KHÔNG phải 1 lệnh chạy 1 phát**:

```
[Stage 1: Projector Warmup]   4-5h trên 3090 Ti / 2-3h trên 4090
  ↓ trainable: chỉ projector (3M params)
  ↓ output: runs/stage1_align/projector.pt
[Stage 2: LoRA SFT]            5-6h trên 3090 Ti / 3-4h trên 4090
  ↓ trainable: projector + LoRA (15M params)
  ↓ output: runs/stage2_sft/final/{adapter_model.safetensors, projector.pt}
[Eval]                          20-30 phút
  ↓ output: runs/stage2_sft/eval/{eval_metrics.json, eval_predictions.jsonl}
```

**Tại sao 2 stage?** Đây là best practice của LLaVA / FakeVLM:

- Stage 1: align projector với LLM space trước. Nếu skip → projector random
  output làm noise gradient signal cho LoRA → loss khó hội tụ
- Stage 2: LLM (qua LoRA) học task dựa trên features đã align
- Bỏ stage 1 → accuracy thấp 10-15% (không đáng tiết kiệm 3-4h)

**Cơ chế resume**: HF Trainer auto save checkpoint mỗi 500 steps vào
`runs/<stage>/checkpoint-XXXX/`. Nếu crash giữa chừng, resume với:

```bash
python -m fakedet_vlm.train.stage1 \
    --base configs/base.yaml --stage configs/stage1.yaml \
    --resume_from_checkpoint runs/stage1_align/checkpoint-2000
```

### 7.1 Cách train tốt nhất (recommended workflow)

**Pattern**: Chạy từng lệnh, kiểm tra 1-2 phút đầu mỗi stage để chắc loss
giảm bình thường, rồi detach tmux đi ngủ. KHÔNG chain `&&` cho lần đầu vì
nếu stage 1 toang thì stage 2 cũng toang — đốt 6h vô ích.

```bash
# Pha A: chuẩn bị (~10s)
tmux new -s train
source .venv/bin/activate

# Pha B: data (~30-60 phút, không cần GPU)
make data
# Kiểm tra log cuối: phải có "Train=XXXXX Val=XXXX" + per-category stats
# Nếu chỉ vài trăm samples → có gì sai, dừng lại

# Pha C: stage 1 (~4-5h trên 3090 Ti)
make stage1
# Xem log 2 phút đầu: loss bắt đầu ~4.3, giảm dần
# Nếu sau 100 steps loss vẫn ~4.3 không giảm → có bug, Ctrl+C
# OK rồi → Ctrl+B D detach, đi ngủ / đi làm việc khác

# Pha D: kiểm tra stage 1 xong
# Reattach: tmux a -t train
# Cuối log phải thấy: "[stage1] saved projector → runs/stage1_align/projector.pt"
# Loss epoch cuối phải <3.0 (ideal <2.5)
ls -la runs/stage1_align/projector.pt
# File phải tồn tại ~12 MB

# Pha E: stage 2 (~5-6h)
make stage2
# Loss đầu ~3.0, cuối ~0.4-0.8
# Sau 100 steps loss giảm dưới 2.5 → OK, detach
# Nếu loss kẹt >2.5 → projector stage 1 có vấn đề, dừng debug

# Pha F: eval (~30 phút)
make eval
# In overall + per-category metrics
# Target: acc >0.85, F1 >0.80
```

**Nếu muốn chạy 1 phát đi ngủ luôn** (sau khi đã quen):

```bash
tmux new -s train
source .venv/bin/activate
make data && make stage1 && make stage2 && make eval
# Ctrl+B D, sáng dậy xem kết quả
```

`&&` đảm bảo nếu lệnh nào fail thì dừng ngay, không tiếp tục.

### 7.2 Mở tmux

**RẤT QUAN TRỌNG** — tmux giữ tiến trình chạy ngay cả khi mất kết nối SSH.
Không có tmux → mất mạng = kill train = mất tiền.

```bash
tmux new -s train
```

Bạn sẽ thấy thanh xanh dưới cùng — đang trong tmux session tên "train".

### 7.3 Activate venv (sau khi setup xong)

```bash
source .venv/bin/activate
```

### 7.4 Tải FakeClue dataset

```bash
make data
```

- Thời gian: 30-60 phút
- Tiêu thụ disk: 30-35 GB
- Khi xong sẽ in stats per-category:

```
[train] total=89421  real=44103 (49.3%)  fake=45318 (50.7%)
  per-category:
    deepfake          24532 (27.4%)
    ...
```

### 7.5 Stage 1: Projector alignment (~2-3h trên 4090, ~4-5h trên 3090 Ti)

```bash
make stage1
```

Theo dõi loss giảm dần:

```
{'loss': 4.32, 'grad_norm': 8.2, 'learning_rate': 1.5e-4, 'epoch': 0.05}
{'loss': 3.89, 'grad_norm': 6.1, 'learning_rate': 3.0e-4, 'epoch': 0.10}
...
{'loss': 2.41, ...}                      ← cuối epoch 1, target ~2.5
```

Output cuối:

```
[stage1] saved projector → runs/stage1_align/projector.pt
```

### 7.6 Stage 2: LoRA SFT (~2-3h trên 4090, ~5-6h trên 3090 Ti)

```bash
make stage2
```

Theo dõi:

```
{'loss': 3.12, ...}                      ← bắt đầu
...
{'loss': 0.67, ...}                      ← cuối epoch 2, target 0.4-0.8
```

Output cuối:

```
[stage2] saved LoRA + projector + tokenizer → runs/stage2_sft/final
```

### 7.7 Eval

```bash
make eval
```

In ra:

```
[Overall] n=9936  acc=0.912  P=0.918 R=0.905 F1=0.911 AUC=0.962
[Per-category]
  animal           n=  920  acc=0.945  F1=0.943
  ...
```

🎉 Model train xong!

---

## 8. Theo dõi tiến trình

### Detach khỏi tmux (nhưng training vẫn chạy)

Trong tmux: bấm **`Ctrl+B`** rồi **`D`**.

Bạn sẽ về terminal SSH bình thường. Có thể:
- `exit` SSH (training vẫn chạy trên pod)
- Tắt máy local — pod vẫn train tiếp

### Reattach để xem progress

Vào lại pod:

```bash
ssh -p 12345 root@ssh4.vast.ai
tmux a -t train
```

### Theo dõi GPU usage (terminal khác)

Mở terminal SSH thứ 2 vào pod:

```bash
watch -n 2 nvidia-smi
# Sẽ in mỗi 2s: GPU memory, utilization, power
```

VRAM peak nên ~9-10 GB; util ~95-100% khi đang train.

### Xem log file

```bash
# Trong tmux đang chạy stage1/stage2, log đã in trên màn hình.
# Nếu muốn dump ra file:
tail -f runs/stage1_align/training_log.txt
# (HF Trainer tự log nếu report_to chưa "none")
```

---

## 9. Download model về local

Sau khi train + eval xong, tải model về (chỉ ~50 MB).

### Option A: SCP (đơn giản)

Trên local (mở terminal Git Bash mới):

```bash
cd d:/FakeImageDetector_v1/fakedet_vlm
mkdir -p runs/stage2_sft

scp -r -P 12345 \
    root@ssh4.vast.ai:/workspace/FakeImageDetector_v1/fakedet_vlm/runs/stage2_sft/final \
    runs/stage2_sft/

# Cũng tải kết quả eval:
scp -r -P 12345 \
    root@ssh4.vast.ai:/workspace/FakeImageDetector_v1/fakedet_vlm/runs/stage2_sft/eval \
    runs/stage2_sft/
```

(Đổi `12345` thành port SSH thực, xem trong dialog Connect của Vast.)

### Option B: Upload model lên HF Hub (chia sẻ dễ hơn)

Trên pod:

```bash
# Login với write token (token cũ đã dùng để upload checkpoint)
hf auth login --token $HF_WRITE_TOKEN

# Upload toàn bộ artefacts
hf upload duongvan17/fakedet-vlm-stage2 \
    runs/stage2_sft/final \
    --repo-type model \
    --private
```

Sau đó local chỉ cần:

```bash
hf download duongvan17/fakedet-vlm-stage2 \
    --local-dir runs/stage2_sft/final
```

---

## 10. Stop pod

**CỰC KỲ QUAN TRỌNG** — quên stop = burn money 24/7.

### Stop (giữ data + setup, có thể start lại sau)

1. Vào tab **Instances** trên Vast
2. Tìm pod, bấm **STOP** (icon ⏸)
3. Pod chuyển trạng thái `stopped`
4. Tiền GPU = $0/h. Tiền disk vẫn nhỏ ~$0.003/GB/h ($0.05/ngày cho 70GB)

### Destroy (xóa hẳn, mất data)

Khi không cần dùng nữa:

1. Trong **Instances**, bấm **DESTROY** (icon 🗑)
2. Confirm
3. Pod biến mất hoàn toàn, free disk

> **Workflow hợp lý**: Train xong → SCP model về local → DESTROY pod luôn.
> Lần sau train lại tạo pod mới (5 phút setup).

---

## 11. Troubleshooting

### SSH "Connection refused" / "timeout"

Pod chưa thực sự ready dù status `running`. Đợi thêm 1-2 phút rồi thử lại.

### `git clone` báo Permission denied (publickey)

Repo public dùng HTTPS không cần key. Nếu repo của bạn private, cần PAT:

1. <https://github.com/settings/tokens> → Create token (classic) scope `repo`
2. Clone với token:
   ```bash
   git clone https://<token>@github.com/duongvan17/FakeImageDetector_v1.git
   ```

### `bash scripts/setup_cloud.sh` báo `nvidia-smi: command not found`

Bạn chọn nhầm template không có CUDA. Destroy pod, rent lại với image
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`.

### `pip install bitsandbytes` báo lỗi build

Có thể driver/CUDA mismatch. Thử:

```bash
pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
```

### `make data` chậm bất thường (KB/s thay vì MB/s)

Pod gặp pod xấu hoặc HF Hub regional throttle. Thử:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Mirror châu Á
make data
```

### Stage 2 báo OOM (out of memory)

Mở `configs/stage2.yaml`, giảm:

```yaml
data:
  max_length: 768                          # từ 1024
train:
  gradient_accumulation_steps: 8           # từ 16
lora:
  r: 8                                     # từ 16
```

Re-run `make stage2`.

### Disconnect giữa train

Nếu dùng tmux thì training vẫn chạy. Reconnect:

```bash
ssh -p 12345 root@ssh4.vast.ai
tmux a -t train
```

Nếu không có tmux thì training đã chết. HF Trainer save checkpoint mỗi 500
steps → resume:

```bash
python -m fakedet_vlm.train.stage2 \
    --base configs/base.yaml --stage configs/stage2.yaml \
    --resume_from_checkpoint runs/stage2_sft/checkpoint-XXXX
```

(XXXX = step cuối cùng đã save).

### Pod bị xóa đột ngột (provider terminate)

Hiếm khi xảy ra với Vast secure cloud. Nếu xảy ra:
- Mất disk + checkpoint trên pod
- Rent pod mới + restart từ đầu (loss đã lose)
- → Lý do nên upload checkpoint lên HF Hub thường xuyên (sau mỗi stage)

### Tiêu hết credit giữa train

Vast sẽ pause pod (không destroy ngay). Add thêm credit, pod tự resume.

---

## Tóm tắt commands theo thứ tự

```bash
# === TRÊN LOCAL ===
git push origin main                       # đã làm
# Upload best_model.pth lên HF              # đã làm

# === TRÊN VAST ===
# 1. Add credit + add SSH key (web UI)
# 2. Rent pod #30709378
# 3. SSH vào:
ssh -p <PORT> root@ssh4.vast.ai

# 4. Setup
cd /workspace
git clone https://github.com/duongvan17/FakeImageDetector_v1.git
cd FakeImageDetector_v1/fakedet_vlm
export HF_TOKEN=hf_xxx_read_token
export HF_CKPT_REPO=duongvan17/fakedet-vit-b16-fakeclue
bash scripts/setup_cloud.sh

# 5. Train trong tmux
tmux new -s train
source .venv/bin/activate
make data         # ~45 min
make stage1       # ~2.5 h
make stage2       # ~3 h
make eval         # ~20 min
# Detach: Ctrl+B, D

# === LOCAL: download kết quả ===
cd d:/FakeImageDetector_v1/fakedet_vlm
mkdir -p runs/stage2_sft
scp -r -P <PORT> root@ssh4.vast.ai:/workspace/FakeImageDetector_v1/fakedet_vlm/runs/stage2_sft/final runs/stage2_sft/
scp -r -P <PORT> root@ssh4.vast.ai:/workspace/FakeImageDetector_v1/fakedet_vlm/runs/stage2_sft/eval runs/stage2_sft/

# === DESTROY POD trên Vast ===
```

Total: **~6-7 giờ**, **~$2.5-4**.

---

## FAQ

**Q: Tôi disconnect giữa `make data` mà không tmux, làm sao?**
A: HF datasets sẽ resume từ chỗ break. Chạy lại `make data` thôi.

**Q: Stage 1 loss không giảm dưới 3.0, có sao không?**
A: Không sao nếu dataset bạn nhỏ (smoke test 200 mẫu). Full FakeClue thường về <2.5.

**Q: Stage 2 mất 5h thay vì 3h?**
A: Pod 4090 này có CPU EPYC 7R32 — đủ nhưng không nhanh nhất. PCIe x8 cũng giới hạn data load. Bình thường.

**Q: Tôi muốn train lại từ đầu, làm sao?**
A: `rm -rf runs/` rồi chạy lại từ stage1.

**Q: Có cách nào pause Stage 2 mà không mất tiến trình?**
A: HF Trainer auto save checkpoint. Bấm `Ctrl+C` trong tmux để stop. Lần
sau resume:
```bash
make stage2 ARGS="--resume_from_checkpoint runs/stage2_sft/checkpoint-XXXX"
```

---

**Chúc bạn train thành công!** Bất kỳ lỗi gì copy log báo về.
