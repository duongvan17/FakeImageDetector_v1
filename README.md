# Hệ Thống Phát Hiện Ảnh Giả - AI Fake Image Detector

> Model AI sử dụng Vision Transformer để phát hiện ảnh giả/thật với độ chính xác 94.66%

---

## 📋 Tổng Quan

Hệ thống sử dụng model **ViT-Base-CLIP** (86M parameters) được training trên FakeVLM dataset để phân biệt ảnh thật và ảnh giả.

**Kết quả:**
- Test Accuracy: **94.66%**
- Test AUC: **99.70%**
- Hỗ trợ 5 categories: chameleon, doc, ff++, genimage, satellite

---

## 📁 Cấu Trúc Project

```
phan_biet_anh_real_fake/
├── app.py                      # Flask web application
├── train_vision_fakeclue.py    # Script training model
├── predict_test.py             # Đánh giá model trên test set
├── infer_one.py                # Predict 1 ảnh đơn lẻ
├── check_gpu.py                # Kiểm tra GPU
├── checkpoints/
│   └── best_model.pth          # Model đã train (94.66% accuracy)
├── templates/
│   └── index.html              # Web UI
├── test_samples/               # Ảnh mẫu để test
│   ├── real/                   # Ảnh thật
│   └── fake/                   # Ảnh giả
├── TRAINING_GUIDE.md           # Hướng dẫn training chi tiết
├── CODE_EXPLANATION.md         # Giải thích code
├── requirements.txt            # Dependencies
├── QUICKSTART.txt              # Hướng dẫn nhanh
└── README.md                   # File này
```

---

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

**Phần cứng:**
- GPU NVIDIA với CUDA support (khuyến nghị 4GB+ VRAM)
- RAM: 8GB minimum, 16GB khuyến nghị
- Lưu trữ: 25GB (bao gồm dataset)

**Phần mềm:**
- Windows 10/11
- Python 3.11
- CUDA 12.1

### Bước 1: Tạo Virtual Environment

```bash



# Tạo virtual environment
python -m venv .venv

# Kích hoạt
.venv/Scripts/activate
```

### Bước 2: Cài Đặt Dependencies

```bash
# Cài PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài các thư viện khác
pip install -r requirements.txt
```

### Bước 3: Kiểm Tra GPU

```bash
python check_gpu.py
```

Kết quả mong đợi:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 2050
VRAM: 4096 MB
```

---

## 💻 Sử Dụng

### 0. Quick Demo (Không Cần Cài Đặt GPU)

**Test nhanh với ảnh mẫu có sẵn:**
```bash
# Activate virtual environment
.venv/Scripts/activate

# Test với ảnh thật
python infer_one.py test_samples/real/sample_real_1.jpg

# Test với ảnh giả
python infer_one.py test_samples/fake/sample_fake_1.jpg
```

**Kết quả mong đợi:**
```
Real image → Prediction: REAL (Confidence: 98%+)
Fake image → Prediction: FAKE (Confidence: 95%+)
```

### 1. Web Application (Demo)

**Khởi động web server:**
```bash
python app.py
```

**Truy cập:** http://localhost:5000

**Chức năng:**
- Upload ảnh để kiểm tra
- Hiển thị kết quả Real/Fake
- Confidence score

### 2. Predict 1 Ảnh Từ Command Line

```bash
python infer_one.py path/to/image.jpg
```

**Output:**
```
Image: path/to/image.jpg
Prediction: FAKE
Confidence: 98.5%
```

### 3. Đánh Giá Trên Test Set

```bash
python predict_test.py
```

Output: File `preds.csv` với predictions cho tất cả test images

### 4. Training Lại Model (Nếu Cần)

```bash
python train_vision_fakeclue.py --epochs 10 --batch_size 4 --patience 5
```

**Xem chi tiết:** `TRAINING_GUIDE.md`

---

## 📊 Hiệu Suất Model

### Tổng Thể
- **Accuracy:** 94.66%
- **AUC:** 99.70%
- **Thời gian training:** ~20 giờ (RTX 2050 4GB)

### Theo Category

| Category | Accuracy | Số Lượng Test |
|----------|----------|---------------|
| Satellite | 99.48% | 1,546 |
| Chameleon | 98.01% | 2,514 |
| FF++ | 95.72% | 1,168 |
| Genimage | 89.38% | 1,940 |
| Doc | 88.72% | 1,152 |

---

## 📖 Tài Liệu

### Hướng Dẫn Chi Tiết

1. **TRAINING_GUIDE.md** - Hướng dẫn training model đầy đủ
   - Giải thích thuật ngữ kỹ thuật
   - Pipeline training từng bước
   - Hyperparameters
   - Troubleshooting

2. **CODE_EXPLANATION.md** - Giải thích code chi tiết
   - Từng phần code
   - Các kỹ thuật sử dụng
   - Best practices

### Files Code

| File | Mô Tả |
|------|-------|
| `app.py` | Flask web app cho demo |
| `train_vision_fakeclue.py` | Training script chính |
| `predict_test.py` | Đánh giá trên test set |
| `infer_one.py` | Inference đơn lẻ |
| `check_gpu.py` | Kiểm tra GPU |

---

## 🔧 Kỹ Thuật Sử Dụng

### Model Architecture
- **Base:** ViT-Base-CLIP (OpenAI pretrained)
- **Parameters:** 86M
- **Input:** 224×224×3 RGB images
- **Output:** Binary classification (real/fake)

### Training Techniques
- **Mixed Precision (FP16):** Nhanh 2×, tiết kiệm 50% VRAM
- **Gradient Accumulation:** Effective batch size 32
- **Gradient Checkpointing:** Giảm 40% VRAM
- **Weighted Sampling:** Cân bằng imbalanced data
- **Early Stopping:** Tự động dừng khi overfit

---

## ⚠️ Lưu Ý

### Giới Hạn
1. **Selfies:** Model có hiệu suất thấp hơn trên selfies do không có trong training data
2. **Domain Mismatch:** Hoạt động tốt nhất với các loại ảnh giống training data
3. **GPU Required:** Cần GPU để chạy nhanh, CPU rất chậm

### Khuyến Nghị Sử Dụng
- ✅ Phát hiện deepfake từ các phương pháp đã biết
- ✅ Ảnh chuyên nghiệp, ảnh vệ tinh
- ✅ Ảnh generated từ GAN/Diffusion models
- ⚠️ Cần human review cho quyết định quan trọng
- ❌ Không dùng cho selfies hoặc ảnh social media

---

## 🆘 Hỗ Trợ

### Lỗi Thường Gặp

**1. CUDA Out of Memory**
```bash
# Giảm batch size trong train_vision_fakeclue.py
python train_vision_fakeclue.py --batch_size 2
```

**2. Model Load Lỗi**
- Kiểm tra file `checkpoints/best_model.pth` có tồn tại
- Re-download nếu file bị corrupt

**3. Import Error**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Liên Hệ Support
- Xem file `TRAINING_GUIDE.md` phần Troubleshooting
- Xem file `CODE_EXPLANATION.md` để hiểu code

---

## 📝 Changelog

### Version 1.0 (2025-12-01)
- ✅ Model ViT-Base-CLIP training xong (94.66% accuracy)
- ✅ Web application Flask
- ✅ Inference scripts
- ✅ Documentation đầy đủ tiếng Việt

---

## 📄 License

Dự án này sử dụng pretrained model từ OpenAI CLIP.

---

## 🙏 Acknowledgments

- **Model:** ViT-Base-CLIP from OpenAI
- **Dataset:** FakeVLM
- **Framework:** PyTorch + timm

---

**Phát triển bởi:** AI Training Pipeline  
**Ngày hoàn thành:** 2025-12-01
