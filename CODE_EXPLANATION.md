# Giải Thích Code - train_vision_fakeclue.py

> Hướng dẫn chuyên nghiệp cho pipeline training Vision Transformer

---

## Mục Lục

1. [Imports & Thư Viện](#1-imports--thư-viện)
2. [Cấu Hình Logging](#2-cấu-hình-logging)
3. [Class Dataset](#3-class-dataset)
4. [Weighted Sampler](#4-weighted-sampler)
5. [Hàm Training](#5-hàm-training)
6. [Hàm Validation](#6-hàm-validation)
7. [Hàm Main](#7-hàm-main)

---

## 1. Imports & Thư Viện

### Thư Viện Cơ Bản

```python
import os
import argparse
import logging
from pathlib import Path
from collections import Counter
import numpy as np
```

**Mục đích:**
- `os`: Cấu hình biến môi trường
- `argparse`: Phân tích tham số dòng lệnh (--epochs, --batch_size)
- `logging`: Ghi log quá trình training
- `Path`: Xử lý đường dẫn cross-platform
- `Counter`: Đếm tần suất để cân bằng classes
- `numpy`: Xử lý mảng số học

### Hệ Sinh Thái PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
```

**Các thành phần:**
- `torch`: Thư viện PyTorch core
- `nn`: Module mạng neural (layers, loss functions)
- `optim`: Thuật toán tối ưu hóa
- `Dataset/DataLoader`: Abstractions để load dữ liệu
- `WeightedRandomSampler`: Sampling cân bằng classes
- `autocast/GradScaler`: Thành phần **Mixed Precision Training**

**Thuật Ngữ Kỹ Thuật - Mixed Precision (Độ Chính Xác Hỗn Hợp):**
- Kết hợp FP16 (float 16-bit) và FP32 (float 32-bit)
- `autocast`: Tự động chuyển operations sang FP16 khi an toàn
- `GradScaler`: Scale loss để tránh gradient underflow trong FP16
- **Lợi ích:** Nhanh gấp 2×, giảm 50% bộ nhớ

### Thư Viện Chuyên Dụng

```python
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
```

**Chức năng:**
- `timm`: PyTorch Image Models (kho pretrained models)
- `PIL`: Load và xử lý ảnh
- `tqdm`: Hiển thị thanh tiến trình
- `roc_auc_score`: Tính **AUC metric**

**Thuật Ngữ Kỹ Thuật - AUC (Area Under ROC Curve):**
- Đo khả năng phân biệt classes của classifier
- Khoảng: 0.0 (tệ nhất) đến 1.0 (hoàn hảo)
- Bền vững với class imbalance (khác accuracy)

---

## 2. Cấu Hình Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

**Cấu hình:**
- `level=INFO`: Ghi log từ mức INFO trở lên
- `format`: Cấu trúc Timestamp - Level - Message
- `FileHandler`: Ghi vào file `training.log`
- `StreamHandler`: Xuất ra console (stdout)

**Ví dụ output:**
```
2025-12-01 10:00:00 - INFO - Epoch 1 started
2025-12-01 10:31:24 - INFO - Epoch 1: Acc=0.9245
```

---

## 3. Class Dataset

### Khởi Tạo

```python
class FakeClueDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []       # List các tuple (path, label_idx, category_idx)
        self.targets = []       # Chỉ labels (tương thích PyTorch)
        self.categories = []    # Chỉ category indices
```

**Mẫu Thiết Kế:**
- Kế thừa từ `torch.utils.data.Dataset`
- Triển khai lazy loading (ảnh được load khi cần, không phải lúc khởi tạo)
- Duy trì 3 lists song song cho các pattern truy cập khác nhau

### Phát Hiện Categories

```python
        self.classes = ['real', 'fake']
        self.class_to_idx = {'real': 0, 'fake': 1}
        
        self.category_names = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.cat_to_idx = {name: i for i, name in enumerate(self.category_names)}
```

**Quy trình:**
1. Định nghĩa binary classes (real/fake)
2. Quét thư mục gốc tìm category folders
3. **`sorted()`**: Đảm bảo thứ tự deterministic giữa các lần chạy
4. Tạo ánh xạ hai chiều (name ↔ index)

**Thuật Ngữ Kỹ Thuật - Deterministic (Xác Định):**
- Cho kết quả giống nhau với cùng input
- Quan trọng cho tính reproducibility (tái tạo được) trong thí nghiệm ML

### Quét Dataset

```python
    def _scan_dataset(self):
        count_by_cat_label = Counter()
        
        for cat in self.category_names:
            cat_dir = self.root_dir / cat
            cat_idx = self.cat_to_idx[cat]
            
            for label in self.classes:
                label_dir = cat_dir / label
                if not label_dir.exists():
                    continue
                    
                images = (
                    list(label_dir.rglob("*.jpg")) +
                    list(label_dir.rglob("*.png")) +
                    list(label_dir.rglob("*.jpeg")) +
                    list(label_dir.rglob("*.JPG"))
                )
                
                label_idx = self.class_to_idx[label]
                
                for img_path in images:
                    self.samples.append((str(img_path), label_idx, cat_idx))
                    self.targets.append(label_idx)
                    self.categories.append(cat_idx)
```

**Thuật Ngữ Kỹ Thuật - rglob (Recursive Glob):**
- Tìm kiếm đệ quy trong cây thư mục
- Khớp files theo pattern (ví dụ: `*.jpg`)
- Xử lý cấu trúc folder lồng nhau

**Ví dụ:**
```
chameleon/fake/
  ├── subfolder1/img1.jpg  ← Tìm thấy
  ├── subfolder2/img2.jpg  ← Tìm thấy
  └── img3.jpg             ← Tìm thấy
```

### Lấy Item

```python
    def __getitem__(self, idx):
        path, label, cat = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label, cat, path
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
```

**Điểm quan trọng:**
- `.convert('RGB')`: Chuẩn hóa về 3 channels (một số ảnh là grayscale/RGBA)
- Xử lý lỗi: Trả về ảnh tiếp theo nếu ảnh hiện tại lỗi (tránh crash training)
- `% len(self)`: Modulo đảm bảo indexing vòng tròn

---

## 4. Weighted Sampler

```python
def get_balanced_sampler(dataset):
    counts = Counter()
    for i in range(len(dataset)):
        _, label, cat = dataset.samples[i]
        counts[(cat, label)] += 1
```

**Bước 1: Đếm Tần Suất Classes**
```python
# Ví dụ output:
{
    (0, 0): 41394,  # chameleon/real
    (0, 1): 20310,  # chameleon/fake
    (1, 0): 5216,   # doc/real
    (1, 1): 18868   # doc/fake
}
```

```python
    weights_per_group = {k: 1.0 / v if v > 0 else 0 for k, v in counts.items()}
```

**Bước 2: Tính Trọng Số Nghịch Đảo Tần Suất**

**Công thức:** `weight = 1 / count`

**Lý giải Toán Học:**
- Classes thiểu số nhận trọng số cao hơn
- Khi sampling, xác suất ∝ trọng số
- Số samples kỳ vọng mỗi class ≈ hằng số

**Ví dụ:**
```python
chameleon/real: 41394 samples → weight = 0.000024
chameleon/fake: 20310 samples → weight = 0.000049  (cao gấp 2×)
```

```python
    sample_weights = []
    for i in range(len(dataset)):
        _, label, cat = dataset.samples[i]
        weight = weights_per_group.get((cat, label), 0)
        sample_weights.append(weight)
    
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler
```

**Bước 3: Gán Trọng Số Cho Samples**
- Mỗi sample thừa kế trọng số của class
- `WeightedRandomSampler` sample có replacement
- Qua epoch, phân phối class trở nên cân bằng

**Thuật Ngữ Kỹ Thuật - Sampling with Replacement:**
- Cùng sample có thể được chọn nhiều lần mỗi epoch
- Đảm bảo classes thiểu số được đại diện đầy đủ

---

## 5. Hàm Training

### Chữ Ký Hàm

```python
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, accum_iter=8):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
```

**Tham số:**
- `model`: Mạng neural
- `loader`: DataLoader (cung cấp batches)
- `criterion`: Hàm loss
- `optimizer`: Thuật toán cập nhật trọng số
- `scaler`: GradScaler cho mixed precision
- `device`: 'cuda' hoặc 'cpu'
- `epoch`: Số epoch hiện tại
- `accum_iter`: Số bước **Gradient Accumulation**

**Thuật Ngữ Kỹ Thuật - Gradient Accumulation (Tích Lũy Gradient):**
- Mô phỏng batch sizes lớn hơn
- Công thức: `Batch Hiệu Quả = Batch Size × Bước Tích Lũy`
- Ví dụ: Batch=4, Accum=8 → Hiệu Quả=32
- Use Case: VRAM hạn chế

### Vòng Lặp Training

```python
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels, cats, paths) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device).float()
```

**Truyền Dữ Liệu:**
- `.to(device)`: Chuyển tensors sang GPU
- `.float()`: Đảm bảo dtype FP32 (yêu cầu của BCEWithLogitsLoss)

### Forward Pass

```python
        with autocast():
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            loss = loss / accum_iter
```

**Thuật Ngữ Kỹ Thuật - autocast:**
- Context manager cho automatic mixed precision
- Operations tự động dùng FP16 khi có lợi
- Giữ FP32 cho ops nhạy cảm về số

**Xử Lý Output Shape:**
- `model(images)`: Trả về `[batch_size, 1]`
- `.squeeze(-1)`: Xóa dimension cuối → `[batch_size]`
- **Tại sao `-1`?** Bảo toàn batch dimension khi `batch_size=1`

**Chuẩn Hóa Loss:**
- `loss / accum_iter`: Chia cho số bước tích lũy
- Đảm bảo magnitude gradient nhất quán
- Tương đương toán học: `mean(losses) = sum(losses/N) / N`

### Backward Pass

```python
        scaler.scale(loss).backward()
        
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

**Quy Trình Scaling Gradient:**
1. `scaler.scale(loss)`: Nhân loss với scale factor (vd: 2¹⁶)
2. `.backward()`: Tính scaled gradients
3. `scaler.step()`: Unscale gradients rồi cập nhật weights
4. `scaler.update()`: Điều chỉnh scale factor cho lần tiếp

**Thuật Ngữ Kỹ Thuật - Gradient Underflow:**
- Khoảng FP16: ~10⁻⁴ đến 10⁴
- Gradients nhỏ (vd: 10⁻⁷) trở thành zero trong FP16
- Scaling ngăn chặn mất độ chính xác này

**Điều Kiện Optimizer Step:**
- Mỗi `accum_iter` batches HOẶC batch cuối
- Ví dụ: accum_iter=8 → step tại batch 8, 16, 24, ..., và batch cuối

### Tính Metrics

```python
        running_loss += (loss.item() * accum_iter) * images.size(0)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss / total, correct / total
```

**Pipeline Prediction:**
1. `torch.sigmoid(outputs)`: Chuyển logits thành xác suất [0, 1]
2. `> 0.5`: Ngưỡng nhị phân
3. `.float()`: Chuyển boolean thành 0.0/1.0

**Tích Lũy Metric:**
- Running loss hoàn nguyên chuẩn hóa và scale theo batch size
- Accuracy: tỷ lệ đơn giản của predictions đúng

---

## 6. Hàm Validation

### Decorator & Setup

```python
@torch.no_grad()
def validate(model, loader, criterion, device, epoch, use_tta=False):
    model.eval()
```

**Thuật Ngữ Kỹ Thuật - @torch.no_grad():**
- Decorator vô hiệu hóa tính toán gradient
- **Memory:** Giảm ~50% (không lưu activations)
- **Tốc độ:** Nhanh hơn ~30% (không xây backward graph)

**Thuật Ngữ Kỹ Thuật - model.eval():**
- Đặt model về chế độ evaluation
- **Hiệu ứng:**
  - Dropout: Tắt (dùng tất cả neurons)
  - BatchNorm: Dùng running statistics (không phải batch statistics)

### Test-Time Augmentation

```python
        with autocast():
            if use_tta:
                outputs1 = model(images).squeeze(-1)
                outputs2 = model(torch.flip(images, dims=[3])).squeeze(-1)
                outputs = (outputs1 + outputs2) / 2
            else:
                outputs = model(images).squeeze(-1)
```

**Thuật Ngữ Kỹ Thuật - TTA (Test-Time Augmentation):**
- Áp dụng augmentations trong inference
- Lấy trung bình predictions từ nhiều phiên bản
- **Triển khai:** Original + Lật Ngang

**Dimensions Tensor:**
```
Image tensor: [batch, channels, height, width]
              [  4,      3,      224,    224  ]
               dim0   dim1    dim2    dim3

torch.flip(dims=[3]): Lật dimension width (lật ngang)
```

**Tác Động Hiệu Suất:**
- Accuracy: +0.3-0.5%
- Tốc độ: Chậm 2× (hai forward passes)

### Tính AUC

```python
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
```

**Pipeline Chuyển Đổi Dữ Liệu:**
1. `.cpu()`: GPU tensor → CPU tensor
2. `.numpy()`: PyTorch tensor → NumPy array
3. `.extend()`: Thêm elements (không phải nested list)

**Xử Lý Edge Case:**
- `len(set(all_labels)) > 1`: Đảm bảo có ít nhất 2 classes
- Tránh lỗi `roc_auc_score` trên batch đơn class

---

## 7. Hàm Main

### Phân Tích Arguments

```python
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='...')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--use_tta', action='store_true')
args = parser.parse_args()
```

**Loại Arguments:**
- `type=str/int/float`: Tự động chuyển đổi kiểu
- `action='store_true'`: Boolean flag (có mặt = True)

### Thống Kê Normalization

```python
if 'clip' in args.model_name:
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
else:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
```

**Chi Tiết Triển Khai Quan Trọng:**
- Normalization phải khớp với dữ liệu training của pretrained model
- CLIP models: Dùng CLIP statistics
- ImageNet models: Dùng ImageNet statistics
- **Tác động khi sai:** Giảm accuracy nghiêm trọng (có thể mất 40%+)

**Công thức Normalization:**
```
pixel_normalized = (pixel - mean) / std
```

### Transforms Dữ Liệu

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
    transforms.RandomRotation(degrees=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

**Lý Do Thiết Kế:**
- **Augmentation Nhẹ:** Bảo toàn artifacts của ảnh fake
- ColorJitter=0.05: Chỉ 5% variation (vs thông thường 0.2-0.4)
- Rotation=3°: Biến dạng hình học tối thiểu
- **Trade-off:** Generalization vs bảo toàn artifacts

### Khởi Tạo Model

```python
model = timm.create_model(
    'vit_base_patch16_clip_224.openai',
    pretrained=True,
    num_classes=1
)
model.set_grad_checkpointing(enable=True)
```

**Thuật Ngữ Kỹ Thuật - Gradient Checkpointing:**
- Đánh đổi tính toán lấy bộ nhớ
- Tính lại activations trong backward pass (thay vì lưu)
- **Tác động:** -40% VRAM, +15% thời gian training

### Cấu Hình Optimizer

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.05,
    betas=(0.9, 0.98)
)
```

**Thuật Ngữ Kỹ Thuật - AdamW:**
- Adam với decoupled weight decay
- **Weight Decay:** Hệ số regularization L₂
- Công thức: `loss = task_loss + weight_decay × ||weights||²`
- **Betas:** Tỷ lệ suy giảm exponential cho moment estimates

### Lập Lịch Learning Rate

```python
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=1
)
main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs - 1
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[1]
)
```

**Lịch Hai Giai Đoạn:**

**Giai Đoạn 1 - Warmup (Epoch 1):**
- Tăng tuyến tính: 0.01× → 1× của base LR
- **Mục đích:** Tránh cập nhật gradient lớn đầu training

**Giai Đoạn 2 - Cosine Annealing (Epochs 2+):**
- Công thức: `lr = 0.5 × lr_max × (1 + cos(π × t / T))`
- Giảm mượt về gần zero
- **Lợi thế:** Hội tụ dần dần

### Vòng Lặp Training với Early Stopping

```python
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc, val_auc = validate(...)
    scheduler.step()
    
    if val_auc > best_auc:
        torch.save({...}, save_path)
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= args.patience:
        break
```

**Thuật Ngữ Kỹ Thuật - Early Stopping:** - Theo dõi validation metric (AUC)
- Dừng nếu không cải thiện trong N (patience) epochs
- **Mục đích:** Tránh overfitting, tiết kiệm tính toán

**Nội Dung Checkpoint:**
- `model_state_dict`: Trọng số model
- `optimizer_state_dict`: Trạng thái optimizer (để resume)
- `scheduler_state_dict`: Trạng thái LR schedule
- Metrics: acc, auc
- Metadata: epoch, category_names

---

## Tóm Tắt

### Khái Niệm Kỹ Thuật Chính

1. **Mixed Precision (FP16):** Nhanh 2×, giảm 50% bộ nhớ
2. **Gradient Accumulation:** Mô phỏng batches lớn hơn
3. **Gradient Checkpointing:** Đổi tính toán lấy bộ nhớ
4. **Weighted Sampling:** Cân bằng imbalanced classes
5. **AUC Metric:** Bền vững với class imbalance
6. **TTA:** Ensemble của augmented predictions
7. **Early Stopping:** Tự động ngăn overfitting

### Tối Ưu Hiệu Suất

| Kỹ Thuật | Lợi Ích |
|----------|---------|
| Mixed Precision | Nhanh 2×, VRAM 50% |
| Gradient Accumulation | Batch hiệu quả 32 trên GPU 4GB |
| Gradient Checkpointing | -40% VRAM |
| Multi-worker Loading | Tốc độ load data 2-3× |
| Weighted Sampling | +1.5% accuracy |

### Kiến Trúc Cuối Cùng

```
Input: Ảnh RGB 224×224×3
  ↓
ViT-Base-CLIP (86M params, pretrained)
  ↓
Classification Head Nhị Phân (1 output)
  ↓
BCEWithLogitsLoss
  ↓
Output: Prediction Real/Fake
```

---

**Cập nhật:** 2025-12-01  
**Phiên bản Code:** Production v1.0
