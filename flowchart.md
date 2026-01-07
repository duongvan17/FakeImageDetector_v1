# Flowchart - Hệ Thống Phát Hiện Ảnh Giả

## 1. Tổng Quan Hệ Thống

```mermaid
flowchart TD
    START(["Bắt đầu"])
    
    A[/"Nhận ảnh đầu vào"/]
    B["Kiểm tra định dạng"]
    C["Chuyển đổi sang RGB"]
    D["Resize 224×224 pixels"]
    E["Chuẩn hóa CLIP"]
    F[["Model ViT-Base-CLIP"]]
    G["Hàm Sigmoid"]
    H{"Xác suất > 0.5?"}
    I[/"Kết quả: GIẢ"/]
    J[/"Kết quả: THẬT"/]
    K["Tính độ tin cậy"]
    L[/"Trả kết quả"/]
    
    START --> A
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H -->|"Có"| I
    H -->|"Không"| J
    I --> K
    J --> K
    K --> L
    L --> END(["Kết thúc"])
```

---

## 2. Module Tiền Xử Lý Ảnh

```mermaid
flowchart TD
    START(["Bắt đầu"])
    
    A[/"Ảnh PIL: H×W×3"/]
    B["Resize về 224×224"]
    B1["Phương pháp: BICUBIC"]
    C["Chuyển sang Tensor"]
    C1["Shape: 3, 224, 224"]
    D["Chuẩn hóa thống kê CLIP"]
    D1["Mean: 0.4815, 0.4578, 0.4082"]
    D2["Std: 0.2686, 0.2613, 0.2758"]
    E["Thêm chiều batch"]
    E1["Shape: 1, 3, 224, 224"]
    F[/"Tensor đã xử lý"/]
    
    START --> A
    A --> B --> B1 --> C --> C1 --> D --> D1 --> D2 --> E --> E1 --> F
    F --> END(["Kết thúc"])
```

---

## 3. Kiến Trúc Vision Transformer

```mermaid
flowchart TD
    START(["Đầu vào Model"])
    
    A[/"Tensor: B, 3, 224, 224"/]
    B["Conv2D Projection"]
    B1["Kernel: 16×16, Stride: 16"]
    C["Chia thành 196 patches"]
    D["Thêm CLS Token"]
    E["Thêm Position Embedding"]
    E1[/"Sequence: B, 197, 768"/]
    
    F[["Transformer Block 1"]]
    G[["Transformer Block 2"]]
    H["... ×12 blocks ..."]
    I[["Transformer Block 12"]]
    J[/"Output: B, 197, 768"/]
    
    K["Lấy CLS Token"]
    L["Layer Normalization"]
    M["Linear: 768 → 1"]
    N[/"Logit đầu ra"/]
    
    START --> A --> B --> B1 --> C --> D --> E --> E1
    E1 --> F --> G --> H --> I --> J
    J --> K --> L --> M --> N
    N --> END(["Đầu ra Model"])
```

---

## 4. Chi Tiết Transformer Block

```mermaid
flowchart TD
    START(["Đầu vào Block"])
    
    A[/"Input: B, 197, 768"/]
    B["Layer Norm 1"]
    C["Tính Q, K, V"]
    D["Chia thành 12 Heads"]
    E["Attention: Q×K^T / √64"]
    F["Softmax"]
    G["Nhân với V"]
    H["Ghép các Heads"]
    I["Linear Projection"]
    J["Residual + Input"]
    
    K["Layer Norm 2"]
    L["Linear: 768 → 3072"]
    M["GELU Activation"]
    N["Linear: 3072 → 768"]
    O["Residual + Previous"]
    P[/"Output: B, 197, 768"/]

    START --> A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    A --> J
    J --> K --> L --> M --> N --> O
    J --> O
    O --> P --> END(["Đầu ra Block"])
```

---

## 5. Quy Trình Suy Luận

```mermaid
flowchart TD
    START(["Bắt đầu Inference"])
    
    A[/"Tensor: 1, 3, 224, 224"/]
    B["Tắt tính Gradient"]
    C["Model eval mode"]
    D[["Forward ViT-Base-CLIP"]]
    E[/"Logit thô"/]
    F["Sigmoid: 1/(1+e^-x)"]
    G[/"Xác suất: 0.0 - 1.0"/]
    H{"P_fake > 0.5?"}
    I["Label: GIẢ"]
    J["Label: THẬT"]
    K["Confidence = P × 100%"]
    L["Confidence = (1-P) × 100%"]
    M[/"Trả về: Label + Confidence"/]
    
    START --> A --> B --> C --> D --> E --> F --> G --> H
    H -->|"Có"| I --> K
    H -->|"Không"| J --> L
    K --> M
    L --> M
    M --> END(["Kết thúc Inference"])
```

---

## 6. Quy Trình Training

```mermaid
flowchart TD
    START(["Bắt đầu Training"])
    
    A[("Dataset FakeVLM")]
    B["Tính trọng số mẫu"]
    C["WeightedRandomSampler"]
    D["DataLoader: Batch=4"]
    
    E["Augmentation"]
    E1["HorizontalFlip, ColorJitter, Rotation"]
    
    F["Forward Pass FP16"]
    G["Tính BCEWithLogitsLoss"]
    H["Loss / 8"]
    
    I["Backward Pass"]
    J["Tích lũy Gradient"]
    K{"Đủ 8 steps?"}
    L["Tiếp tục batch"]
    M["Optimizer Step"]
    N["Zero Gradient"]
    
    O["Đánh giá Test Set"]
    P["Tính Accuracy, AUC"]
    Q{"AUC > Best?"}
    R[("Lưu Checkpoint")]
    S["Patience++"]
    T["Reset Patience"]
    U{"Patience >= 5?"}
    V(["Early Stop"])
    W["Tiếp tục Epoch"]
    
    START --> A --> B --> C --> D --> E --> E1 --> F --> G --> H --> I --> J --> K
    K -->|"Không"| L --> F
    K -->|"Có"| M --> N --> O --> P --> Q
    Q -->|"Có"| R --> T --> W
    Q -->|"Không"| S --> U
    U -->|"Có"| V
    U -->|"Không"| W --> D
```

---

## 7. Xử Lý Lỗi API

```mermaid
flowchart TD
    START(["Nhận Request"])
    
    A[/"Request đến"/]
    B{"Ảnh hợp lệ?"}
    C[/"HTTP 400: Định dạng sai"/]
    D{"Model đã load?"}
    E[/"HTTP 500: Lỗi Model"/]
    F{"Inference thành công?"}
    G[/"HTTP 500: Lỗi xử lý"/]
    H[/"HTTP 200: Kết quả"/]
    
    START --> A --> B
    B -->|"Không"| C --> END1(["Lỗi Client"])
    B -->|"Có"| D
    D -->|"Không"| E --> END2(["Lỗi Server"])
    D -->|"Có"| F
    F -->|"Không"| G --> END2
    F -->|"Có"| H --> END3(["Thành công"])
```

---

## 8. Kiến Trúc Triển Khai

```mermaid
flowchart TD
    subgraph CLIENT["Tầng Client"]
        A1[/"Web Browser"/]
        A2[/"CLI Terminal"/]
        A3[/"Batch Script"/]
    end

    subgraph SERVER["Tầng Ứng Dụng"]
        B1["Flask Server"]
        B2[["Inference Engine"]]
        B3["Image Processor"]
    end

    subgraph STORAGE["Tầng Lưu Trữ"]
        C1[("Model Checkpoint")]
        C2[("Cấu hình")]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B2
    B1 --> B3 --> B2
    C1 --> B2
    C2 --> B3
```
