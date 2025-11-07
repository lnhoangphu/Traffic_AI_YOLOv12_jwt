# 🚦 Traffic AI YOLOv12 - 11 Class Detection

## 🎯 Tổng quan

Hệ thống nhận diện **11 class đối tượng giao thông** cho điều kiện Việt Nam sử dụng YOLOv12, được xây dựng từ 4 datasets nguồn:

1. **Intersection Flow 5K** - Traffic surveillance
2. **Object Detection 35** - VisionGuard dataset  
3. **Road Issues** - Infrastructure problems
4. **VN Traffic Sign** - Vietnamese traffic signs

## ✅ 11 Class được Training

| ID | Class | Mô tả | Priority |
|----|-------|-------|----------|
| 0 | **Vehicle** | Xe cộ nói chung (car, sedan, van) | HIGH ⭐ |
| 1 | **Bus** | Xe buýt, xe khách | MEDIUM |
| 2 | **Bicycle** | Xe đạp | MEDIUM |
| 3 | **Person** | Người (pedestrian, passenger) | HIGH ⭐ |
| 4 | **Engine** | Xe máy, xe 2 bánh có động cơ | MEDIUM |
| 5 | **Truck** | Xe tải | MEDIUM |
| 6 | **Tricycle** | Xe 3 bánh | LOW |
| 7 | **Obstacle** | Vật cản, chướng ngại vật | LOW |
| 8 | **Pothole** | Ổ gà, hư hỏng mặt đường | LOW |
| 9 | **Traffic Light** | Đèn giao thông | HIGH ⭐ |
| 10 | **Traffic Sign** | Biển báo giao thông | HIGH ⭐ |

## 📊 Dataset Statistics

```
Total images: 4,807
├── Train: 3,364 (70%)
├── Val:     961 (20%)
└── Test:    482 (10%)

Total objects: 178,437
├── Vehicle:       89,640 (50.24%)
├── Tricycle:      48,152 (26.99%)
├── Person:        29,418 (16.49%)
├── Bus:            4,767 (2.67%)
├── Bicycle:        1,827 (1.02%)
├── Engine:         1,061 (0.59%)
├── Truck:            833 (0.47%)
├── Traffic Light:    749 (0.42%)
├── Traffic Sign:     947 (0.53%)
├── Obstacle:         575 (0.32%)
└── Pothole:          468 (0.26%)
```

**⚠️ Vấn đề:** Dataset mất cân bằng nghiêm trọng (93% chỉ là 3 class)  
**✅ Giải pháp:** Class weights + augmentation mạnh trong training

## 🚀 Quick Start

### 1. Cài đặt Dependencies

```powershell
pip install ultralytics opencv-python matplotlib seaborn pyyaml
```

### 2. Chạy Full Pipeline

```powershell
python run_11class_pipeline.py
```

Pipeline bao gồm:
1. ✅ Phân tích dataset
2. ✅ Training với class weights
3. ✅ Evaluation + confusion matrix

### 3. Hoặc chạy từng bước

**Bước 1: Phân tích dataset**
```powershell
python scripts\analyze_11class_dataset.py
```

**Bước 2: Training**
```powershell
python training\train_11class_final.py
```

**Bước 3: Evaluation**
```powershell
python training\evaluate_11class.py
```

## 📁 Cấu trúc Project

```
Traffic_AI_YOLOv12_jwt/
├── 📄 run_11class_pipeline.py           # Master script (chạy toàn bộ)
├── 📖 TRAINING_GUIDE_11CLASS.md         # Hướng dẫn chi tiết
├── 📖 README_11CLASS.md                 # File này
│
├── 📁 datasets/
│   └── traffic_ai_balanced_11class_processed/
│       ├── data.yaml                    # Dataset config ⭐
│       ├── images/train/val/test/
│       └── labels/train/val/test/
│
├── 📁 config/
│   └── taxonomy_complete_11class.yaml   # Class mapping
│
├── 📁 scripts/
│   ├── analyze_11class_dataset.py       # Phân tích dataset
│   └── create_balanced_11class_dataset.py
│
└── 📁 training/
    ├── train_11class_final.py           # Training script ⭐
    └── evaluate_11class.py              # Evaluation script ⭐
```

## 🎓 Training Details

### Hyperparameters

```yaml
Model: YOLOv8n (hoặc YOLOv12n nếu có)
Epochs: 300
Batch size: 16
Image size: 640x640
Optimizer: AdamW
Learning rate: 0.001 -> 0.00001
Patience: 50 (early stopping)
```

### Class Weights (Tự động tính)

Model tự động tính class weights theo công thức:
```
weight[i] = total_objects / (num_classes * class_count[i])
```

Normalized về [0.5, 2.0] để tránh extreme values.

### Augmentation (Mạnh)

```yaml
Mosaic: 1.0
Mixup: 0.1
Copy-paste: 0.1
HSV: (0.015, 0.7, 0.4)
Rotation: ±10°
Scale: ±50%
Shear: ±2°
Flip horizontal: 50%
```

## 📊 Kết quả mong đợi

| Metric | Target | Status |
|--------|--------|--------|
| mAP@50 | ≥ 0.60 | 🎯 |
| mAP@50-95 | ≥ 0.40 | 🎯 |
| Vehicle mAP | ≥ 0.70 | ⭐ |
| Person mAP | ≥ 0.60 | ⭐ |
| Traffic Light mAP | ≥ 0.50 | ⭐ |
| Traffic Sign mAP | ≥ 0.50 | ⭐ |
| Other classes mAP | ≥ 0.30 | ✅ |

## 🔍 Evaluation & Testing

### Confusion Matrix

Sau khi training, check confusion matrix để phát hiện:
- ❌ Class nào bị nhầm lẫn nhiều
- ❌ Class nào có performance thấp
- ✅ Điều chỉnh training strategy

### Test trên ảnh thật

```python
from ultralytics import YOLO

model = YOLO('runs/train_11class_final/.../weights/best.pt')

# Predict
results = model.predict(
    source='test_images/',
    conf=0.25,
    save=True
)
```

## ⚠️ Common Issues

### 1. Class thiếu số bị bỏ qua

**Nguyên nhân:** Quá ít training samples  
**Giải pháp:**
- ✅ Tăng class weight
- ✅ Tăng augmentation
- ✅ Giảm confidence threshold khi inference
- ✅ Thu thập thêm data

### 2. Class bị nhầm lẫn

**Ví dụ:** Bicycle ↔ Engine, Bus ↔ Truck

**Giải pháp:**
- ✅ Review lại labels
- ✅ Tăng augmentation để model học phân biệt
- ✅ Sử dụng larger model (s, m thay vì n)

### 3. Model overfitting

**Dấu hiệu:** Val loss tăng, train loss giảm

**Giải pháp:**
- ✅ Early stopping (đã có patience=50)
- ✅ Tăng augmentation
- ✅ Giảm model size

## 📖 Documentation

- **📘 [TRAINING_GUIDE_11CLASS.md](TRAINING_GUIDE_11CLASS.md)** - Hướng dẫn chi tiết
- **📗 [config/taxonomy_complete_11class.yaml](config/taxonomy_complete_11class.yaml)** - Class mapping
- **📕 [Dataset Analysis](dataset_11class_analysis.txt)** - Kết quả phân tích

## 🛠️ Development

### Thêm dataset mới

1. Convert về YOLO format
2. Map classes về 11 class chuẩn (xem `taxonomy_complete_11class.yaml`)
3. Merge vào dataset hiện tại
4. Re-run analysis

### Fine-tune model

```python
model = YOLO('runs/.../weights/best.pt')  # Load trained model

# Continue training
model.train(
    data='data.yaml',
    epochs=50,
    resume=True  # Continue from checkpoint
)
```

## 📞 Support

Nếu gặp vấn đề:
1. Xem logs trong `runs/train_11class_final/`
2. Check confusion matrix
3. Analyze per-class metrics
4. Review [TRAINING_GUIDE_11CLASS.md](TRAINING_GUIDE_11CLASS.md)

## 🎯 Roadmap

- [x] Dataset 11 class chuẩn
- [x] Training script với class weights
- [x] Evaluation script
- [x] Documentation đầy đủ
- [ ] Training hoàn thành (300 epochs)
- [ ] Model đạt mAP target
- [ ] Deploy API
- [ ] Real-time inference

---

**Version:** 1.0  
**Last Updated:** 2025-11-07  
**Author:** Traffic AI YOLOv12 Project  
**License:** MIT
