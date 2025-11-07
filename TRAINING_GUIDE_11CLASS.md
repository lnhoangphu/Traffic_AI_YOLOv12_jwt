# YOLOv12 11-Class Traffic Detection - Pipeline Hoàn Chỉnh

## 📋 Mục tiêu

Xây dựng mô hình YOLOv12 nhận diện chính xác **11 class đối tượng giao thông** cho điều kiện Việt Nam:

| ID | Class | Mô tả |
|----|-------|-------|
| 0 | Vehicle | Xe cộ nói chung (car, sedan, van, etc.) |
| 1 | Bus | Xe buýt, xe khách |
| 2 | Bicycle | Xe đạp |
| 3 | Person | Người (pedestrian, passenger) |
| 4 | Engine | Xe máy, xe 2 bánh có động cơ |
| 5 | Truck | Xe tải |
| 6 | Tricycle | Xe 3 bánh |
| 7 | Obstacle | Vật cản, chướng ngại vật |
| 8 | Pothole | Ổ gà, hư hỏng mặt đường |
| 9 | Traffic Light | Đèn giao thông |
| 10 | Traffic Sign | Biển báo giao thông |

## ✅ Đảm bảo chất lượng

- ✅ **Chỉ 11 class** (0-10), không có class dư thừa
- ✅ **Annotation chuẩn YOLO** format
- ✅ **Class mapping chính xác** từ các dataset nguồn
- ✅ **Class weights tự động** để cân bằng loss
- ✅ **Augmentation mạnh** cho class thiếu số
- ✅ **Confusion matrix** để phát hiện class nhầm lẫn

## 📁 Cấu trúc Dataset

```
datasets/traffic_ai_balanced_11class_processed/
├── data.yaml                    # Config file cho YOLOv12
├── images/
│   ├── train/                  # 3,364 images (70%)
│   ├── val/                    # 961 images (20%)
│   └── test/                   # 482 images (10%)
└── labels/
    ├── train/                  # YOLO format annotations
    ├── val/
    └── test/
```

## 📊 Phân bố Class (Hiện tại)

```
Class Distribution (toàn dataset - 178,437 objects):
   0: Vehicle         -  89,640 (50.24%) ⚠️  Nhiều
   1: Bus             -   4,767 ( 2.67%) ✅
   2: Bicycle         -   1,827 ( 1.02%) ⚠️  Ít
   3: Person          -  29,418 (16.49%) ✅
   4: Engine          -   1,061 ( 0.59%) ⚠️  Rất ít
   5: Truck           -     833 ( 0.47%) ⚠️  Rất ít
   6: Tricycle        -  48,152 (26.99%) ⚠️  Nhiều
   7: Obstacle        -     575 ( 0.32%) ⚠️  Rất ít
   8: Pothole         -     468 ( 0.26%) ⚠️  Rất ít
   9: Traffic Light   -     749 ( 0.42%) ⚠️  Rất ít
   10: Traffic Sign    -     947 ( 0.53%) ⚠️  Rất ít
```

**Vấn đề:** Dataset **mất cân bằng nghiêm trọng**
- Vehicle (50%) + Tricycle (27%) + Person (16%) = **93%**
- 8 class còn lại chỉ **7%**

**Giải pháp:** Sử dụng **class weights + augmentation mạnh** trong training thay vì oversample.

## 🚀 Quy trình sử dụng

### 1️⃣ Phân tích Dataset

Kiểm tra dataset hiện tại:

```powershell
python scripts\analyze_11class_dataset.py
```

Kết quả:
- ✅ Tất cả class hợp lệ (0-10)
- ✅ Không có class dư thừa
- ✅ Không có images thiếu label
- ⚠️  Mất cân bằng class (sẽ xử lý bằng class weights)

### 2️⃣ Training Model

**Script:** `training/train_11class_final.py`

**Tính năng:**
- ✅ Tự động tính class weights (inverse frequency)
- ✅ Augmentation mạnh cho class thiếu số
- ✅ Early stopping (patience=50)
- ✅ Save checkpoint mỗi 10 epochs
- ✅ Automatic Mixed Precision (AMP)
- ✅ Confusion matrix + plots

**Chạy training:**

```powershell
python training\train_11class_final.py
```

**Cấu hình mặc định:**
```python
{
    'model_size': 'n',           # nano (nhẹ nhất, nhanh nhất)
    'epochs': 300,               # 300 epochs
    'batch_size': 16,            # Điều chỉnh theo GPU RAM
    'img_size': 640,             # Standard YOLO
    'patience': 50,              # Early stopping
    'device': '0',               # GPU 0
    'use_class_weights': True,   # Class weights
    'pretrained': True           # Pretrained weights
}
```

**Class weights tự động:**
```
Class  0: 82185 objects -> weight = 0.589
Class  1:  3425 objects -> weight = 1.456
Class  2:  1276 objects -> weight = 2.000 (max)
Class  3:   117 objects -> weight = 2.000
Class  4:   742 objects -> weight = 1.870
...
```

**Augmentation:**
- Mosaic: 1.0
- Mixup: 0.1
- Copy-paste: 0.1
- HSV, rotation, scale, shear, flip
- MẠNHhơn cho class thiếu số

### 3️⃣ Evaluation

**Script:** `training/evaluate_11class.py`

```powershell
python training\evaluate_11class.py
```

**Metrics:**
- ✅ Overall mAP@50, mAP@50-95
- ✅ Per-class Precision, Recall, mAP
- ✅ Confusion Matrix (heatmap)
- ✅ Phát hiện class có performance thấp

**Test trên ảnh thật:**
```python
test_on_images(
    model_path="runs/.../best.pt",
    image_folder="test_images/",
    output_folder="runs/test_predictions/",
    conf_threshold=0.25
)
```

### 4️⃣ Inference (Sử dụng Model)

```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/train_11class_final/yolov12_11class_weighted/weights/best.pt')

# Predict trên ảnh
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,
    save=True
)

# Predict trên video
results = model.predict(
    source='path/to/video.mp4',
    conf=0.25,
    save=True
)

# Real-time webcam
results = model.predict(
    source=0,  # webcam
    conf=0.25,
    stream=True
)
```

## 📊 Kết quả mong đợi

### Mục tiêu Performance

| Metric | Target |
|--------|--------|
| mAP@50 | ≥ 0.60 |
| mAP@50-95 | ≥ 0.40 |
| Per-class mAP | ≥ 0.30 cho tất cả class |

### Confusion Matrix

- ✅ Không nhầm lẫn giữa các class chính (Vehicle ↔ Person)
- ⚠️  Có thể nhầm: Bicycle ↔ Engine (cần xem xét)
- ⚠️  Có thể nhầm: Bus ↔ Truck (cần xem xét)

### Common Issues

**1. Class có mAP thấp (< 0.3):**
- ❌ Nguyên nhân: Quá ít training samples
- ✅ Giải pháp: Tăng class weight, tăng augmentation
- ✅ Giải pháp: Thu thập thêm dữ liệu

**2. Class bị nhầm lẫn nhiều:**
- ❌ Nguyên nhân: Label không chính xác hoặc class quá giống nhau
- ✅ Giải pháp: Review lại labels
- ✅ Giải pháp: Tăng augmentation để model học phân biệt

**3. Model overfitting:**
- ❌ Nguyên nhân: Training quá lâu, dataset quá nhỏ
- ✅ Giải pháp: Early stopping (đã có patience=50)
- ✅ Giải pháp: Tăng augmentation

## 🛠️ Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'ultralytics'"

```powershell
pip install ultralytics
```

### Lỗi: GPU Out of Memory

Giảm batch size trong `train_11class_final.py`:
```python
'batch_size': 8,  # Giảm từ 16 -> 8
```

Hoặc cache=False:
```python
'cache': False,
```

### Model không học (loss không giảm)

- Kiểm tra learning rate (giảm lr0)
- Kiểm tra class weights (có thể disable tạm)
- Kiểm tra labels (có thể bị sai)

### Một số class bị bỏ qua (không detect)

- Kiểm tra class có đủ samples không
- Tăng class weight cho class đó
- Giảm confidence threshold khi inference

## 📝 Files quan trọng

| File | Mô tả |
|------|-------|
| `datasets/traffic_ai_balanced_11class_processed/data.yaml` | Dataset config |
| `config/taxonomy_complete_11class.yaml` | Class mapping từ datasets nguồn |
| `scripts/analyze_11class_dataset.py` | Phân tích dataset |
| `training/train_11class_final.py` | Training script (CLASS WEIGHTS) |
| `training/evaluate_11class.py` | Evaluation script |
| `dataset_11class_analysis.txt` | Kết quả phân tích dataset |

## 🎯 Next Steps

1. **Chạy training:**
   ```powershell
   python training\train_11class_final.py
   ```

2. **Monitor training:**
   - Xem tensorboard hoặc plots trong `runs/train_11class_final/`
   - Theo dõi loss, mAP tăng dần

3. **Evaluate:**
   ```powershell
   python training\evaluate_11class.py
   ```

4. **Review confusion matrix:**
   - Kiểm tra class nào bị nhầm lẫn nhiều
   - Quyết định có cần thu thập thêm data không

5. **Test trên ảnh thật:**
   - Kiểm tra xe hơi, xe buýt, người, xe đạp, v.v.
   - Nếu sai → back to step 1 (review labels, augment more)

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs trong `runs/train_11class_final/`
2. Xem confusion matrix
3. Phân tích per-class metrics
4. Review dataset lại với `analyze_11class_dataset.py`

---

**Version:** 1.0  
**Updated:** 2025-11-07  
**Author:** Traffic AI YOLOv12 Project
