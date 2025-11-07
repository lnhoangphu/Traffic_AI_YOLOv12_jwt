# 📚 YOLOv12 11-Class Traffic Detection - Documentation Index

## 🎯 Bắt đầu tại đây

**Bạn muốn làm gì?**

### 👀 Tìm hiểu tổng quan
→ Đọc [README_11CLASS.md](README_11CLASS.md)
- Dataset statistics
- 11 class là gì
- Quick start

### 🚀 Chạy training ngay
→ Chạy [run_11class_pipeline.py](run_11class_pipeline.py)
```powershell
python run_11class_pipeline.py
```

### 📖 Hướng dẫn chi tiết từng bước
→ Đọc [TRAINING_GUIDE_11CLASS.md](TRAINING_GUIDE_11CLASS.md)
- Dataset structure
- Training guide chi tiết
- Evaluation guide
- Troubleshooting

### ✅ Kiểm tra checklist
→ Đọc [CHECKLIST_11CLASS.md](CHECKLIST_11CLASS.md)
- Pre-training checklist
- Training monitoring
- Post-training verification
- Acceptance criteria

### 📊 Xem tổng kết toàn bộ
→ Đọc [SUMMARY_11CLASS.md](SUMMARY_11CLASS.md)
- Đã làm gì
- Kết quả đạt được
- Next steps
- Known issues

---

## 📁 Files & Scripts

### 📊 Analysis
| File | Mô tả | Usage |
|------|-------|-------|
| `scripts/analyze_11class_dataset.py` | Phân tích dataset | `python scripts\analyze_11class_dataset.py` |
| `dataset_11class_analysis.txt` | Kết quả phân tích | (auto-generated) |

### 🏋️ Training
| File | Mô tả | Usage |
|------|-------|-------|
| `training/train_11class_final.py` | Training script chính | `python training\train_11class_final.py` |
| `datasets/.../data.yaml` | Dataset config | (used by training) |
| `config/taxonomy_complete_11class.yaml` | Class mapping | (reference) |

### 📈 Evaluation
| File | Mô tả | Usage |
|------|-------|-------|
| `training/evaluate_11class.py` | Evaluation script | `python training\evaluate_11class.py` |
| `runs/.../confusion_matrix.png` | Confusion matrix | (auto-generated) |
| `runs/.../results.csv` | Training metrics | (auto-generated) |

### 🎯 Pipeline
| File | Mô tả | Usage |
|------|-------|-------|
| `run_11class_pipeline.py` | Master pipeline (all-in-one) | `python run_11class_pipeline.py` |

### 📚 Documentation
| File | Mô tả |
|------|-------|
| `README_11CLASS.md` | Overview tổng quan |
| `TRAINING_GUIDE_11CLASS.md` | Hướng dẫn chi tiết |
| `CHECKLIST_11CLASS.md` | Checklist kiểm tra |
| `SUMMARY_11CLASS.md` | Tổng kết toàn bộ |
| `INDEX_11CLASS.md` | File này (index) |

---

## 🗂️ Dataset Structure

```
datasets/traffic_ai_balanced_11class_processed/
├── data.yaml                     # ⭐ Config chính
├── images/
│   ├── train/  (3,364 images)
│   ├── val/    (961 images)
│   └── test/   (482 images)
└── labels/
    ├── train/  (YOLO format .txt)
    ├── val/
    └── test/
```

---

## 🎓 11 Classes

| ID | Class Name | Mô tả | Priority |
|----|-----------|-------|----------|
| 0 | Vehicle | Xe hơi, sedan, van | HIGH ⭐ |
| 1 | Bus | Xe buýt | MEDIUM |
| 2 | Bicycle | Xe đạp | MEDIUM |
| 3 | Person | Người | HIGH ⭐ |
| 4 | Engine | Xe máy, scooter | MEDIUM |
| 5 | Truck | Xe tải | MEDIUM |
| 6 | Tricycle | Xe 3 bánh | LOW |
| 7 | Obstacle | Vật cản | LOW |
| 8 | Pothole | Ổ gà | LOW |
| 9 | Traffic Light | Đèn giao thông | HIGH ⭐ |
| 10 | Traffic Sign | Biển báo | HIGH ⭐ |

---

## 🚀 Quick Commands

### Phân tích dataset
```powershell
python scripts\analyze_11class_dataset.py
```

### Training
```powershell
python training\train_11class_final.py
```

### Evaluation
```powershell
python training\evaluate_11class.py
```

### All-in-one pipeline
```powershell
python run_11class_pipeline.py
```

### Inference (sau khi có model)
```python
from ultralytics import YOLO
model = YOLO('runs/train_11class_final/.../weights/best.pt')
results = model.predict('test_images/', save=True)
```

---

## 📊 Expected Results

| Metric | Target |
|--------|--------|
| mAP@50 | ≥ 0.60 |
| mAP@50-95 | ≥ 0.40 |
| Vehicle mAP | ≥ 0.70 |
| Person mAP | ≥ 0.60 |
| Traffic Light mAP | ≥ 0.50 |
| Traffic Sign mAP | ≥ 0.50 |
| Other classes | ≥ 0.30 |

---

## ⚠️ Important Notes

### ✅ Đảm bảo
- Dataset chỉ có 11 class (0-10)
- Không có class dư thừa
- Format YOLO chuẩn
- data.yaml đúng đường dẫn

### ⚠️ Chú ý
- Dataset mất cân bằng (93% là 3 class)
- Class weights tự động để cân bằng
- Training mất ~6-8 giờ (YOLOv8n)

### 🎯 Mục tiêu
- Model nhận diện đúng 11 class
- Không nhầm lẫn nghiêm trọng
- Sẵn sàng deploy

---

## 📞 Troubleshooting

### Model không học
→ Xem [TRAINING_GUIDE_11CLASS.md](TRAINING_GUIDE_11CLASS.md) - Troubleshooting section

### Class bị skip
→ Kiểm tra class weights, giảm confidence threshold

### Overfitting
→ Early stopping (patience=50), tăng augmentation

### Confusion matrix có vấn đề
→ Review labels, tăng training epochs cho class yếu

---

## 🎉 Status

- ✅ Dataset ready
- ✅ Scripts ready
- ✅ Documentation complete
- ⏳ Training pending (cần user chạy)
- ⏳ Evaluation pending
- ⏳ Deployment pending

---

**🚀 START HERE:**
```powershell
python run_11class_pipeline.py
```

**Version:** 1.0  
**Last Updated:** 2025-11-07
