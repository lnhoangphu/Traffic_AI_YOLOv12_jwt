# ⚡ Quick Training Guide - YOLOv12n 11-Class

## 🎯 Mục đích

Script **quick_train_yolov12.py** giúp bạn:
- ✅ Test nhanh pipeline (30 epochs, ~30-45 phút)
- ✅ Verify dataset và config hoạt động
- ✅ Kiểm tra kết quả ban đầu trước khi commit full training

## 🚀 Chạy Quick Training

```powershell
python training\quick_train_yolov12.py
```

## ⚙️ Configuration

| Parameter | Quick Train | Full Train |
|-----------|-------------|------------|
| Model | YOLOv12n | YOLOv12n |
| Epochs | **30** | 300 |
| Patience | 10 | 50 |
| Batch | 16 | 16 |
| Time | **~30-45 min** | ~6-8 hours |
| Augmentation | Vừa phải | Mạnh |
| Save period | Every 5 epochs | Every 10 epochs |

## 📊 Expected Quick Results

Sau 30 epochs, bạn nên thấy:

| Metric | Expected |
|--------|----------|
| mAP@50 | ~0.35-0.45 |
| mAP@50-95 | ~0.20-0.28 |
| Loss | Giảm dần |

**Lưu ý:** Đây chỉ là kết quả **ban đầu**. Full training (300 epochs) sẽ tốt hơn nhiều!

## 📂 Output

```
runs/quick_train_11class/yolov12n_quick_test/
├── weights/
│   ├── best.pt                 # Best model (30 epochs)
│   └── last.pt
├── results.csv                 # Training metrics
├── results.png                 # Loss/mAP plots
├── confusion_matrix.png        # Confusion matrix
└── class_weights.yaml          # Class weights used
```

## ✅ Kiểm tra kết quả

### 1. Loss curves
```
results.png
```
- train/loss phải giảm dần
- val/loss không tăng (overfitting)

### 2. mAP curves
- mAP@50 phải tăng dần
- Có thể chưa ổn định sau 30 epochs (bình thường)

### 3. Confusion matrix
```
confusion_matrix.png
```
- Kiểm tra class nào bị nhầm lẫn
- Vehicle ↔ Person phải < 1%

## 🎯 Next Steps

### ✅ Nếu kết quả tốt (loss giảm, mAP tăng):

**Chạy full training:**
```powershell
python training\train_11class_final.py
```

### ⚠️ Nếu có vấn đề:

**1. Loss không giảm:**
- Kiểm tra dataset (labels có đúng không?)
- Thử giảm learning rate

**2. Val loss tăng nhanh:**
- Tăng augmentation
- Kiểm tra overfitting

**3. Một số class mAP = 0:**
- Kiểm tra class có đủ samples không
- Xem class weights

## 💡 Tips

### Tăng tốc độ training

**Option 1: Tăng batch size** (nếu GPU đủ mạnh)
```python
'batch_size': 32,  # Thay vì 16
```

**Option 2: Cache images** (nếu RAM đủ lớn)
```python
'cache': True,     # Load images to RAM
```

### Giảm thời gian

**Option 1: Giảm epochs**
```python
'epochs': 20,      # Thay vì 30
```

**Option 2: Giảm workers**
```python
'workers': 2,      # Thay vì 4
```

## 🔧 Troubleshooting

### GPU Out of Memory

```python
# Giảm batch size
'batch_size': 8,   # Hoặc 4
```

### Training quá chậm

```python
# Tăng workers
'workers': 8,
# Tắt một số augmentation
'mosaic': 0.5,
'mixup': 0.0,
```

### YOLOv12n không tìm thấy

Script sẽ tự động fallback về YOLOv8n. Hoặc download YOLOv12n:
```powershell
# Đảm bảo yolo12n.pt có trong thư mục gốc
ls yolo12n.pt
```

## 📝 Comparison

| Feature | Quick Train | Full Train |
|---------|-------------|------------|
| Purpose | Test & verify | Production model |
| Time | 30-45 min | 6-8 hours |
| Epochs | 30 | 300 |
| mAP@50 | ~0.40 | ~0.60+ |
| mAP@50-95 | ~0.25 | ~0.40+ |
| Ready for deploy? | ❌ No | ✅ Yes |
| Good for debugging? | ✅ Yes | ❌ Too long |

## 🎉 Summary

**Quick training is perfect for:**
- ✅ Testing pipeline lần đầu
- ✅ Verify dataset OK
- ✅ Debug issues nhanh
- ✅ Thử nghiệm hyperparameters

**NOT for:**
- ❌ Production deployment
- ❌ High accuracy requirements
- ❌ Final model evaluation

**After quick train passes → Run full training!**

---

**Quick Start:**
```powershell
python training\quick_train_yolov12.py
```

**Then:**
```powershell
python training\train_11class_final.py
```

Good luck! 🚀
