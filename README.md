# 🚗 Traffic AI - YOLOv12 Detection System

> AI-powered traffic detection system với YOLOv12, phát hiện 11 loại đối tượng giao thông

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv12](https://img.shields.io/badge/YOLOv12-latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [11 Classes phát hiện](#-11-classes-phát-hiện)
- [Cài đặt](#-cài-đặt)
- [Dataset](#-dataset)
- [Training](#-training)
- [API Server](#-api-server)
- [Scripts](#-scripts-chính)
- [Kết quả](#-kết-quả-training)

---

## 🎯 Giới thiệu

Hệ thống phát hiện đối tượng giao thông sử dụng YOLOv12n với **11 classes** được tối ưu cho giao thông Việt Nam:

### Tính năng chính:
- ✅ **11 classes** phát hiện đối tượng giao thông
- ✅ **YOLOv12n** - Model nhẹ, nhanh (< 5MB)
- ✅ **FastAPI** - REST API server
- ✅ **Balanced & Imbalanced datasets** - So sánh performance
- ✅ **Class weighting** - Tối ưu cho rare classes
- ✅ **Real-time detection** - Camera support

---

## 🏷️ 11 Classes Phát hiện

| Class ID | Tên Class | Mô tả |
|----------|-----------|-------|
| 0 | **Vehicle** | Xe hơi, ô tô |
| 1 | **Bus** | Xe buýt |
| 2 | **Bicycle** | Xe đạp |
| 3 | **Person** | Người đi bộ |
| 4 | **Engine** | Xe máy |
| 5 | **Truck** | Xe tải |
| 6 | **Tricycle** | Xe ba bánh |
| 7 | **Obstacle** | Chướng ngại vật |
| 8 | **Pothole** | Ổ gà, hố đường |
| 9 | **Traffic Light** | Đèn giao thông |
| 10 | **Traffic Sign** | Biển báo giao thông |

---

## 🔧 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/lnhoangphu/Traffic_AI_YOLOv12_jwt.git
cd Traffic_AI_YOLOv12_jwt
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 4. Kiểm tra setup
```bash
python setup_check.py
```

**Output mong đợi:**
```
✅ ultralytics đã được cài đặt
✅ opencv-python đã được cài đặt
✅ matplotlib đã được cài đặt
✅ yolo12n.pt đã tồn tại
✅ TẤT CẢ DEPENDENCIES ĐÃ SẴN SÀNG!
```

---

## 📊 Dataset

### Datasets hiện có:

#### 1. **Balanced Dataset** (Khuyến nghị)
```
Path: datasets/traffic_ai_final_balanced/
Train: 24,000 images
Val:   3,000 images
Test:  3,000 images
Total: 30,000 images
```

**Đặc điểm:**
- ✅ **Đã cân bằng classes** (oversample rare, undersample common)
- ✅ **Augmentation adaptive** (strong cho rare classes)
- ✅ Training nhanh (~6-8h cho 300 epochs)
- 🎯 **Dùng cho production**

**Phân phối:**
- Vehicle: 20%
- Person: 12%
- Bicycle/Engine: 10%
- Bus/Truck/Tricycle/Obstacle: 8%
- Traffic Light: 6%
- Traffic Sign/Pothole: 5%

#### 2. **Imbalanced Dataset** (So sánh)
```
Path: datasets/traffic_ai_final_imbalanced/
Train: 21,420 images
Val:   2,677 images
Test:  2,678 images
Total: 26,775 images
```

**Đặc điểm:**
- ⚖️ **Giữ phân phối tự nhiên** (không cân bằng)
- ❌ Không augmentation
- 📊 Phản ánh thực tế giao thông

**Phân phối:**
- Person: 52.43% (dominant)
- Tricycle: 22.57%
- Vehicle: 11.93%
- Traffic Light: 0.16% (very rare)

### Tạo dataset mới:

#### Balanced dataset:
```bash
python scripts/create_balanced_dataset.py \
    --target-images 30000 \
    --balance-mode semi-balanced
```

#### Imbalanced dataset:
```bash
python scripts/create_imbalanced_dataset.py \
    --output datasets/traffic_ai_final_imbalanced
```

---

## 🚀 Training

### Quick Training (30 epochs - Test nhanh)
```bash
python training/quick_train_yolov12.py
```
- ⏱️ Thời gian: ~30-45 phút
- 📊 Kết quả: mAP@50 ~40-50%
- 🎯 Mục đích: Kiểm tra pipeline

### Full Training (300 epochs - Production)

#### Option 1: Train trên Balanced Dataset
```bash
python training/train_11class_final.py
```

**Cấu hình:**
- Model: YOLOv12n
- Epochs: 300
- Batch: 16
- Image size: 640
- Class weights: Auto (inverse frequency)
- Augmentation: mosaic, mixup, copy-paste

**Kết quả mong đợi:**
- mAP@50: 60-70%
- mAP@50-95: 40-50%
- Training time: ~6-8 giờ

#### Option 2: Two-Stage Training (Khuyến nghị)
```bash
# Stage 1: Pretrain trên Balanced (học tất cả classes)
python training/train_11class_final.py \
    --data datasets/traffic_ai_final_balanced/data.yaml \
    --epochs 50

# Stage 2: Fine-tune trên Imbalanced (adapt thực tế)
python training/train_11class_final.py \
    --data datasets/traffic_ai_final_imbalanced/data.yaml \
    --weights runs/train_11class_final/yolov12n_balanced/weights/best.pt \
    --epochs 50 \
    --lr 0.0001
```

### Evaluation
```bash
python training/evaluate_11class.py \
    --weights runs/train_11class_final/yolov12n_11class_weighted/weights/best.pt \
    --data datasets/traffic_ai_final_balanced/data.yaml \
    --split test
```

### So sánh kết quả:
```bash
python training/compare_results.py
```

---

## 🌐 API Server

### Start server:
```bash
python run_api.py
```

**Hoặc với uvicorn trực tiếp:**
```bash
uvicorn src.ai_service.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Detect từ file upload
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "conf_threshold=0.25" \
  -F "return_image=true"
```

#### 3. Detect từ URL
```bash
curl -X POST "http://localhost:8000/detect-url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg", "conf_threshold": 0.25}'
```

### API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📜 Scripts Chính

### Dataset Management:
```bash
# Phân tích dataset
python scripts/analyze_11class_dataset.py

# Kiểm tra datasets sẵn sàng
python scripts/check_datasets_ready.py

# Validate class mapping
python scripts/validate_class_mapping.py

# Tạo balanced dataset
python scripts/create_balanced_dataset.py

# Tạo imbalanced dataset
python scripts/create_imbalanced_dataset.py
```

### Data Processing:
```bash
# Merge datasets
python scripts/merge_datasets_final_correct.py

# Convert road issues dataset
python scripts/convert_road_issues.py

# Filter Object Detection 35
python scripts/filter_object_detection_35.py

# Complete taxonomy
python scripts/complete_11class_taxonomy.py
```

### Model & Testing:
```bash
# Download YOLO weights
python scripts/download_yolo12n.py

# Quick model test
python test_model_quick.py

# API test
python test_api.py

# Check GPU
python check_gpu.py
```

---

## 📈 Kết quả Training

### Latest Model Performance:

**Model:** YOLOv12n 11-Class Weighted  
**Dataset:** traffic_ai_balanced_11class_processed  
**Epochs:** 300

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 59.5% |
| **mAP@0.5:0.95** | 42.3% |
| **Precision** | 70.9% |
| **Recall** | 52.1% |
| **Training Time** | ~8.5 hours |

### Per-Class Performance:

| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Vehicle | 0.85 | 0.72 | 0.78 |
| Bus | 0.72 | 0.65 | 0.68 |
| Bicycle | 0.68 | 0.58 | 0.62 |
| Person | 0.75 | 0.48 | 0.55 |
| Engine | 0.70 | 0.55 | 0.60 |
| Truck | 0.65 | 0.50 | 0.55 |
| Tricycle | 0.78 | 0.68 | 0.72 |
| Obstacle | 0.55 | 0.42 | 0.48 |
| Pothole | 0.60 | 0.45 | 0.50 |
| Traffic Light | 0.65 | 0.38 | 0.45 |
| Traffic Sign | 0.68 | 0.52 | 0.58 |

---

## 📂 Cấu trúc Project

```
Traffic_AI_YOLOv12_jwt/
├── datasets/                          # Datasets
│   ├── traffic_ai_final_balanced/     # Balanced dataset ✅
│   └── traffic_ai_final_imbalanced/   # Imbalanced dataset
├── scripts/                           # Processing scripts
│   ├── create_balanced_dataset.py     # Tạo balanced dataset
│   ├── create_imbalanced_dataset.py   # Tạo imbalanced dataset
│   ├── analyze_11class_dataset.py     # Phân tích dataset
│   ├── check_datasets_ready.py        # Kiểm tra datasets
│   └── ...
├── training/                          # Training scripts
│   ├── train_11class_final.py         # Full training
│   ├── quick_train_yolov12.py         # Quick test
│   ├── evaluate_11class.py            # Evaluation
│   └── compare_results.py             # So sánh kết quả
├── src/                               # Source code
│   └── ai_service/                    # API service
│       ├── main.py                    # FastAPI app
│       └── detect.py                  # Detection logic
├── runs/                              # Training outputs
│   ├── train_11class_final/           # Full training results
│   └── quick_train_11class/           # Quick test results
├── yolo12n.pt                         # YOLOv12 pretrained weights
├── requirements.txt                   # Dependencies
├── run_api.py                         # Run API server
├── setup_check.py                     # Setup verification
└── README.md                          # This file
```

---

## ⚙️ Configuration

### Environment Variables (.env):
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# AI Service
AI_CONFIDENCE_THRESHOLD=0.25
AI_IOU_THRESHOLD=0.45
AI_MAX_DETECTIONS=100

# CORS
AI_ALLOWED_ORIGINS=*

# File Upload
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp

# Logging
LOG_LEVEL=info
```

---

## 🐛 Troubleshooting

### 1. Lỗi import ultralytics
```bash
pip install --upgrade ultralytics
```

### 2. CUDA out of memory
```python
# Giảm batch size trong training script
batch = 8  # thay vì 16
```

### 3. Model không load được
```bash
# Redownload weights
python scripts/download_yolo12n.py
```

### 4. API không start
```bash
# Kiểm tra port
netstat -ano | findstr :8000

# Kill process nếu port bị chiếm
taskkill /PID <PID> /F
```

---

## 📖 So sánh Balanced vs Imbalanced

| Tiêu chí | Balanced ✅ | Imbalanced |
|----------|------------|------------|
| **Kích thước** | 30,000 | 26,775 |
| **Training time** | 6-8h | 6-8h |
| **mAP@50** | 60-70% | 55-65% |
| **Rare classes** | Tốt | Kém |
| **Common classes** | Tốt | Tốt |
| **Production** | ✅ Khuyến nghị | ❌ Research only |

### Khi nào dùng Balanced:
- ✅ Cần detect tốt tất cả classes
- ✅ Quan trọng rare classes (Pothole, Traffic Light)
- ✅ Training model production
- ✅ Safety-critical applications

### Khi nào dùng Imbalanced:
- 📊 Research: So sánh với balanced
- 🧪 Experiment: Class weighting techniques
- 🌍 Phản ánh phân phối thực tế
- ❌ KHÔNG dùng production trừ khi có lý do cụ thể

---

## 📝 Training Tips

### 1. Data Augmentation
- Mosaic, mixup, copy-paste đã enable mặc định
- Strong augmentation cho rare classes
- Có thể điều chỉnh trong training script

### 2. Class Weights
```python
# Auto calculate trong script
class_weights = {
    "Person": 2.5,      # Rare class → weight cao
    "Traffic Light": 3.0,
    "Vehicle": 0.5,     # Common class → weight thấp
    # ...
}
```

### 3. Learning Rate Schedule
- Initial LR: 0.01
- Cosine annealing với warmup
- Auto adjust based on loss plateau

### 4. Early Stopping
- Patience: 50 epochs
- Monitor: val/mAP@50
- Save best weights

---

## 🎓 Citation

Nếu sử dụng project này, vui lòng cite:

```bibtex
@misc{traffic_ai_yolov12,
  author = {Your Name},
  title = {Traffic AI - YOLOv12 Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/lnhoangphu/Traffic_AI_YOLOv12_jwt}
}
```

---

## 📧 Contact

- **Author:** Lê Nhật Hoàng Phú
- **GitHub:** [@lnhoangphu](https://github.com/lnhoangphu)
- **Repository:** [Traffic_AI_YOLOv12_jwt](https://github.com/lnhoangphu/Traffic_AI_YOLOv12_jwt)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- Các datasets nguồn:
  - Intersection-Flow-5K
  - VN Traffic Sign
  - Road Issues
  - Object Detection 35

---

**Made with ❤️ for Vietnamese Traffic Safety** 🚦
