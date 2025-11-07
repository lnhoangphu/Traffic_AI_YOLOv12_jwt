# 📊 GIẢI THÍCH CÁC DATASET

## 🎯 Tổng quan

Trong project có **4 datasets**, nhưng chỉ **1 dataset đang được sử dụng**:

| Dataset | Train | Val | Test | Trạng thái | Mục đích |
|---------|-------|-----|------|------------|----------|
| **traffic_ai_balanced_11class_processed** | 3,364 | 961 | 482 | ✅ **ĐANG DÙNG** | Production |
| traffic_ai_imbalanced_11class_processed | 18,271 | 5,220 | 2,611 | ❌ Không dùng | So sánh |
| traffic_ai_11class_final | - | - | - | ❌ Rỗng | Cũ (bỏ) |
| traffic_ai_11class_rebalanced | 29,030 | 31,757 | 32,519 | ❌ Không dùng | Thử nghiệm |

---

## 📦 Chi tiết từng Dataset

### 1. ✅ **traffic_ai_balanced_11class_processed** (ĐANG DÙNG)

```yaml
Path: datasets/traffic_ai_balanced_11class_processed/
Train: 3,364 images
Val:   961 images  
Test:  482 images
Total: 4,807 images
```

**Đặc điểm:**
- ✅ **Dataset chính thức đang được sử dụng**
- ✅ Đã được cân bằng (balanced) để giảm class imbalance
- ✅ Kích thước vừa phải → Training nhanh (~30-45 phút cho 30 epochs)
- ✅ Đã train model: `runs/quick_train_11class/yolov12n_quick_test/weights/best.pt`
- ✅ Kết quả tốt: **mAP@50 = 59.5%**

**Phân bố class:**
```
Vehicle:       50% (82,185 objects)
Tricycle:      27% (33,707 objects)
Person:        16% (117 objects - rất ít!)
Bus:           2.8% (3,425 objects)
[Các class khác < 1%]
```

**Khi nào dùng:**
- ✅ Training production model
- ✅ Quick testing (30 epochs)
- ✅ Demo và API deployment
- ✅ Cân bằng giữa tốc độ và accuracy

---

### 2. ❌ **traffic_ai_imbalanced_11class_processed** (KHÔNG DÙNG)

```yaml
Path: datasets/traffic_ai_imbalanced_11class_processed/
Train: 18,271 images
Val:   5,220 images
Test:  2,611 images
Total: 26,102 images
```

**Đặc điểm:**
- ❌ **Không cân bằng** (imbalanced) - một số class rất ít
- 📊 Dataset lớn hơn 5.4x so với balanced
- ⏱️ Training chậm hơn nhiều (~2-3 giờ cho 30 epochs)
- 🎯 Dùng để **so sánh hiệu quả** balanced vs imbalanced

**Vấn đề:**
- Class phân bố không đều
- Model dễ bị bias về class đông (Vehicle, Tricycle)
- Class hiếm (Person, Obstacle, Pothole) bị học kém

**Khi nào dùng:**
- 📊 Research: So sánh balanced vs imbalanced
- 🧪 Experiment: Test ảnh hưởng của class weights
- ❌ KHÔNG dùng cho production

---

### 3. ❌ **traffic_ai_11class_final** (RỖng - BỎ)

```yaml
Path: datasets/traffic_ai_11class_final/
Train: KHÔNG CÓ
Val:   KHÔNG CÓ
Test:  KHÔNG CÓ
```

**Đặc điểm:**
- 🗑️ **Dataset cũ, chỉ có file config**
- ❌ Không có ảnh/labels → Không sử dụng được
- 📝 Chỉ còn lại `data.yaml` (link đến balanced dataset)

**Kết luận:**
- ❌ **XÓA được** - không ảnh hưởng gì
- Hoặc giữ lại làm config backup

---

### 4. ❌ **traffic_ai_11class_rebalanced** (THỬ NGHIỆM)

```yaml
Path: datasets/traffic_ai_11class_rebalanced/
Train: 29,030 images
Val:   31,757 images (!!)
Test:  32,519 images (!!)
Total: 93,306 images
```

**Đặc điểm:**
- 🔬 **Dataset thử nghiệm với augmentation mạnh**
- 📈 Rất lớn: 19.4x so với balanced dataset
- ⚠️ **Validation > Training** (bất thường!)
- 🐌 Training RẤT chậm (~6-8 giờ cho 30 epochs)
- 🎲 Có nhiều augmented images (aug1049, aug1129...)

**Vấn đề:**
- Split không chuẩn (val/test > train)
- Quá nhiều augmentation → có thể overfit
- Tốn nhiều thời gian training
- Chưa verify hiệu quả

**Khi nào dùng:**
- 🧪 Research: Test extreme augmentation
- 📊 Benchmark: So sánh với balanced
- ❌ KHÔNG dùng cho production (chưa proven)

---

## 🎯 KHUYẾN NGHỊ NGAY BÂY GIỜ

### ✅ SỬ DỤNG
```bash
datasets/traffic_ai_balanced_11class_processed/
```

**Lý do:**
1. ✅ Đã train thành công với kết quả tốt (mAP@50 = 59.5%)
2. ✅ Training nhanh (30-45 phút)
3. ✅ Kích thước hợp lý (4,807 images)
4. ✅ Đang được dùng trong API
5. ✅ Model weights có sẵn

### ❌ KHÔNG SỬ DỤNG

**Imbalanced dataset:**
- Chỉ dùng khi cần so sánh performance
- Không dùng cho production

**Rebalanced dataset:**
- Chưa verify hiệu quả
- Training quá lâu
- Split không chuẩn

**11class_final:**
- Rỗng → Xóa hoặc ignore

---

## 📝 HÀNH ĐỘNG NGAY

### 1. Giữ lại (KEEP)
```
✅ datasets/traffic_ai_balanced_11class_processed/  → PRODUCTION
```

### 2. Có thể xóa (OPTIONAL DELETE)
```
❌ datasets/traffic_ai_imbalanced_11class_processed/  → 26GB (nếu không nghiên cứu)
❌ datasets/traffic_ai_11class_rebalanced/             → 93GB (nếu không thử nghiệm)  
❌ datasets/traffic_ai_11class_final/                  → Rỗng
```

**Lợi ích khi xóa:**
- Tiết kiệm ~120GB dung lượng
- Workspace sạch sẽ hơn
- Tránh nhầm lẫn

**Lưu ý:**
- Backup trước khi xóa (nếu muốn research sau)
- Chỉ xóa khi chắc chắn không cần

---

## 🔄 TRAINING WORKFLOW HIỆN TẠI

```mermaid
graph LR
    A[Balanced Dataset<br/>4,807 images] --> B[Quick Train<br/>30 epochs]
    B --> C[Model<br/>mAP@50=59.5%]
    C --> D[API Deployment]
    
    style A fill:#90EE90
    style C fill:#87CEEB
```

**File paths:**
```bash
# Dataset
datasets/traffic_ai_balanced_11class_processed/data.yaml

# Training scripts
training/quick_train_yolov12.py        # 30 epochs (~45 min)
training/train_11class_final.py        # 300 epochs (~6-8h)

# Trained model
runs/quick_train_11class/yolov12n_quick_test/weights/best.pt

# API
src/ai_service/detect.py               # Auto load best.pt
src/ai_service/main.py                 # FastAPI server
```

---

## 🚀 NEXT STEPS

### Nếu muốn accuracy cao hơn:

#### Option 1: Full Training (Balanced)
```bash
python training/train_11class_final.py
# → 300 epochs trên balanced dataset
# → ~6-8 giờ
# → mAP@50 dự kiến: ~65-70%
```

#### Option 2: Train trên Imbalanced (Thử nghiệm)
```bash
# Sửa data_yaml trong training script
# → 300 epochs trên imbalanced dataset  
# → ~20-24 giờ
# → So sánh với balanced
```

#### Option 3: Collect thêm data
- Tập trung vào class hiếm: Person, Obstacle, Pothole
- Augment class hiếm
- Retrain

---

## 📊 TỔNG KẾT

| Tiêu chí | Balanced ✅ | Imbalanced | Rebalanced |
|----------|------------|------------|------------|
| **Kích thước** | 4,807 ✅ | 26,102 | 93,306 |
| **Training time** | 30-45min ✅ | 2-3h | 6-8h |
| **Đã train** | ✅ Yes | ❌ No | ❌ No |
| **mAP@50** | 59.5% ✅ | Unknown | Unknown |
| **Production** | ✅ YES | ❌ No | ❌ No |
| **Khuyến nghị** | **USE NOW** | Research | Experiment |

---

**🎯 KẾT LUẬN:**
- ✅ Tiếp tục dùng **balanced_11class_processed**
- ✅ Model hiện tại đã tốt (mAP@50 = 59.5%)
- ✅ Có thể xóa các dataset khác để tiết kiệm dung lượng
- 🚀 Nếu cần accuracy cao hơn → Full training 300 epochs

