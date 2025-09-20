# Traffic AI YOLOv12 - 11-Class Object Detection# 🚗 Traffic AI YOLOv12 - Phân loại đối tượng giao thông



🚦 **Advanced traffic object detection system using YOLOv12 with optimized 11-class taxonomy**> **Đề tài**: Phân loại đối tượng tham gia giao thông sử dụng YOLOv12 (Biển báo, xe máy, người đi bộ, xe hơi, xe tải, xe đạp, ổ gà, xe bus). So sánh hiệu quả cân bằng và mất cân bằng dữ liệu.



## 📋 Project Overview![Project Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

![Python](https://img.shields.io/badge/Python-3.13-blue)

This project implements a comprehensive traffic object detection system using YOLOv12, specifically designed for Vietnamese traffic conditions. The system combines 4 different datasets and uses an optimized 11-class taxonomy for maximum accuracy and coverage.![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu118-red)

![CUDA](https://img.shields.io/badge/CUDA-11.8-green)

## 🎯 11-Class Taxonomy

## 📋 Tổng quan dự án

| ID | Class Name | Description | Source Datasets |

|----|------------|-------------|-----------------|Dự án này nghiên cứu hiệu quả của việc cân bằng dữ liệu trong phân loại đối tượng giao thông sử dụng YOLOv12. Hệ thống được thiết kế để:

| 0 | `pedestrian` | People walking/crossing roads | Object Detection 35, Intersection Flow 5K |

| 1 | `bicycle` | Bicycles and cycling | Object Detection 35, Intersection Flow 5K |- **🎯 Phân loại 8 loại đối tượng giao thông**: pedestrian, motorcycle, bicycle, car, truck, bus, pothole, traffic_sign

| 2 | `motorcycle` | Motorcycles, scooters, tricycles | Object Detection 35, Intersection Flow 5K |- **⚖️ So sánh** hiệu suất giữa dataset cân bằng và mất cân bằng

| 3 | `car` | Cars, passenger vehicles | Object Detection 35, Intersection Flow 5K |- **🤖 Tự động hóa** toàn bộ pipeline từ chuẩn bị dữ liệu đến đánh giá kết quả

| 4 | `bus` | Buses, public transport | Object Detection 35, Intersection Flow 5K |- **⚡ Tối ưu cho RTX 3050Ti** với 4GB VRAM

| 5 | `truck` | Trucks, commercial vehicles | Object Detection 35, Intersection Flow 5K |- **📊 Tạo báo cáo** nghiên cứu chi tiết với visualization

| 6 | `train` | Trains, railways | Object Detection 35 |

| 7 | `traffic_light` | Traffic lights | Object Detection 35 |## 🚦 Trạng thái dự án hiện tại

| 8 | `traffic_sign` | Traffic signs, road signs | Object Detection 35, VN Traffic Sign, Road Issues |

| 9 | `pothole` | Potholes, road damage | Object Detection 35, Road Issues |### ✅ HOÀN THÀNH

| 10 | `infrastructure` | Barriers, road infrastructure | Object Detection 35 |

1. **📥 Download datasets từ Kaggle** - 4 datasets đã tải về thành công

## 📊 Dataset Sources2. **🔍 Dataset analysis** - Kiểm tra format và cấu trúc thực tế  

3. **🔄 Dataset conversion** - Convert 100% chính xác:

### 1. Object Detection 35 (VisionGuard)   - VN Traffic Sign: 10,170 images (29 classes → traffic_sign)

- **Classes**: 35 (11 traffic-related)   - Road Issues: 4,025 images (categorical → pothole detection)

- **Images**: 15,893   - Intersection Flow 5K: 5,483 images (8 classes → 7 traffic classes)

- **Annotations**: 22,792    - **TOTAL**: 23,377 images với 668,497 annotations

- **Description**: Custom dataset for blind assistance with comprehensive object detection4. **🔗 Dataset merging** - Hợp nhất thành dataset thống nhất với 8 classes

- **Traffic Coverage**: Person, Car, Bus, Bicycle, Truck, Motorcycles, Traffic Light, Stop Sign, Barriers, Path Holes, Train5. **🎮 CUDA setup** - PyTorch 2.7.1+cu118 với RTX 3050Ti support

6. **🤖 YOLOv12n model** - Downloaded và ready for training (5.3MB, 2.6M parameters)

### 2. Intersection Flow 5K

- **Classes**: 8 traffic objects### 🔄 ĐANG THỰC HIỆN

- **Images**: ~13,856

- **Annotations**: ~406,000- **⚖️ Balanced vs Imbalanced training** - So sánh hiệu quả 2 approaches

- **Description**: Traffic intersection surveillance data- **🚀 GPU-optimized training** - RTX 3050Ti với batch=8, AMP=True

- **Coverage**: Pedestrian, bicycle, vehicle types, traffic flow

### 📋 TIẾP THEO

### 3. VN Traffic Sign

- **Classes**: 29 Vietnamese traffic signs1. **✅ Complete training comparison** - Hoàn thành so sánh balanced vs imbalanced

- **Images**: ~20,3402. **📊 Model evaluation** - Đánh giá chi tiết metrics và per-class performance

- **Annotations**: ~2,8003. **🌐 API deployment** - FastAPI service cho production

- **Description**: Vietnamese traffic signs dataset4. **📦 Export multiple formats** - ONNX, TensorRT, TorchScript

- **Coverage**: Local traffic signs and regulations5. **📄 Research report** - Báo cáo nghiên cứu hoàn chỉnh



### 4. Road Issues## ⚙️ Thông số kỹ thuật

- **Classes**: 7 infrastructure problems

- **Images**: 9,660- **💻 Hardware**: RTX 3050Ti 4GB VRAM, i5-12500H

- **Description**: Road infrastructure problems detection- **🤖 Model**: YOLOv12n (nano version) - 2.6M parameters, 272 layers

- **Coverage**: Potholes, broken road signs, damaged roads- **🔧 Framework**: Ultralytics YOLO, PyTorch 2.7.1+cu118

- **🎯 Batch size**: 8 (tối ưu cho 4GB VRAM)

## 🔧 Installation & Setup- **📐 Image size**: 640x640

- **⚡ Mixed Precision**: Enabled (AMP)

### Prerequisites- **📊 Dataset size**: 23,377 images, 668,497 annotations

```bash

# Python 3.8+## 🏗️ Cấu trúc dự án

pip install -r requirements.txt

```

# Install YOLOv12Traffic_AI_YOLOv12_jwt/

pip install ultralytics├── 📁 config/                  # Cấu hình hệ thống

│   └── taxonomy.yaml          # Mapping classes gốc → 8 classes chuẩn

# Kaggle CLI (for dataset download)├── 📁 datasets/               # Datasets đã xử lý (tạo tự động)

pip install kaggle│   ├── traffic_ai/           # Dataset gộp (imbalanced)

```│   └── traffic_ai_balanced/  # Dataset cân bằng (tạo khi training)

├── 📁 datasets_src/          # Datasets gốc (nguồn từ Kaggle)

### Dataset Download│   ├── intersection_flow_5k/ # Dataset giao lộ

```bash│   ├── vn_traffic_sign/     # Dataset biển báo VN

# Place kaggle.json in project root│   └── road_issues/         # Dataset vấn đề đường

# Run download script├── 📁 experiments/           # Kết quả thí nghiệm (tạo tự động)

./scripts/download_kaggle.ps1  # Windows│   └── balanced_vs_imbalanced/

./scripts/download_kaggle.sh   # Linux/Mac├── 📁 scripts/              # Scripts chính

```├── 📁 src/ai_service/       # API service

├── 📁 training/             # Scripts training

## 🚀 Usage├── yolo12n.pt              # YOLOv12 pretrained model (5.3MB)

├── requirements.txt        # Python dependencies

### 1. Dataset Preparation└── README.md              # Tài liệu này

```bash```

# Convert Road Issues to YOLO format

python scripts/convert_road_issues.py## 📊 Classes và Taxonomy



# Organize Object Detection 35 (keep 35 classes)```yaml

python scripts/organize_object_detection_35_keep_original.pyClasses (đã có đầy đủ 8 classes từ 3 datasets):

  0: pedestrian      # Người đi bộ (48,366 annotations - 7.2%)

# Generate 11-class taxonomy configuration  1: motorcycle      # Xe máy (6,330 annotations - 0.9%) 🔴 UNDERREPRESENTED

python scripts/complete_11class_taxonomy.py  2: bicycle         # Xe đạp (59,182 annotations - 8.9%)

```  3: car             # Xe hơi (476,412 annotations - 71.3%) 🟢 DOMINANT

  4: truck           # Xe tải (19,719 annotations - 2.9%) 🔴 UNDERREPRESENTED

### 2. Dataset Merging  5: bus             # Xe bus (7,578 annotations - 1.1%) 🔴 UNDERREPRESENTED

```bash  6: pothole         # Ổ gà đường (3,348 annotations - 0.5%) 🔴 MINORITY

# Merge all 4 datasets into 11-class taxonomy  7: traffic_sign    # Biển báo giao thông (47,562 annotations - 7.1%)

python scripts/merge_datasets_final_correct.py

```📊 THỐNG KÊ:

  • Total: 668,497 annotations từ 23,377 images

### 3. Training  • Imbalance ratio: 142.3:1 (car vs pothole)

```bash  • Status: 🔴 SEVERELY IMBALANCED

# Train YOLOv12 with balanced/imbalanced comparison```

python scripts/train_balanced_vs_imbalanced_fixed.py

```## 📄 Chi tiết các Scripts



### 4. Verification### 🔧 Scripts chính (`scripts/`)

```bash

# Verify dataset conversions#### `download_yolo12n.py` 📥

python scripts/verify_converted_datasets.py**Công dụng**: Download và setup YOLOv12n model

- **Chức năng**: 

# Quick model check  - Download YOLOv12n pretrained model từ Ultralytics

python scripts/quick_check_yolo12n.py  - Check GPU compatibility (RTX 3050Ti)

```  - Verify CUDA installation

- **Output**: `yolo12n.pt` model file (5.3MB, 2.6M parameters)

## 📁 Project Structure

#### `create_taxonomy_mapping.py` 📊

```**Công dụng**: Tạo taxonomy mapping từ classes gốc về 8 classes chuẩn

Traffic_AI_YOLOv12_jwt/- **Chức năng**: Map classes từ 3 datasets khác nhau về 8 classes thống nhất

├── config/                          # Configuration files- **Input**: Classes definitions từ datasets gốc

│   └── taxonomy_complete_11class.yaml- **Output**: `config/taxonomy.yaml`, validation report

├── datasets_src/                    # Source datasets (gitignored)

│   ├── intersection_flow_5k/#### `analyze_datasets_enhanced.py` 🔍

│   ├── object_detection_35_organized/**Công dụng**: Phân tích chi tiết toàn bộ datasets

│   ├── road_issues_yolo/- **Chức năng**: 

│   └── vn_traffic_sign/  - Đếm số lượng images/labels chính xác

├── data/                           # Processed datasets (gitignored)  - Phát hiện format datasets (YOLO/Classification/COCO)

│   └── traffic_ai_final_11class/  - Thống kê class distribution

├── scripts/                        # Processing scripts  - Đánh giá chất lượng datasets

│   ├── merge_datasets_final_correct.py  # ⭐ Main merger- **Output**: Detailed analysis report với recommendations

│   ├── complete_11class_taxonomy.py     # Taxonomy generator

│   ├── convert_road_issues.py           # Road Issues converter#### `merge_datasets_fixed.py` 🔄

│   ├── organize_object_detection_35_keep_original.py**Công dụng**: Gộp 3 datasets thành 1 dataset thống nhất

│   └── ...- **Chức năng**:

├── requirements.txt                # Python dependencies  - Convert classification → YOLO detection format

└── README.md                      # This file  - Apply taxonomy mapping

```  - Tạo train/val/test splits chuẩn (70/20/10)

  - Generate statistics và data.yaml

## 🎯 Key Features- **Output**: `datasets/traffic_ai/` (unified YOLO dataset)



- **Multi-Dataset Integration**: Combines 4 different traffic datasets#### `train_balanced_vs_imbalanced_fixed.py` ⚖️

- **Optimized Taxonomy**: 11 classes specifically chosen for traffic AI**Công dụng**: So sánh training giữa balanced vs imbalanced datasets

- **Vietnamese Traffic Focus**: Includes local traffic signs and conditions- **Chức năng**:

- **YOLO Format**: Ready for YOLOv12 training  - Analyze class imbalance của dataset (142.3:1 ratio)

- **Comprehensive Coverage**: Vehicles, pedestrians, infrastructure, signs  - Tạo balanced dataset bằng smart oversampling

- **Quality Assurance**: Verification and validation scripts  - Train 2 models riêng biệt với GPU optimization

  - So sánh metrics và tạo visualization

## 📈 Performance Metrics  - Export models (.pt, .onnx, .engine, .torchscript)

- **Tối ưu**: RTX 3050Ti 4GB VRAM, batch=8, AMP enabled

The merged dataset provides:- **Sử dụng**: `python scripts/train_balanced_vs_imbalanced_fixed.py`

- **Total Images**: ~59,749 images

- **Total Annotations**: ~431,592 annotations  ### 🌐 API Service (`src/ai_service/`)

- **Class Balance**: Optimized distribution across 11 classes

- **Coverage**: Comprehensive traffic scenarios#### `main.py` 🚀

- **Quality**: Verified and validated data**Công dụng**: FastAPI server cho inference

- **Chức năng**: REST API endpoints cho object detection

## 🛠️ Development- **Endpoints**: `/detect`, `/health`, `/model-info`

- **Input**: Image files qua HTTP

### Dataset Analysis- **Output**: JSON với detected objects và confidence scores

```bash

# Analyze Object Detection 35 classes#### `detect.py` 🎯

python scripts/analyze_object_detection_35_correct.py**Công dụng**: Core detection logic

```- **Chức năng**: 

  - Load trained YOLOv12 model

### Configuration  - Process images và return predictions

- Taxonomy mapping: `config/taxonomy_complete_11class.yaml`  - Handle batch processing

- Dataset configs: Each dataset has its own `data.yaml`- **Input**: Image arrays/files

- **Output**: Detection results với bounding boxes

## 📄 License

## ⚡ Quick Start

This project is released under MIT License. Individual datasets may have their own licenses:

- Object Detection 35: Custom license (VisionGuard project)### 🔍 Kiểm tra trạng thái cài đặt

- Intersection Flow 5K: Academic use

- VN Traffic Sign: Open dataset```bash

- Road Issues: CC0-1.0# Kiểm tra nhanh toàn bộ hệ thống

python scripts/status_check.py

## 🤝 Contributing

# Kiểm tra chi tiết YOLOv12n

1. Fork the repositorypython scripts/quick_check_yolo12n.py

2. Create feature branch

3. Test dataset processing# Kiểm tra toàn bộ dependencies

4. Submit pull requestpython scripts/check_yolo12n_installation.py



## 📞 Contact# Kiểm tra requirements.txt

python scripts/check_requirements.py

- **Project**: Traffic AI YOLOv12```

- **Repository**: [Traffic_AI_YOLOv12_jwt](https://github.com/lnhoangphu/Traffic_AI_YOLOv12_jwt)

- **Owner**: lnhoangphu### 1. Cài đặt môi trường



---```bash

# Clone repository

🚦 **Building safer roads through AI-powered traffic detection** 🚦git clone <repository-url>
cd Traffic_AI_YOLOv12_jwt

# Cài đặt dependencies  
pip install -r requirements.txt
```

### 2. Cấu hình Kaggle API

1. Đăng nhập [kaggle.com](https://kaggle.com)
2. Vào Account → API → Create New API Token
3. Đặt file `kaggle.json` vào thư mục gốc dự án

### 3. Chạy training comparison

```bash
# Chạy so sánh balanced vs imbalanced
python scripts/train_balanced_vs_imbalanced_fixed.py

# Hoặc với tùy chọn
python scripts/train_balanced_vs_imbalanced_fixed.py --epochs 50
```

### 4. Khởi động API service

```bash
cd src/ai_service
python main.py
```

API sẽ chạy tại `http://localhost:8000`

## 📊 Kết quả và báo cáo

Sau khi chạy pipeline, các kết quả sẽ được tạo tại:

- **`experiments/balanced_vs_imbalanced/`**: Training results
  - `imbalanced/`: Results từ imbalanced dataset
  - `balanced/`: Results từ balanced dataset
  - `comparison_plots/`: Visualization so sánh
  - `*_exports/`: Multiple model formats (.pt, .onnx, .engine)

- **`reports/`**: Research reports
  - `traffic_classification_report.md`: Final research report
  - `pipeline_results.json`: Detailed results

## 📈 Methodology

### Câu hỏi nghiên cứu
> Làm thế nào việc mất cân bằng dữ liệu ảnh hưởng đến hiệu suất phân loại đối tượng giao thông sử dụng YOLOv12?

### Phương pháp
1. **Dataset Preparation**: Gộp 3 datasets thành unified format
2. **Taxonomy Standardization**: Map về 8 classes chuẩn
3. **Smart Balancing**: Oversampling với disk space optimization
4. **GPU-Optimized Training**: RTX 3050Ti với AMP, batch=8
5. **Comprehensive Evaluation**: mAP@50, mAP@50-95, per-class analysis
6. **Multi-Format Export**: .pt, .onnx, .engine, .torchscript

### Dataset Statistics
```
Total: 23,377 images, 668,497 annotations
Imbalance ratio: 142.3:1 (car vs pothole)
Train/Val/Test split: 70%/20%/10%
```

## ⚙️ Configuration

### `config/taxonomy.yaml` 📝
**Công dụng**: Định nghĩa mapping rules từ classes gốc về 8 classes chuẩn
- **Chứa**: Mapping tables cho từng dataset
- **Tự động tạo**: bởi `create_taxonomy_mapping.py`

### `requirements.txt` 📦
**Công dụng**: Định nghĩa Python dependencies
- **Core**: ultralytics, torch, torchvision, matplotlib
- **API**: fastapi, uvicorn, python-multipart
- **ML**: numpy, pandas, opencv-python, Pillow

## 🚨 Troubleshooting

### Common Issues

1. **Kaggle API Error**
   ```bash
   # Kiểm tra kaggle.json permissions
   chmod 600 kaggle.json  # Linux/Mac
   # Hoặc đặt file trong %USERPROFILE%/.kaggle/ (Windows)
   ```

2. **CUDA/GPU Issues**
   ```python
   # Kiểm tra CUDA availability
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

3. **Memory Issues (VRAM 4GB)**
   ```python
   # Giảm batch size nếu gặp OOM
   batch=4  # thay vì 8
   # Hoặc giảm image size
   imgsz=480  # thay vì 640
   ```

4. **Import Errors**
   ```bash
   # Cài đặt lại dependencies
   pip install -r requirements.txt --force-reinstall
   ```

5. **YOLOv12 Model Issues**
   ```bash
   # Download lại model nếu corrupted
   python scripts/download_yolo12n.py
   ```

## 📊 Expected Results

### Training Performance
- **Training time**: ~2-3 hours per model (30 epochs) trên RTX 3050Ti
- **Memory usage**: ~3.5GB VRAM với batch=8
- **Model size**: ~5.3MB (.pt), ~10MB (.onnx)

### Evaluation Metrics
- **mAP@50**: Expected 0.6-0.8 range
- **mAP@50-95**: Expected 0.4-0.6 range
- **Inference speed**: ~50-100 FPS trên RTX 3050Ti

### Class Performance
- **High performance**: car, pedestrian, traffic_sign (abundant data)
- **Medium performance**: bicycle, truck (moderate data)
- **Challenging**: motorcycle, bus, pothole (limited data)

## 🔮 Future Work

1. **🌦️ Weather Augmentation**: Thêm rain/fog/snow effects
2. **🎯 Advanced Balancing**: GAN-based synthetic data generation
3. **📱 Mobile Deployment**: TensorRT/TFLite optimization
4. **🔄 Real-time Processing**: Video stream analysis
5. **📊 Advanced Metrics**: Class-specific evaluation
6. **🌐 Web Interface**: User-friendly detection interface

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Tạo Pull Request

## 📝 License

MIT License - xem file LICENSE để biết chi tiết.

## 📚 References

- [YOLOv12 Paper](https://arxiv.org/abs/2502.12524)
- [Ultralytics Documentation](https://docs.ultralytics.com/vi/models/yolo12/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)

## 📧 Contact

Nếu có thắc mắc hoặc cần hỗ trợ, vui lòng tạo issue trong repository này.

---

**🎓 Dự án đại học - Phân loại đối tượng giao thông với YOLOv12**

*Dự án được phát triển với mục đích học tập và nghiên cứu. Đảm bảo tuân thủ các quy định giao thông khi sử dụng trong thực tế.*