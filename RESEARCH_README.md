# 🚗 Traffic AI YOLOv12 - Balanced vs Imbalanced Dataset Research

## 📋 Mô tả đề tài
Nghiên cứu ảnh hưởng của **cân bằng và mất cân bằng dữ liệu** đến performance của model YOLOv12 trong việc nhận diện các object giao thông tại Việt Nam.

## 🎯 Mục tiêu nghiên cứu
1. **So sánh performance** giữa balanced và imbalanced datasets
2. **Phân tích ảnh hưởng** của data imbalance đến từng class
3. **Đưa ra khuyến nghị** về chiến lược data preparation cho traffic AI

## 📊 Dataset Information

### 🔢 11-Class Taxonomy
| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Vehicle | Các phương tiện giao thông khác |
| 1 | Bus | Xe buýt, xe khách |
| 2 | Bicycle | Xe đạp các loại |
| 3 | Person | Tất cả người (đi bộ, trên xe) |
| 4 | Engine | Xe 2 bánh có động cơ |
| 5 | Truck | Xe tải các loại |
| 6 | Tricycle | Xe 3 bánh |
| 7 | Obstacle | Vật cản, chướng ngại vật |
| 8 | Pothole | Ổ gà, hư hỏng mặt đường |
| 9 | Traffic Light | Đèn giao thông |
| 10 | Traffic Sign | Biển báo giao thông |

### 📁 Source Datasets
1. **Object Detection 35** - 15,893 images, VisionGuard dataset
2. **Intersection Flow 5K** - 6,928 images, Traffic flow surveillance
3. **VN Traffic Sign** - 9,000 images, Vietnamese traffic signs
4. **Road Issues** - 9,660 images, Infrastructure problems

### ⚖️ Generated Research Datasets

#### Balanced Dataset
- **Samples per class**: 437 (cân bằng hoàn toàn)
- **Total images**: 4,807
- **Coefficient of Variation**: 0.000
- **Purpose**: Nghiên cứu performance khi data được cân bằng

#### Imbalanced Dataset
- **Total images**: 26,102
- **Class distribution**: Giữ nguyên phân phối tự nhiên
- **Coefficient of Variation**: 0.775
- **Most samples**: Traffic Sign (11,248), Person (7,640), Vehicle (7,496)
- **Least samples**: Traffic Light (437), Obstacle (503)

## 🚀 Hướng dẫn sử dụng

### 1. Chuẩn bị Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure YOLOv12 model available
# yolo12n.pt should be in project root
```

### 2. Tạo Balanced & Imbalanced Datasets
```bash
# Chạy script tạo 2 datasets
python scripts/create_balanced_imbalanced_datasets.py

# Kết quả:
# datasets/traffic_ai_balanced_11class/
# datasets/traffic_ai_imbalanced_11class/
```

### 3. Training Models
```bash
# Option 1: Train cả 2 models tự động
python scripts/train_balanced_vs_imbalanced.py --mode full --epochs 100

# Option 2: Train riêng từng model
python scripts/train_balanced_vs_imbalanced.py --mode balanced --epochs 100
python scripts/train_balanced_vs_imbalanced.py --mode imbalanced --epochs 100

# Option 3: Chỉ verify datasets
python scripts/train_balanced_vs_imbalanced.py --mode verify
```

### 4. Phân tích kết quả
```bash
# Chạy analysis sau khi training xong
python scripts/analyze_balanced_vs_imbalanced.py

# Kết quả analysis sẽ có trong:
# training/analysis/
```

## 📈 Expected Results Structure

```
training/
├── runs/
│   ├── balanced/
│   │   └── yolov12_balanced_experiment/
│   │       ├── weights/
│   │       ├── results.csv
│   │       ├── confusion_matrix.png
│   │       └── ...
│   └── imbalanced/
│       └── yolov12_imbalanced_experiment/
│           ├── weights/
│           ├── results.csv
│           ├── confusion_matrix.png
│           └── ...
├── analysis/
│   ├── training_curves.png
│   ├── metrics_comparison.png
│   ├── final_metrics_comparison.csv
│   └── analysis_report.txt
└── comparison_report.txt
```

## 📊 Key Research Metrics

### 1. Overall Performance
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision & Recall**: Overall detection accuracy

### 2. Class-specific Analysis
- **Per-class AP**: Average Precision cho từng class
- **Minority class performance**: Đặc biệt quan tâm Traffic Light, Obstacle
- **Majority class stability**: Vehicle, Person, Traffic Sign

### 3. Training Dynamics
- **Convergence speed**: Tốc độ hội tụ của loss
- **Training stability**: Độ ổn định trong quá trình training
- **Overfitting tendency**: Xu hướng overfit của từng dataset

## 🔍 Research Questions

1. **Primary Question**: Liệu balanced dataset có cải thiện overall performance không?

2. **Secondary Questions**:
   - Class nào hưởng lợi nhiều nhất từ data balancing?
   - Trade-off giữa balanced vs natural distribution?
   - Balanced dataset có giúp minority classes detect tốt hơn?

3. **Practical Questions**:
   - Nên dùng approach nào cho traffic AI deployment?
   - Cost-benefit của data balancing effort?

## 📋 Research Methodology

### Experiment Design
1. **Control Variables**: Model architecture (YOLOv12n), training hyperparameters
2. **Independent Variable**: Dataset balance (balanced vs imbalanced)
3. **Dependent Variables**: mAP, precision, recall, per-class performance

### Statistical Analysis
- **Descriptive Statistics**: Mean, std deviation của metrics
- **Comparative Analysis**: Balanced vs Imbalanced performance
- **Effect Size**: Magnitude của improvement/degradation

## 🎯 Expected Outcomes

### Scenario 1: Balanced Dataset Superior
- **Finding**: Balanced shows higher mAP, especially for minority classes
- **Implication**: Data balancing worth the effort for traffic AI
- **Recommendation**: Implement balancing strategies in production

### Scenario 2: Imbalanced Dataset Sufficient
- **Finding**: Natural distribution performs equally well or better
- **Implication**: Real-world distribution already optimal
- **Recommendation**: Focus on data quality over balancing

### Scenario 3: Mixed Results
- **Finding**: Trade-offs between overall vs class-specific performance
- **Implication**: Need hybrid approaches
- **Recommendation**: Class-weighted loss or focal loss

## 📝 Report Template

### Abstract
- Objective, methodology, key findings, implications

### Introduction
- Problem statement, research questions, hypothesis

### Literature Review
- Data imbalance in computer vision
- Object detection challenges
- Traffic AI specific considerations

### Methodology
- Dataset preparation
- Model architecture
- Training procedure
- Evaluation metrics

### Results
- Quantitative results (tables, charts)
- Statistical significance tests
- Per-class analysis

### Discussion
- Interpretation of findings
- Comparison with literature
- Limitations and threats to validity

### Conclusion
- Key contributions
- Practical implications
- Future research directions

## 🔧 Advanced Configurations

### Custom Training Parameters
```python
# Modify base_config in train_balanced_vs_imbalanced.py
base_config = {
    'epochs': 200,          # Tăng epochs cho better convergence
    'batch': 32,            # Tăng batch size nếu có GPU
    'device': 'cuda',       # Sử dụng GPU
    'lr0': 0.01,           # Learning rate
    'patience': 20,         # Early stopping patience
    # ... other parameters
}
```

### Class Weight Implementation
```python
# Để implement class weights cho imbalanced dataset
# Thêm vào training config:
'class_weights': [0.1, 0.2, 0.3, ...]  # Inverse frequency weights
```

## 📚 References

1. Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
2. Buda, M., et al. "A systematic study of the class imbalance problem in convolutional neural networks." Neural Networks 2018.
3. Johnson, J. M., & Khoshgoftaar, T. M. "Survey on deep learning with class imbalance." Journal of Big Data 2019.

## 🤝 Contributors

- **Research Team**: [Your Names]
- **Advisor**: [Advisor Name]
- **Institution**: [University/Organization]

## 📄 License

This research project is for academic purposes. Dataset sources have their own licenses.

---

**🎯 Research Goal**: Contribute to the understanding of data balance impact on traffic AI systems, providing evidence-based recommendations for real-world deployment.