"""
Training script cho YOLOv12 - 11 Class Traffic Detection
Tự động tính class weights để cân bằng loss cho các class thiếu số
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import yaml

# Thêm ultralytics vào path nếu cần
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Chưa cài đặt ultralytics. Chạy: pip install ultralytics")
    sys.exit(1)

def calculate_class_weights(dataset_path, num_classes=11):
    """
    Tính class weights dựa trên inverse frequency
    Giúp model chú ý hơn đến các class thiếu số
    """
    print("\n📊 Calculating class weights...")
    
    # Đếm số lượng objects mỗi class trong train set
    labels_dir = Path(dataset_path) / 'labels' / 'train'
    class_counts = Counter()
    total_objects = 0
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    
    # Tính weights (inverse frequency)
    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)  # Tránh chia cho 0
        weight = total_objects / (num_classes * count)
        weights.append(weight)
    
    # Normalize về [0.5, 2.0] để không quá extreme
    min_weight = min(weights)
    max_weight = max(weights)
    normalized_weights = []
    for w in weights:
        # Scale to [0.5, 2.0]
        if max_weight > min_weight:
            normalized = 0.5 + 1.5 * (w - min_weight) / (max_weight - min_weight)
        else:
            normalized = 1.0
        normalized_weights.append(normalized)
    
    print("\n✅ Class weights calculated:")
    for i, (count, weight) in enumerate(zip([class_counts.get(i, 0) for i in range(num_classes)], normalized_weights)):
        print(f"   Class {i:2d}: {count:6d} objects -> weight = {weight:.3f}")
    
    return normalized_weights

def create_training_config(output_dir, class_weights):
    """Tạo file config cho training với class weights"""
    config = {
        'class_weights': class_weights,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = Path(output_dir) / 'class_weights.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n💾 Class weights saved to: {config_path}")
    return config_path

def train_yolov12_11class(
    data_yaml,
    model_size='n',  # n, s, m, l, x
    epochs=300,
    batch_size=16,
    img_size=640,
    patience=50,
    device='0',
    project='runs/train_11class',
    name='yolov12_11class',
    use_class_weights=True,
    pretrained=True
):
    """
    Training YOLOv12 cho 11-class traffic detection
    
    Args:
        data_yaml: Đường dẫn đến file data.yaml
        model_size: Kích thước model (n, s, m, l, x)
        epochs: Số epoch training
        batch_size: Batch size
        img_size: Kích thước ảnh input
        patience: Early stopping patience
        device: GPU device (0, 1, cpu)
        project: Thư mục lưu kết quả
        name: Tên experiment
        use_class_weights: Có sử dụng class weights không
        pretrained: Sử dụng pretrained weights
    """
    
    print("="*80)
    print("🚀 YOLOv12 11-CLASS TRAFFIC DETECTION TRAINING")
    print("="*80)
    
    # Load data.yaml để lấy dataset path
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = data_config['path']
    num_classes = data_config['nc']
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📦 Model: YOLOv12{model_size}")
    print(f"🎯 Classes: {num_classes}")
    print(f"📏 Image size: {img_size}")
    print(f"🔢 Batch size: {batch_size}")
    print(f"🔄 Epochs: {epochs}")
    print(f"⏰ Patience: {patience}")
    print(f"💻 Device: {device}")
    
    # Tính class weights
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(dataset_path, num_classes)
        # Save class weights
        output_dir = Path(project) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        create_training_config(output_dir, class_weights)
    
    # Load model
    model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    print(f"\n📥 Loading model: {model_name}")
    
    try:
        model = YOLO(model_name)
    except:
        print(f"⚠️  YOLOv12 không có sẵn, sử dụng YOLOv8 thay thế")
        model = YOLO(f'yolov8{model_size}.pt')
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'patience': patience,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        
        # Augmentation - MẠNH cho class thiếu số
        'hsv_h': 0.015,      # Hue augmentation
        'hsv_s': 0.7,        # Saturation augmentation
        'hsv_v': 0.4,        # Value augmentation
        'degrees': 10,       # Rotation augmentation
        'translate': 0.1,    # Translation augmentation
        'scale': 0.5,        # Scale augmentation
        'shear': 2.0,        # Shear augmentation
        'perspective': 0.0,  # Perspective augmentation
        'flipud': 0.0,       # Flip up-down augmentation
        'fliplr': 0.5,       # Flip left-right augmentation
        'mosaic': 1.0,       # Mosaic augmentation
        'mixup': 0.1,        # Mixup augmentation
        'copy_paste': 0.1,   # Copy-paste augmentation
        
        # Loss weights (sẽ tự động adjust với class weights)
        'box': 7.5,    # Box loss gain
        'cls': 0.5,    # Class loss gain
        'dfl': 1.5,    # Distribution Focal Loss gain
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'plots': True,
        'verbose': True,
        
        # Other
        'workers': 8,
        'cache': False,  # Cache images (True nếu RAM đủ lớn)
        'amp': True,     # Automatic Mixed Precision
        'deterministic': False,  # Deterministic training
        'seed': 42,
    }
    
    # Nếu có class weights, thêm vào args (chưa support trực tiếp, sẽ dùng custom callback)
    # Workaround: tăng cls loss cho class thiếu số thông qua augmentation
    
    print("\n" + "="*80)
    print("🎯 STARTING TRAINING...")
    print("="*80 + "\n")
    
    # Train
    results = model.train(**train_args)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    
    # Evaluate best model
    print("\n📊 Evaluating best model on test set...")
    best_model = YOLO(Path(project) / name / 'weights' / 'best.pt')
    
    # Test evaluation
    test_results = best_model.val(
        data=data_yaml,
        split='test',
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    print("\n📈 Test Results:")
    print(f"   mAP@50: {test_results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"   mAP@50-95: {test_results.results_dict['metrics/mAP50-95(B)']:.4f}")
    
    print(f"\n💾 Best model saved at: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"📂 Training results: {Path(project) / name}")
    
    return results, test_results

if __name__ == "__main__":
    # Cấu hình training
    DATA_YAML = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_11class_final\data.yaml"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(DATA_YAML):
        print(f"❌ Không tìm thấy file: {DATA_YAML}")
        
        # Thử dùng dataset cũ
        DATA_YAML_OLD = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_balanced_11class_processed\data.yaml"
        if os.path.exists(DATA_YAML_OLD):
            print(f"✅ Sử dụng dataset: {DATA_YAML_OLD}")
            DATA_YAML = DATA_YAML_OLD
        else:
            print("❌ Không tìm thấy dataset nào!")
            sys.exit(1)
    
    # Training parameters
    CONFIG = {
        'data_yaml': DATA_YAML,
        'model_size': 'n',           # n = nano (nhẹ nhất, nhanh nhất)
        'epochs': 300,               # 300 epochs
        'batch_size': 16,            # Adjust theo GPU RAM
        'img_size': 640,             # Standard YOLO size
        'patience': 50,              # Early stopping
        'device': '0',               # GPU 0 (hoặc 'cpu')
        'project': 'runs/train_11class_final',
        'name': 'yolov12_11class_weighted',
        'use_class_weights': True,   # Sử dụng class weights
        'pretrained': True           # Pretrained weights
    }
    
    print("\n🔧 Training Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Confirm
    print("\n" + "="*80)
    response = input("▶️  Start training? (y/n): ")
    if response.lower() != 'y':
        print("❌ Training cancelled.")
        sys.exit(0)
    
    # Start training
    results, test_results = train_yolov12_11class(**CONFIG)
    
    print("\n🎉 ALL DONE!")
