"""
Quick Training Script - YOLOv12n 11-Class Traffic Detection
Dùng để test nhanh với ít epochs trước khi train full
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Chưa cài đặt ultralytics. Chạy: pip install ultralytics")
    sys.exit(1)

def calculate_class_weights(dataset_path, num_classes=11):
    """Tính class weights dựa trên inverse frequency"""
    print("\n📊 Calculating class weights...")
    
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
    
    # Tính weights
    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        weight = total_objects / (num_classes * count)
        weights.append(weight)
    
    # Normalize về [0.5, 2.0]
    min_weight = min(weights)
    max_weight = max(weights)
    normalized_weights = []
    for w in weights:
        if max_weight > min_weight:
            normalized = 0.5 + 1.5 * (w - min_weight) / (max_weight - min_weight)
        else:
            normalized = 1.0
        normalized_weights.append(normalized)
    
    print("\n✅ Class weights:")
    class_names = ['Vehicle', 'Bus', 'Bicycle', 'Person', 'Engine', 'Truck', 
                   'Tricycle', 'Obstacle', 'Pothole', 'Traffic Light', 'Traffic Sign']
    for i, (count, weight) in enumerate(zip([class_counts.get(i, 0) for i in range(num_classes)], normalized_weights)):
        print(f"   {i:2d} {class_names[i]:15s}: {count:6d} objects -> weight = {weight:.3f}")
    
    return normalized_weights

def quick_train_yolov12(
    data_yaml,
    epochs=30,           # QUICK: Chỉ 30 epochs để test
    batch_size=16,
    img_size=640,
    device='0',          # Will auto-detect and use 'cpu' if GPU not available
    project='runs/quick_train_11class',
    name='yolov12n_quick_test'
):
    """
    Quick training để test nhanh
    - Chỉ 30 epochs
    - Patience=10
    - YOLOv12n
    """
    
    print("="*80)
    print("🚀 YOLOv12n QUICK TRAINING - 11 CLASS (30 EPOCHS)")
    print("="*80)
    
    # Auto-detect GPU/CPU
    import torch
    if device == '0' and not torch.cuda.is_available():
        print("\n⚠️  GPU not available, switching to CPU mode")
        device = 'cpu'
        print(f"💡 TIP: Training trên CPU sẽ chậm hơn nhiều (~3-4 giờ thay vì 30-45 phút)")
        print(f"💡 Để dùng GPU, cài PyTorch với CUDA:")
        print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Load data.yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = data_config['path']
    num_classes = data_config['nc']
    
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📦 Model: YOLOv12n")
    print(f"🎯 Classes: {num_classes}")
    print(f"🔄 Epochs: {epochs} (QUICK TEST)")
    print(f"📏 Batch: {batch_size}")
    print(f"💻 Device: {device}")
    if device == 'cpu':
        print(f"⚠️  CPU mode: Training sẽ chậm hơn nhiều!")
    
    # Tính class weights
    class_weights = calculate_class_weights(dataset_path, num_classes)
    
    # Save class weights
    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'class_weights': class_weights,
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'Quick test training - 30 epochs'
    }
    config_path = output_dir / 'class_weights.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load YOLOv12n model
    print(f"\n📥 Loading YOLOv12n...")
    
    # Kiểm tra file tồn tại
    yolo12n_path = 'yolo12n.pt'
    if not os.path.exists(yolo12n_path):
        print(f"⚠️  Không tìm thấy {yolo12n_path}")
        print(f"🔍 Tìm kiếm trong thư mục hiện tại...")
        
        # Tìm trong thư mục gốc project
        for potential_path in [
            'yolo12n.pt',
            '../yolo12n.pt',
            '../../yolo12n.pt',
            'd:/DH_K47/nam_tu/HK1/Do_an_2/Traffic_AI_YOLOv12_jwt/yolo12n.pt'
        ]:
            if os.path.exists(potential_path):
                yolo12n_path = potential_path
                print(f"✅ Found: {yolo12n_path}")
                break
    
    try:
        model = YOLO(yolo12n_path)
        print(f"✅ Loaded YOLOv12n successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLOv12n: {e}")
        print(f"⚠️  Fallback to YOLOv8n...")
        model = YOLO('yolov8n.pt')
    
    # Training arguments - QUICK VERSION
    train_args = {
        'data': data_yaml,
        'epochs': epochs,          # QUICK: 30 epochs
        'batch': batch_size,
        'imgsz': img_size,
        'patience': 10,            # QUICK: patience=10
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Augmentation - VỪA PHẢI cho quick test
        'hsv_h': 0.015,
        'hsv_s': 0.5,        # Giảm từ 0.7
        'hsv_v': 0.3,        # Giảm từ 0.4
        'degrees': 5,        # Giảm từ 10
        'translate': 0.1,
        'scale': 0.3,        # Giảm từ 0.5
        'shear': 1.0,        # Giảm từ 2.0
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,       # Giảm từ 1.0
        'mixup': 0.05,       # Giảm từ 0.1
        'copy_paste': 0.0,   # Tắt cho quick test
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 5,    # Save mỗi 5 epochs
        'plots': True,
        'verbose': True,
        
        # Other
        'workers': 4,        # Giảm từ 8
        'cache': False,
        'amp': True,
        'seed': 42,
    }
    
    print("\n" + "="*80)
    print("🎯 STARTING QUICK TRAINING (30 EPOCHS)...")
    print("="*80 + "\n")
    
    # Train
    results = model.train(**train_args)
    
    print("\n" + "="*80)
    print("✅ QUICK TRAINING COMPLETED!")
    print("="*80)
    
    # Quick evaluation
    print("\n📊 Quick evaluation on val set...")
    best_model = YOLO(Path(project) / name / 'weights' / 'best.pt')
    
    val_results = best_model.val(
        data=data_yaml,
        split='val',
        conf=0.25,
        iou=0.7,
        plots=True
    )
    
    print("\n📈 Quick Results:")
    print(f"   mAP@50:    {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP@50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision: {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall:    {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
    
    print(f"\n💾 Model saved: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"📂 Results: {Path(project) / name}")
    
    print("\n💡 TIP: Nếu kết quả tốt, chạy full training với 300 epochs!")
    print("    python training\\train_11class_final.py")
    
    return results, val_results

if __name__ == "__main__":
    # Cấu hình
    DATA_YAML = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_balanced_11class_processed\data.yaml"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(DATA_YAML):
        print(f"❌ Không tìm thấy: {DATA_YAML}")
        sys.exit(1)
    
    # Quick training config
    CONFIG = {
        'data_yaml': DATA_YAML,
        'epochs': 30,              # QUICK: 30 epochs (~30-45 phút)
        'batch_size': 16,          # Có thể tăng lên 32 nếu GPU đủ mạnh
        'img_size': 640,
        'device': '0',             # GPU 0
        'project': 'runs/quick_train_11class',
        'name': 'yolov12n_quick_test'
    }
    
    print("\n🔧 Quick Training Configuration:")
    print("="*80)
    for key, value in CONFIG.items():
        print(f"   {key:15s}: {value}")
    print("="*80)
    
    print("\n⏰ Estimated time: ~30-45 minutes")
    print("💡 Purpose: Quick test để verify pipeline hoạt động")
    print("💡 Sau khi OK, chạy full training (300 epochs)")
    
    # Auto-start without confirmation
    print("\n" + "="*80)
    print("🚀 Starting quick training...\n")
    results, val_results = quick_train_yolov12(**CONFIG)
    
    print("\n🎉 QUICK TEST DONE!")
    print("\n📝 Next steps:")
    print("   1. Check results in: runs/quick_train_11class/yolov12n_quick_test/")
    print("   2. Review confusion matrix")
    print("   3. If OK, run full training: python training\\train_11class_final.py")
