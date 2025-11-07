"""
Evaluation script cho YOLOv12 11-class traffic model
- Per-class mAP
- Confusion matrix
- Class-wise precision, recall, F1-score
- Sample predictions visualization
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Chưa cài đặt ultralytics. Chạy: pip install ultralytics")
    sys.exit(1)

CLASS_NAMES = {
    0: "Vehicle",
    1: "Bus",
    2: "Bicycle",
    3: "Person",
    4: "Engine",
    5: "Truck",
    6: "Tricycle",
    7: "Obstacle",
    8: "Pothole",
    9: "Traffic Light",
    10: "Traffic Sign"
}

def evaluate_model(model_path, data_yaml, split='test', conf_threshold=0.25, iou_threshold=0.7):
    """
    Evaluate model trên test/val set
    
    Args:
        model_path: Đường dẫn đến model .pt
        data_yaml: Đường dẫn đến data.yaml
        split: 'test', 'val', hoặc 'train'
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold for NMS
    """
    
    print("="*80)
    print(f"📊 EVALUATING MODEL ON {split.upper()} SET")
    print("="*80)
    
    print(f"\n📥 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📂 Dataset: {data_yaml}")
    print(f"🎯 Split: {split}")
    print(f"📏 Confidence threshold: {conf_threshold}")
    print(f"📏 IOU threshold: {iou_threshold}")
    
    # Run validation
    print("\n🔄 Running validation...")
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,
        save_hybrid=True,
        plots=True,
        verbose=True
    )
    
    # Extract metrics
    metrics = results.results_dict
    
    print("\n" + "="*80)
    print("📈 OVERALL METRICS")
    print("="*80)
    
    print(f"\n✅ mAP@50:      {metrics['metrics/mAP50(B)']:.4f}")
    print(f"✅ mAP@50-95:   {metrics['metrics/mAP50-95(B)']:.4f}")
    print(f"✅ Precision:   {metrics['metrics/precision(B)']:.4f}")
    print(f"✅ Recall:      {metrics['metrics/recall(B)']:.4f}")
    
    # Per-class metrics
    print("\n" + "="*80)
    print("📊 PER-CLASS METRICS")
    print("="*80)
    
    # YOLO returns class-wise metrics in results.maps (mAP per class)
    if hasattr(results, 'maps') and results.maps is not None:
        maps = results.maps  # mAP@50-95 per class
        
        print(f"\n{'Class':<15} {'Class ID':>8} {'mAP@50-95':>12} {'Precision':>12} {'Recall':>12}")
        print("-" * 60)
        
        for class_id in sorted(CLASS_NAMES.keys()):
            class_name = CLASS_NAMES[class_id]
            map_val = maps[class_id] if class_id < len(maps) else 0.0
            
            # Get precision and recall per class (if available)
            # Note: YOLO might not expose per-class P/R directly, we show mAP
            print(f"{class_name:<15} {class_id:>8} {map_val:>12.4f}")
    
    # Box metrics (if available)
    if hasattr(results, 'box'):
        box_metrics = results.box
        if hasattr(box_metrics, 'mp'):  # Mean Precision per class
            print("\n📊 Detailed per-class metrics:")
            print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'mAP@50':>12} {'mAP@50-95':>12}")
            print("-" * 70)
            
            for i, class_name in CLASS_NAMES.items():
                if i < len(box_metrics.mp):
                    p = box_metrics.mp[i]
                    r = box_metrics.mr[i] if hasattr(box_metrics, 'mr') else 0
                    map50 = box_metrics.map50[i] if hasattr(box_metrics, 'map50') else 0
                    map = box_metrics.map[i] if hasattr(box_metrics, 'map') else 0
                    print(f"{class_name:<15} {p:>12.4f} {r:>12.4f} {map50:>12.4f} {map:>12.4f}")
    
    # Confusion matrix
    print("\n" + "="*80)
    print("🔍 CONFUSION MATRIX")
    print("="*80)
    
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        conf_matrix = results.confusion_matrix.matrix
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='.0f',
            cmap='Blues',
            xticklabels=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
            yticklabels=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        )
        plt.title('Confusion Matrix - 11 Class Traffic Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        output_dir = Path(model_path).parent.parent
        conf_matrix_path = output_dir / 'confusion_matrix.png'
        plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Confusion matrix saved: {conf_matrix_path}")
        plt.close()
    
    # Check for common errors
    print("\n" + "="*80)
    print("⚠️  POTENTIAL ISSUES")
    print("="*80)
    
    if hasattr(results, 'maps') and results.maps is not None:
        low_map_classes = []
        for i, map_val in enumerate(results.maps):
            if i < len(CLASS_NAMES) and map_val < 0.3:
                low_map_classes.append((CLASS_NAMES[i], map_val))
        
        if low_map_classes:
            print("\n⚠️  Classes with low mAP (< 0.3):")
            for class_name, map_val in low_map_classes:
                print(f"   - {class_name}: mAP = {map_val:.4f}")
            print("\n💡 Suggestions:")
            print("   1. Check if these classes have enough training samples")
            print("   2. Increase augmentation for these classes")
            print("   3. Verify label quality")
            print("   4. Consider collecting more data")
        else:
            print("\n✅ All classes have reasonable performance (mAP >= 0.3)")
    
    return results

def test_on_images(model_path, image_folder, output_folder, conf_threshold=0.25):
    """
    Test model trên folder ảnh thật
    
    Args:
        model_path: Đường dẫn đến model .pt
        image_folder: Thư mục chứa ảnh test
        output_folder: Thư mục lưu kết quả
        conf_threshold: Confidence threshold
    """
    
    print("\n" + "="*80)
    print("🖼️  TESTING ON REAL IMAGES")
    print("="*80)
    
    model = YOLO(model_path)
    
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
    
    print(f"\n📁 Found {len(image_files)} images in {image_folder}")
    
    if len(image_files) == 0:
        print("⚠️  No images found!")
        return
    
    # Predict
    print("\n🔄 Running predictions...")
    results = model.predict(
        source=str(image_folder),
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_folder),
        name='predictions',
        exist_ok=True
    )
    
    # Analyze predictions
    print("\n📊 Prediction summary:")
    class_detections = defaultdict(int)
    total_detections = 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                class_detections[class_id] += 1
                total_detections += 1
    
    print(f"\nTotal detections: {total_detections}")
    print(f"\nDetections per class:")
    for class_id in sorted(CLASS_NAMES.keys()):
        count = class_detections.get(class_id, 0)
        if count > 0:
            print(f"   {CLASS_NAMES[class_id]:<15}: {count:4d} detections")
    
    print(f"\n✅ Results saved to: {output_folder / 'predictions'}")

if __name__ == "__main__":
    # Cấu hình
    MODEL_PATH = r"runs/train_11class_final/yolov12_11class_weighted/weights/best.pt"
    DATA_YAML = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_11class_final\data.yaml"
    
    # Fallback nếu không tìm thấy
    if not os.path.exists(DATA_YAML):
        DATA_YAML = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_balanced_11class_processed\data.yaml"
    
    # Evaluate trên test set
    if os.path.exists(MODEL_PATH):
        print(f"✅ Found model: {MODEL_PATH}\n")
        
        # Evaluate
        results = evaluate_model(
            model_path=MODEL_PATH,
            data_yaml=DATA_YAML,
            split='test',
            conf_threshold=0.25,
            iou_threshold=0.7
        )
        
        # Test trên ảnh thật (nếu có)
        test_image_folder = r"test_images"  # Thay đổi path này
        if os.path.exists(test_image_folder):
            test_on_images(
                model_path=MODEL_PATH,
                image_folder=test_image_folder,
                output_folder="runs/test_predictions",
                conf_threshold=0.25
            )
    else:
        print(f"❌ Model not found: {MODEL_PATH}")
        print("\n💡 Please train the model first using train_11class_final.py")
