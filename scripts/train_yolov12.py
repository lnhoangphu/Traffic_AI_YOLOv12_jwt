"""
Training script cho YOLOv12 với dataset giao thông đã chuẩn bị.
Bao gồm hyperparameter tuning và model evaluation.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_yolov12():
    """Train YOLOv12 model với dataset giao thông"""
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_YAML = REPO_ROOT / "data" / "traffic" / "data.yaml"
    
    if not DATA_YAML.exists():
        print(f"Không tìm thấy file cấu hình: {DATA_YAML}")
        print("Hãy chạy script merge_datasets.py trước.")
        return None
    
    print(f"=== TRAINING YOLOv12 ===")
    print(f"Data config: {DATA_YAML}")
    
    # Load pretrained YOLOv12n model
    model = YOLO('yolov12n.pt')
    
    # Training với hyperparameters tối ưu cho traffic detection
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,          # Có thể tăng lên 200-300 cho kết quả tốt hơn
        imgsz=640,          # Kích thước ảnh input
        batch=16,           # Batch size - điều chỉnh theo GPU memory
        device='cpu',       # Hoặc 'cuda' nếu có GPU
        patience=20,        # Early stopping patience
        save_period=10,     # Save checkpoint mỗi 10 epochs
        
        # Hyperparameters cho traffic detection
        lr0=0.01,           # Initial learning rate
        lrf=0.1,            # Final learning rate (lr0 * lrf)
        momentum=0.937,     # SGD momentum
        weight_decay=0.0005, # Optimizer weight decay
        warmup_epochs=3,    # Warmup epochs
        warmup_momentum=0.8, # Warmup initial momentum
        
        # Data augmentation
        hsv_h=0.015,        # Hue augmentation (fraction of color circle)
        hsv_s=0.7,          # Saturation augmentation (fraction)
        hsv_v=0.4,          # Value (brightness) augmentation (fraction)
        degrees=10,         # Rotation augmentation (degrees)
        translate=0.1,      # Translation augmentation (fraction)
        scale=0.5,          # Scale augmentation (fraction)
        shear=2,            # Shear augmentation (degrees)
        perspective=0.0,    # Perspective augmentation (probability)
        flipud=0.0,         # Vertical flip augmentation (probability)
        fliplr=0.5,         # Horizontal flip augmentation (probability)
        mosaic=1.0,         # Mosaic augmentation (probability)
        mixup=0.1,          # Mixup augmentation (probability)
        
        # Validation settings
        val=True,           # Validate during training
        plots=True,         # Save training plots
        save=True,          # Save checkpoints
        
        # Project organization
        project='runs/detect', # Project name
        name='traffic_yolov12', # Experiment name
        exist_ok=True,      # Overwrite existing experiment
    )
    
    print(f"Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    return results

def evaluate_model(model_path: str = None):
    """Đánh giá model trên test set"""
    
    if model_path is None:
        # Tìm best model từ training mới nhất
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            latest_exp = max(runs_dir.glob("traffic_yolov12*"), key=os.path.getctime, default=None)
            if latest_exp:
                model_path = latest_exp / "weights" / "best.pt"
    
    if not model_path or not Path(model_path).exists():
        print("Không tìm thấy model để evaluate")
        return None
    
    print(f"=== EVALUATING MODEL ===")
    print(f"Model: {model_path}")
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DATA_YAML = REPO_ROOT / "data" / "traffic" / "data.yaml"
    
    # Load trained model
    model = YOLO(str(model_path))
    
    # Evaluate on test set
    results = model.val(
        data=str(DATA_YAML),
        split='test',        # Sử dụng test set
        imgsz=640,
        batch=8,
        device='cpu',        # Hoặc 'cuda'
        plots=True,          # Tạo confusion matrix, P-R curves, etc.
        save_json=True,      # Lưu kết quả JSON
    )
    
    print("=== EVALUATION RESULTS ===")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    # In kết quả per-class
    if hasattr(results.box, 'maps'):
        print("\nPer-class mAP50:")
        with open(DATA_YAML) as f:
            data_config = yaml.safe_load(f)
            
        for i, map_val in enumerate(results.box.maps):
            class_name = data_config['names'][i]
            print(f"  {class_name}: {map_val:.3f}")
    
    return results

def predict_sample():
    """Test prediction trên một số ảnh mẫu"""
    
    # Tìm best model
    runs_dir = Path("runs/detect")
    if runs_dir.exists():
        latest_exp = max(runs_dir.glob("traffic_yolov12*"), key=os.path.getctime, default=None)
        if latest_exp:
            model_path = latest_exp / "weights" / "best.pt"
            
            if model_path.exists():
                print(f"=== TESTING PREDICTIONS ===")
                print(f"Model: {model_path}")
                
                model = YOLO(str(model_path))
                
                # Test trên test images
                REPO_ROOT = Path(__file__).resolve().parents[1]
                test_img_dir = REPO_ROOT / "data" / "traffic" / "images" / "test"
                
                if test_img_dir.exists():
                    test_images = list(test_img_dir.glob("*.jpg"))[:5]  # Test 5 ảnh đầu
                    
                    for img_path in test_images:
                        print(f"Predicting: {img_path.name}")
                        results = model.predict(
                            source=str(img_path),
                            save=True,
                            conf=0.25,  # Confidence threshold
                            project='runs/predict',
                            name='traffic_test',
                            exist_ok=True
                        )
                        
                        # In kết quả detection
                        for r in results:
                            if len(r.boxes) > 0:
                                print(f"  Detected {len(r.boxes)} objects")
                                for box in r.boxes:
                                    class_id = int(box.cls)
                                    conf = float(box.conf)
                                    class_name = r.names[class_id]
                                    print(f"    {class_name}: {conf:.2f}")
                            else:
                                print("  No objects detected")

def main():
    """Main training pipeline"""
    
    print("=== YOLOV12 TRAFFIC DETECTION TRAINING ===")
    
    # 1. Train model
    print("\n1. Training model...")
    train_results = train_yolov12()
    
    if train_results is None:
        print("Training failed!")
        return
    
    # 2. Evaluate model
    print("\n2. Evaluating model...")
    eval_results = evaluate_model()
    
    # 3. Test predictions
    print("\n3. Testing predictions...")
    predict_sample()
    
    print("\n=== TRAINING PIPELINE COMPLETED ===")
    print("Kiểm tra thư mục runs/ để xem kết quả chi tiết")

if __name__ == "__main__":
    main()