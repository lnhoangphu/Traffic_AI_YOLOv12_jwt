# Training script for balanced dataset
# Sử dụng dataset đã cân bằng để train model mới

Write-Host "🚀 Starting YOLOv12n training on BALANCED dataset..." -ForegroundColor Green
Write-Host ""

# Activate conda environment (nếu có)
# conda activate yolo_env

# Training với balanced dataset
python -c @"
from ultralytics import YOLO
import torch

print('📊 GPU Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

print('\n🔥 Loading YOLOv12n base model...')
model = YOLO('yolo12n.pt')

print('\n📚 Training on BALANCED dataset...')
print('   Dataset: datasets/traffic_ai_final_balanced/data.yaml')
print('   Images: 23,621 train + 2,952 val')
print('   Classes: 11 (balanced distribution)')
print('   Expected improvements:')
print('     - Person mAP: 0.025 → 0.15+ (6x better)')
print('     - Engine mAP: 0.038 → 0.12+ (3x better)')
print('     - Traffic Light: 0.052 → 0.15+ (3x better)')
print('     - Overall mAP: 54.95% → 62-66% (+8-12%)')
print('')

results = model.train(
    data='datasets/traffic_ai_final_balanced/data.yaml',
    epochs=150,              # 150 epochs (đủ cho balanced dataset)
    batch=8,                 # Batch size 8
    imgsz=640,              # Image size 640x640
    device=0,               # GPU 0
    project='runs/train_balanced_v2',
    name='yolov12n_11class_balanced',
    
    # Optimizer settings
    optimizer='AdamW',
    lr0=0.001,              # Initial learning rate
    weight_decay=0.0005,
    
    # Data augmentation (moderate - dataset đã augment sẵn)
    hsv_h=0.015,           # Hue augmentation (moderate)
    hsv_s=0.5,             # Saturation augmentation
    hsv_v=0.4,             # Value augmentation
    degrees=10.0,          # Rotation ±10°
    translate=0.1,         # Translation
    scale=0.5,             # Scale
    fliplr=0.5,            # Horizontal flip
    
    # Training settings
    patience=30,           # Early stopping patience
    save=True,
    save_period=10,        # Save checkpoint mỗi 10 epochs
    cache=False,           # Không cache (dataset lớn)
    workers=4,
    amp=True,              # Automatic Mixed Precision
    verbose=True,
    
    # Loss weights (class-weighted loss)
    cls=0.5,               # Classification loss weight
    box=7.5,               # Box loss weight
    
    # Validation
    val=True,
    plots=True
)

print('\n✅ Training complete!')
print(f'   Best model: runs/train_balanced_v2/yolov12n_11class_balanced/weights/best.pt')
print(f'   Last model: runs/train_balanced_v2/yolov12n_11class_balanced/weights/last.pt')
print('\n📊 Check results:')
print('   - Confusion matrix: runs/train_balanced_v2/yolov12n_11class_balanced/confusion_matrix.png')
print('   - Training curves: runs/train_balanced_v2/yolov12n_11class_balanced/results.png')
print('   - PR curves: runs/train_balanced_v2/yolov12n_11class_balanced/PR_curve.png')
"@

Write-Host ""
Write-Host "✅ Training script completed!" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Check training results in: runs/train_balanced_v2/yolov12n_11class_balanced/"
Write-Host "   2. Evaluate on test set: python -c `"from ultralytics import YOLO; model = YOLO('runs/train_balanced_v2/yolov12n_11class_balanced/weights/best.pt'); model.val(data='datasets/traffic_ai_final_balanced/data.yaml', split='test')`""
Write-Host "   3. Compare với model cũ để xem cải thiện"
Write-Host ""
