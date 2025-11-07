"""
MASTER SCRIPT - 11 Class Traffic Detection Pipeline
Chạy toàn bộ quy trình từ đầu đến cuối
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """In header đẹp"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def run_step(step_name, script_path, description):
    """Chạy một bước trong pipeline"""
    print_header(f"STEP: {step_name}")
    print(f"📝 {description}")
    print(f"🚀 Running: {script_path}\n")
    
    response = input("▶️  Continue? (y/n/skip): ").lower()
    
    if response == 'skip':
        print("⏭️  Skipped.\n")
        return 'skip'
    elif response != 'y':
        print("❌ Pipeline stopped.\n")
        sys.exit(0)
    
    # Run script
    result = os.system(f'python {script_path}')
    
    if result != 0:
        print(f"\n❌ Error occurred in {step_name}")
        retry = input("⚠️  Retry? (y/n): ").lower()
        if retry == 'y':
            return run_step(step_name, script_path, description)
        else:
            print("❌ Pipeline stopped.")
            sys.exit(1)
    
    print(f"\n✅ {step_name} completed successfully!")
    return 'success'

def main():
    print_header("🚀 YOLOv12 11-CLASS TRAFFIC DETECTION - FULL PIPELINE")
    
    print("📋 Pipeline bao gồm các bước:")
    print("   1. Phân tích dataset hiện tại")
    print("   2. Training model với class weights")
    print("   3. Evaluation trên test set")
    print()
    
    project_root = Path(__file__).parent
    
    # Step 1: Analyze Dataset
    analyze_script = project_root / "scripts" / "analyze_11class_dataset.py"
    if analyze_script.exists():
        result = run_step(
            "1. Dataset Analysis",
            str(analyze_script),
            "Phân tích class distribution, kiểm tra labels hợp lệ"
        )
    else:
        print(f"⚠️  Script not found: {analyze_script}")
        print("⏭️  Skipping dataset analysis...\n")
    
    # Step 2: Training
    train_script = project_root / "training" / "train_11class_final.py"
    if train_script.exists():
        result = run_step(
            "2. Model Training",
            str(train_script),
            "Training YOLOv12 với class weights và augmentation mạnh (300 epochs)"
        )
        
        if result == 'skip':
            print("⚠️  Training skipped. Evaluation có thể không chạy được nếu chưa có model.")
    else:
        print(f"❌ Training script not found: {train_script}")
        sys.exit(1)
    
    # Step 3: Evaluation
    eval_script = project_root / "training" / "evaluate_11class.py"
    if eval_script.exists():
        # Check if model exists
        model_path = project_root / "runs" / "train_11class_final" / "yolov12_11class_weighted" / "weights" / "best.pt"
        
        if model_path.exists():
            result = run_step(
                "3. Model Evaluation",
                str(eval_script),
                "Đánh giá model: mAP, confusion matrix, per-class metrics"
            )
        else:
            print("⚠️  Model not found. Skipping evaluation.")
            print(f"   Expected: {model_path}")
            print("   💡 Train model first!\n")
    else:
        print(f"⚠️  Evaluation script not found: {eval_script}\n")
    
    # Done
    print_header("🎉 PIPELINE COMPLETED!")
    
    print("📂 Kết quả:")
    print(f"   - Dataset analysis: dataset_11class_analysis.txt")
    print(f"   - Training results: runs/train_11class_final/yolov12_11class_weighted/")
    print(f"   - Model weights: runs/train_11class_final/yolov12_11class_weighted/weights/best.pt")
    print(f"   - Confusion matrix: runs/train_11class_final/yolov12_11class_weighted/confusion_matrix.png")
    print()
    
    print("📖 Xem hướng dẫn chi tiết tại: TRAINING_GUIDE_11CLASS.md")
    print()
    
    print("🚀 Next steps:")
    print("   1. Review training metrics (loss, mAP)")
    print("   2. Check confusion matrix")
    print("   3. Test trên ảnh thật")
    print("   4. Deploy model nếu kết quả tốt")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
