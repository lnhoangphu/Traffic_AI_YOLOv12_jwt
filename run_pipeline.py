"""
Script chính để chạy toàn bộ pipeline Traffic AI YOLOv12.
Từ tải data đến training và evaluation.
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd, description=""):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ THÀNH CÔNG")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Chỉ hiện 500 ký tự cuối
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ LỖI: {e}")
        if e.stdout:
            print("Output:", e.stdout[-500:])
        if e.stderr:
            print("Error:", e.stderr[-500:])
        return False

def check_dependencies():
    """Kiểm tra dependencies cần thiết"""
    print("🔍 Kiểm tra dependencies...")
    
    required_packages = [
        'ultralytics', 'fastapi', 'uvicorn', 'pillow', 
        'kaggle', 'albumentations', 'opencv-python', 
        'pandas', 'matplotlib', 'seaborn', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Thiếu packages: {', '.join(missing)}")
        print("Cài đặt với: pip install " + " ".join(missing))
        return False
    
    print("✅ Tất cả dependencies đã có")
    return True

def check_kaggle_config():
    """Kiểm tra cấu hình Kaggle"""
    REPO_ROOT = Path(__file__).resolve().parent
    kaggle_json = REPO_ROOT / "kaggle.json"
    
    if not kaggle_json.exists():
        print("❌ Chưa có file kaggle.json")
        print("Hướng dẫn:")
        print("1. Đăng nhập kaggle.com")
        print("2. Vào Account → API → Create New API Token")
        print("3. Đặt file kaggle.json vào thư mục gốc project")
        return False
    
    print("✅ File kaggle.json đã có")
    return True

def pipeline_full():
    """Chạy toàn bộ pipeline từ đầu"""
    
    print("🚀 BẮT ĐẦU FULL PIPELINE - TRAFFIC AI YOLOv12")
    
    # 1. Kiểm tra prerequisites
    if not check_dependencies():
        print("❌ Thiếu dependencies - dừng pipeline")
        return False
    
    if not check_kaggle_config():
        print("❌ Chưa cấu hình Kaggle - dừng pipeline")
        return False
    
    # 2. Tải datasets
    print("\n📥 BƯỚC 1: TẢI DATASETS")
    if not run_command("powershell -ExecutionPolicy Bypass -File scripts\\download_kaggle.ps1", 
                      "Tải datasets từ Kaggle"):
        print("⚠️ Lỗi tải data - tiếp tục với data có sẵn")
    
    # 3. Convert datasets
    print("\n🔄 BƯỚC 2: CONVERT DATASETS")
    if not run_command("python scripts\\data_adapters.py", 
                      "Convert datasets sang YOLO format"):
        print("❌ Lỗi convert data - dừng pipeline")
        return False
    
    # 4. Merge datasets
    print("\n📋 BƯỚC 3: HỢP NHẤT DATASETS")
    if not run_command("python scripts\\merge_datasets.py", 
                      "Hợp nhất và chia train/val/test"):
        print("❌ Lỗi merge data - dừng pipeline")
        return False
    
    # 5. Analyze dataset
    print("\n📊 BƯỚC 4: PHÂN TÍCH DATASET")
    run_command("python scripts\\analyze_results.py", 
               "Phân tích distribution và class imbalance")
    
    # 6. Data augmentation
    print("\n🌦️ BƯỚC 5: DATA AUGMENTATION")
    run_command("python scripts\\augment_weather.py", 
               "Tạo synthetic weather data")
    
    # 7. Training
    print("\n🎯 BƯỚC 6: TRAINING YOLOV12")
    if not run_command("python scripts\\train_yolov12.py", 
                      "Training YOLOv12 model"):
        print("❌ Lỗi training - dừng pipeline")
        return False
    
    print("\n🎉 HOÀN THÀNH PIPELINE!")
    print("📂 Kiểm tra thư mục runs/ để xem kết quả training")
    print("📊 Kiểm tra thư mục analysis/ để xem biểu đồ phân tích")
    
    return True

def pipeline_train_only():
    """Chỉ chạy training (data đã sẵn sàng)"""
    
    print("🎯 TRAINING PIPELINE - YOLOV12")
    
    # Kiểm tra data
    REPO_ROOT = Path(__file__).resolve().parent
    data_yaml = REPO_ROOT / "data" / "traffic" / "data.yaml"
    
    if not data_yaml.exists():
        print("❌ Không tìm thấy data.yaml - chạy full pipeline trước")
        return False
    
    # Training
    if not run_command("python scripts\\train_yolov12.py", 
                      "Training YOLOv12 model"):
        return False
    
    # Analysis
    run_command("python scripts\\analyze_results.py", 
               "Phân tích kết quả training")
    
    print("🎉 HOÀN THÀNH TRAINING!")
    return True

def pipeline_test_api():
    """Test API service"""
    
    print("🧪 API TESTING PIPELINE")
    
    print("⚠️ CHÚ Ý: Cần start API service trước khi test")
    print("Command: uvicorn src.ai_service.main:app --host 0.0.0.0 --port 8000")
    
    input("Press Enter để tiếp tục test (hoặc Ctrl+C để hủy)...")
    
    run_command("python scripts\\test_api.py", 
               "Test API endpoints và performance")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Traffic AI YOLOv12 Pipeline")
    parser.add_argument("--mode", choices=["full", "train", "test"], 
                       default="full",
                       help="Chế độ chạy: full (toàn bộ), train (chỉ training), test (test API)")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        success = pipeline_full()
    elif args.mode == "train":
        success = pipeline_train_only()
    elif args.mode == "test":
        success = pipeline_test_api()
    
    if success:
        print("\n✅ PIPELINE HOÀN THÀNH THÀNH CÔNG!")
    else:
        print("\n❌ PIPELINE THẤT BẠI!")
        sys.exit(1)

if __name__ == "__main__":
    main()