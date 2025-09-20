#!/usr/bin/env python3
"""
Quick YOLOv12n Status Checker

Kiểm tra nhanh trạng thái cài đặt YOLOv12n model.
"""

import sys
from pathlib import Path

def quick_check():
    """Kiểm tra nhanh YOLOv12n"""
    print("🔍 Quick YOLOv12n Check")
    print("=" * 30)
    
    # 1. Check model file
    model_file = Path("yolo12n.pt")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"✅ Model file: {model_file} ({size_mb:.1f}MB)")
    else:
        print(f"❌ Model file not found: {model_file}")
        return False
    
    # 2. Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        else:
            print("⚠️ CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # 3. Check ultralytics
    try:
        from ultralytics import YOLO
        model = YOLO('yolo12n.pt')
        print("✅ YOLOv12n model loaded successfully")
        
        # Get model info
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"   📊 Parameters: {total_params:,}")
        
    except ImportError:
        print("❌ ultralytics not installed")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # 4. Check dataset
    dataset_path = Path("datasets/traffic_ai")
    if dataset_path.exists():
        print(f"✅ Dataset ready: {dataset_path}")
    else:
        print(f"⚠️ Dataset not found: {dataset_path}")
    
    print("\n🎉 YOLOv12n is ready for training!")
    return True

if __name__ == "__main__":
    try:
        success = quick_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️ Check interrupted")
        sys.exit(1)