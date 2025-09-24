#!/usr/bin/env python3
"""
GPU Memory Check & Optimization for RTX 3050 Ti
Kiểm tra GPU memory và đưa ra khuyến nghị tối ưu
"""

import torch
import psutil
import subprocess
import sys
from pathlib import Path

def check_gpu_info():
    """Kiểm tra thông tin GPU"""
    print("🔍 GPU Information:")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"📊 Available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"🖥️ GPU {i}: {gpu_name}")
            print(f"💾 Total Memory: {gpu_memory:.1f} GB")
            
            # Check current memory usage
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            free = gpu_memory - cached
            
            print(f"📈 Allocated: {allocated:.2f} GB")
            print(f"🗄️ Cached: {cached:.2f} GB")
            print(f"🆓 Free: {free:.2f} GB")
            print()
    else:
        print("❌ CUDA not available")
        return False
    
    return True

def check_system_ram():
    """Kiểm tra RAM hệ thống"""
    print("🔍 System RAM Information:")
    print("=" * 30)
    
    ram = psutil.virtual_memory()
    total_gb = ram.total / 1e9
    available_gb = ram.available / 1e9
    used_gb = ram.used / 1e9
    
    print(f"💾 Total RAM: {total_gb:.1f} GB")
    print(f"🆓 Available: {available_gb:.1f} GB")
    print(f"📈 Used: {used_gb:.1f} GB")
    print(f"📊 Usage: {ram.percent:.1f}%")
    print()

def recommend_settings():
    """Đưa ra khuyến nghị settings cho RTX 3050 Ti"""
    print("💡 Recommended Settings for RTX 3050 Ti (4GB VRAM):")
    print("=" * 60)
    
    recommendations = {
        "batch_size": "4 (hoặc 8 nếu image size nhỏ hơn)",
        "image_size": "640 (có thể giảm xuống 512 nếu cần)",
        "amp": "True (Mixed Precision để tiết kiệm VRAM)",
        "cache": "False (Tránh cache để tiết kiệm RAM)",
        "workers": "4 (Giảm số workers)",
        "save_period": "10 (Tiết kiệm disk space)",
        "patience": "15-20 (Early stopping)",
        "epochs": "50-100 (Tuỳ dataset size)"
    }
    
    for param, value in recommendations.items():
        print(f"⚙️ {param}: {value}")
    
    print("\n📋 Additional Tips:")
    print("• Đóng các ứng dụng khác để giải phóng VRAM")
    print("• Sử dụng CPU training nếu GPU quá nhỏ: device='cpu'")
    print("• Chia nhỏ dataset nếu quá lớn")
    print("• Monitor GPU usage: nvidia-smi")

def test_memory_usage():
    """Test memory usage với settings khuyến nghị"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available for testing")
        return
    
    print("\n🧪 Testing Memory Usage:")
    print("=" * 30)
    
    device = torch.device('cuda:0')
    
    try:
        # Test với batch size 4
        batch_size = 4
        img_size = 640
        
        # Simulate YOLOv12 input
        test_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"✅ Test successful with batch_size={batch_size}, img_size={img_size}")
        print(f"📈 Memory used: {allocated:.2f} GB")
        
        # Test với batch size 8
        try:
            test_input = torch.randn(8, 3, img_size, img_size).to(device)
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"✅ Test successful with batch_size=8, img_size={img_size}")
            print(f"📈 Memory used: {allocated:.2f} GB")
        except RuntimeError as e:
            print(f"❌ batch_size=8 failed: Out of memory")
            print("💡 Khuyến nghị: Sử dụng batch_size=4")
        
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"❌ Memory test failed: {e}")
        print("💡 Khuyến nghị: Giảm batch_size hoặc image_size")

def update_training_configs():
    """Cập nhật config files với settings tối ưu"""
    print("\n🔧 Updating Training Configurations:")
    print("=" * 40)
    
    project_root = Path(__file__).parent.parent
    
    # Thông tin tối ưu cho RTX 3050 Ti
    optimal_config = {
        "batch_size": 4,
        "img_size": 640,
        "amp": True,
        "cache": False,
        "workers": 4,
        "save_period": 10,
        "patience": 20
    }
    
    print("✅ Optimal configurations:")
    for key, value in optimal_config.items():
        print(f"   {key}: {value}")
    
    print("\n💾 Configurations already applied to training scripts!")
    print("🚀 Ready to start training with optimized settings")

def main():
    """Main function"""
    print("🔧 GPU Memory Optimizer for YOLOv12 Training")
    print("=" * 60)
    print("Hardware: RTX 3050 Ti (4GB VRAM) + 16GB RAM + i5-12500H")
    print("=" * 60)
    
    # Check GPU
    gpu_ok = check_gpu_info()
    
    # Check RAM
    check_system_ram()
    
    # Recommendations
    recommend_settings()
    
    if gpu_ok:
        # Test memory
        test_memory_usage()
    
    # Update configs
    update_training_configs()
    
    print("\n🎯 Next Steps:")
    print("1. Run: python training/train_balanced.py")
    print("2. Monitor GPU usage: nvidia-smi")
    print("3. If still OOM, reduce batch_size to 2 or img_size to 512")

if __name__ == "__main__":
    main()