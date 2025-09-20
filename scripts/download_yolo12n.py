"""
Download YOLOv12n Model Script

Downloads the official YOLOv12n pretrained model from Ultralytics.
This model is optimized for RTX 3050Ti with 4GB VRAM constraints.

Usage: python scripts/download_yolo12n.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def download_yolo12n():
    """
    Download YOLOv12n model and verify successful download.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("🚀 DOWNLOADING YOLOv12n MODEL")
    print("=" * 50)
    
    try:
        # Initialize YOLO with yolo12n - this will auto-download if not present
        print("📥 Downloading YOLOv12n pretrained model...")
        model = YOLO('yolo12n.pt')
        
        # Check if model file exists
        model_path = Path('yolo12n.pt')
        if model_path.exists():
            model_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"✅ YOLOv12n downloaded successfully!")
            print(f"   📁 File: {model_path}")
            print(f"   📊 Size: {model_size:.1f} MB")
            
            # Test model loading
            print("🧪 Testing model loading...")
            info = model.info()
            print(f"   📋 Model info: {info}")
            print(f"   🎯 Ready for training!")
            
            return True
        else:
            print("❌ Model file not found after download attempt")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Update ultralytics: pip install -U ultralytics")
        print("   3. Try manual download from: https://github.com/ultralytics/assets/releases/")
        return False


def check_gpu_compatibility():
    """
    Check if CUDA and RTX 3050Ti are properly configured.
    
    Returns:
        bool: True if GPU is available and compatible
    """
    print("\n🖥️ CHECKING GPU COMPATIBILITY")
    print("=" * 50)
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ CUDA available!")
            print(f"   🎮 GPU: {gpu_name}")
            print(f"   💾 VRAM: {gpu_memory:.1f} GB")
            
            # Check if it's RTX 3050Ti or compatible
            if "3050" in gpu_name or "RTX" in gpu_name:
                print(f"   🚀 RTX GPU detected - optimized training enabled!")
                
                # Recommend optimal settings
                if gpu_memory < 5.0:  # Less than 5GB
                    print(f"   💡 Recommendations for {gpu_memory:.1f}GB VRAM:")
                    print(f"      - Batch size: 8")
                    print(f"      - Image size: 640")
                    print(f"      - Mixed precision: Enabled")
                else:
                    print(f"   💡 Recommendations for {gpu_memory:.1f}GB VRAM:")
                    print(f"      - Batch size: 16")
                    print(f"      - Image size: 640")
                    print(f"      - Mixed precision: Optional")
            
            return True
        else:
            print("⚠️ CUDA not available - will use CPU training")
            print("💡 For GPU training:")
            print("   1. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("   2. Update GPU drivers")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        print("💡 Install with: pip install torch torchvision")
        return False


def main():
    """Main function to download model and check compatibility."""
    print("🎯 YOLO12N SETUP FOR TRAFFIC OBJECT CLASSIFICATION")
    print("=" * 70)
    
    # Check GPU compatibility first
    gpu_available = check_gpu_compatibility()
    
    # Download YOLOv12n model
    success = download_yolo12n()
    
    if success:
        print(f"\n🎉 SETUP COMPLETE!")
        print("=" * 50)
        print("✅ YOLOv12n model ready")
        if gpu_available:
            print("✅ GPU acceleration available")
        else:
            print("⚠️ CPU training only")
        print("\n📚 NEXT STEPS:")
        print("   1. Run dataset merger: python scripts/merge_datasets_fixed.py")
        print("   2. Start training: python scripts/train_balanced_vs_imbalanced.py")
        print("   3. Or run full pipeline: python scripts/master_pipeline.py")
    else:
        print(f"\n💥 SETUP FAILED!")
        print("Check error messages above and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()