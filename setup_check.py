"""
Setup script - Kiểm tra và cài đặt dependencies
"""

import subprocess
import sys
import os

def check_and_install_package(package_name, import_name=None):
    """Kiểm tra và cài package nếu chưa có"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} đã được cài đặt")
        return True
    except ImportError:
        print(f"⚠️  {package_name} chưa được cài đặt")
        print(f"📦 Đang cài đặt {package_name}...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ Đã cài đặt {package_name} thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi cài {package_name}: {e}")
            return False

def check_yolo12n():
    """Kiểm tra file yolo12n.pt"""
    if os.path.exists('yolo12n.pt'):
        print("✅ yolo12n.pt đã tồn tại")
        return True
    else:
        print("⚠️  yolo12n.pt không tìm thấy trong thư mục hiện tại")
        print("💡 Model sẽ tự động download khi training lần đầu")
        return False

def main():
    print("="*80)
    print("🔧 KIỂM TRA VÀ CÀI ĐẶT DEPENDENCIES")
    print("="*80)
    
    # Required packages
    packages = [
        ('ultralytics', 'ultralytics'),
        ('opencv-python', 'cv2'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('PyYAML', 'yaml'),
        ('Pillow', 'PIL'),
        ('numpy', 'numpy'),
    ]
    
    print("\n📦 Checking packages...\n")
    
    all_ok = True
    for package, import_name in packages:
        if not check_and_install_package(package, import_name):
            all_ok = False
    
    print("\n📁 Checking model files...\n")
    check_yolo12n()
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ TẤT CẢ DEPENDENCIES ĐÃ SẴN SÀNG!")
        print("\n🚀 Có thể chạy training:")
        print("   - Quick train:  python training\\quick_train_yolov12.py")
        print("   - Full train:   python training\\train_11class_final.py")
    else:
        print("⚠️  MỘT SỐ PACKAGES CHƯA CÀI ĐẶT THÀNH CÔNG")
        print("💡 Thử cài thủ công: pip install ultralytics opencv-python matplotlib seaborn")
    print("="*80)

if __name__ == "__main__":
    main()
