import torch

print("="*60)
print("🔍 GPU CHECK")
print("="*60)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✅ GPU SẴN SÀNG! Có thể bắt đầu training với GPU!")
else:
    print("\n❌ GPU KHÔNG KHẢ DỤNG!")
    print("💡 Kiểm tra:")
    print("   1. NVIDIA driver đã cài chưa?")
    print("   2. CUDA toolkit compatible?")

print("="*60)
