"""
Quick check script to verify all source datasets are ready
"""
from pathlib import Path
from collections import defaultdict

def check_dataset(name, images_dir, labels_dir):
    """Check if dataset exists and count images/labels"""
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    print(f"\n📦 Checking: {name}")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    
    if not images_path.exists():
        print(f"   ❌ Images directory not found")
        return False
    
    if not labels_path.exists():
        print(f"   ❌ Labels directory not found")
        return False
    
    # Count files
    image_files = list(images_path.rglob('*.jpg')) + list(images_path.rglob('*.png'))
    label_files = list(labels_path.rglob('*.txt'))
    
    print(f"   ✅ Images: {len(image_files):,}")
    print(f"   ✅ Labels: {len(label_files):,}")
    
    if len(image_files) == 0:
        print(f"   ⚠️  No images found")
        return False
    
    if len(label_files) == 0:
        print(f"   ⚠️  No labels found")
        return False
    
    return True


def main():
    print("="*70)
    print("🔍 DATASET VERIFICATION")
    print("="*70)
    
    PROJECT_ROOT = Path(__file__).parent.parent
    
    datasets = [
        {
            'name': 'Intersection-Flow-5K (Train)',
            'images_dir': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/images/train',
            'labels_dir': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/labels/train'
        },
        {
            'name': 'Intersection-Flow-5K (Val)',
            'images_dir': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/images/val',
            'labels_dir': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/labels/val'
        },
        {
            'name': 'VN Traffic Sign (Train)',
            'images_dir': PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/images/train',
            'labels_dir': PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/labels/train'
        },
        {
            'name': 'Road Issues (Train)',
            'images_dir': PROJECT_ROOT / 'datasets_src/road_issues_yolo/images/train',
            'labels_dir': PROJECT_ROOT / 'datasets_src/road_issues_yolo/labels/train'
        },
        {
            'name': 'Object Detection 35 (Train)',
            'images_dir': PROJECT_ROOT / 'datasets_src/object_detection_35_organized/images/train',
            'labels_dir': PROJECT_ROOT / 'datasets_src/object_detection_35_organized/labels/train'
        }
    ]
    
    results = []
    for ds in datasets:
        result = check_dataset(ds['name'], ds['images_dir'], ds['labels_dir'])
        results.append((ds['name'], result))
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ MISSING"
        print(f"{name:40s} {status}")
        if not result:
            all_ok = False
    
    print("="*70)
    
    if all_ok:
        print("\n✅ All datasets are ready!")
        print("\n🚀 You can now run:")
        print("   python scripts\\create_balanced_dataset.py")
    else:
        print("\n⚠️  Some datasets are missing!")
        print("\n💡 Please download missing datasets first:")
        print("   1. Intersection-Flow-5K: https://github.com/...")
        print("   2. VN Traffic Sign: Kaggle dataset")
        print("   3. Road Issues: Kaggle dataset")
        print("   4. Object Detection 35: Kaggle dataset")
    
    print()


if __name__ == '__main__':
    main()
