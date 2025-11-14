"""
Filter Object Detection 35 dataset - CHỈ GIỮ TRAFFIC CLASSES
Loại bỏ hoàn toàn các ảnh chỉ chứa đồ vật không liên quan (chair, fork, spoon, etc.)
"""

import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Traffic-related classes trong Object Detection 35 (35 classes)
TRAFFIC_CLASSES = {
    0,   # Person
    10,  # Bus
    12,  # Bicycle
    14,  # Truck
    15,  # Motorcycles
    20,  # Traffic Light
    23,  # Stop Sign
    24,  # Car
    25,  # Barriers
    26,  # Path Holes
    28   # Train (optional)
}

# Mapping sang 11-class taxonomy
CLASS_MAP = {
    0: 3,   # Person -> Person
    10: 1,  # Bus -> Bus
    12: 2,  # Bicycle -> Bicycle
    14: 5,  # Truck -> Truck
    15: 4,  # Motorcycles -> Engine
    20: 9,  # Traffic Light -> Traffic Light
    23: 10, # Stop Sign -> Traffic Sign
    24: 0,  # Car -> Vehicle
    25: 7,  # Barriers -> Obstacle
    26: 8,  # Path Holes -> Pothole
    28: 0   # Train -> Vehicle (optional)
}

TARGET_CLASSES = {
    0: "Vehicle",
    1: "Bus",
    2: "Bicycle",
    3: "Person",
    4: "Engine",
    5: "Truck",
    6: "Tricycle",
    7: "Obstacle",
    8: "Pothole",
    9: "Traffic Light",
    10: "Traffic Sign"
}

def parse_label(label_path):
    """Parse YOLO label và trả về list (class_id, bbox)"""
    annotations = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height]
                    })
    except Exception as e:
        print(f"Error parsing {label_path}: {e}")
    
    return annotations

def filter_and_convert_label(label_path, output_path):
    """
    Lọc label file - chỉ giữ traffic classes và convert sang 11-class IDs
    
    Returns:
        int: Số lượng traffic bboxes sau khi filter
    """
    annotations = parse_label(label_path)
    
    # Lọc chỉ giữ traffic classes và convert IDs
    traffic_annotations = []
    for ann in annotations:
        if ann['class_id'] in TRAFFIC_CLASSES:
            # Convert sang target class ID
            target_class_id = CLASS_MAP.get(ann['class_id'])
            if target_class_id is not None:
                traffic_annotations.append({
                    'class_id': target_class_id,
                    'bbox': ann['bbox']
                })
    
    # Chỉ ghi file nếu có traffic bboxes
    if traffic_annotations:
        with open(output_path, 'w') as f:
            for ann in traffic_annotations:
                bbox = ann['bbox']
                f.write(f"{ann['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    return len(traffic_annotations)

def filter_dataset(source_root, output_root):
    """
    Filter toàn bộ dataset - chỉ giữ ảnh có traffic objects
    
    Args:
        source_root: datasets_src/object_detection_35_organized
        output_root: datasets_src/object_detection_35_traffic_only
    """
    source_root = Path(source_root)
    output_root = Path(output_root)
    
    stats = {
        'total_images': 0,
        'kept_images': 0,
        'removed_images': 0,
        'total_bboxes_before': 0,
        'total_bboxes_after': 0,
        'class_distribution': defaultdict(int)
    }
    
    print("=" * 70)
    print("🔍 FILTERING OBJECT DETECTION 35 DATASET")
    print("=" * 70)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"\nTraffic classes to keep: {len(TRAFFIC_CLASSES)}")
    print(f"Target 11-class taxonomy: {len(TARGET_CLASSES)}")
    print()
    
    for split in ['train', 'val', 'test']:
        print(f"\n📦 Processing {split.upper()} split...")
        
        source_images_dir = source_root / 'images' / split
        source_labels_dir = source_root / 'labels' / split
        
        output_images_dir = output_root / 'images' / split
        output_labels_dir = output_root / 'labels' / split
        
        # Create output directories
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(source_images_dir.glob('*.jpg')) + list(source_images_dir.glob('*.png'))
        print(f"   Found {len(image_files)} images")
        
        kept = 0
        removed = 0
        
        for img_path in tqdm(image_files, desc=f"   Filtering {split}", leave=False):
            stats['total_images'] += 1
            
            # Find label file
            label_path = source_labels_dir / img_path.with_suffix('.txt').name
            
            if not label_path.exists():
                removed += 1
                stats['removed_images'] += 1
                continue
            
            # Count bboxes before
            original_annotations = parse_label(label_path)
            stats['total_bboxes_before'] += len(original_annotations)
            
            # Filter and convert label
            output_label_path = output_labels_dir / label_path.name
            num_traffic_bboxes = filter_and_convert_label(label_path, output_label_path)
            
            # Chỉ copy image nếu có traffic bboxes
            if num_traffic_bboxes > 0:
                output_img_path = output_images_dir / img_path.name
                shutil.copy2(img_path, output_img_path)
                
                kept += 1
                stats['kept_images'] += 1
                stats['total_bboxes_after'] += num_traffic_bboxes
                
                # Count class distribution
                with open(output_label_path, 'r') as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        stats['class_distribution'][class_id] += 1
            else:
                removed += 1
                stats['removed_images'] += 1
                # Xóa label file nếu đã tạo nhưng rỗng
                if output_label_path.exists():
                    output_label_path.unlink()
        
        print(f"   ✅ Kept: {kept} images")
        print(f"   ❌ Removed: {removed} images (no traffic objects)")
    
    # Create data.yaml
    data_yaml_path = output_root / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {output_root.absolute()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"\n")
        f.write(f"nc: 11\n")
        f.write(f"names:\n")
        for i in range(11):
            f.write(f"  - {TARGET_CLASSES[i]}\n")
    
    print("\n" + "=" * 70)
    print("📊 FILTERING SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {stats['total_images']}")
    print(f"✅ Images kept: {stats['kept_images']} ({stats['kept_images']/stats['total_images']*100:.1f}%)")
    print(f"❌ Images removed: {stats['removed_images']} ({stats['removed_images']/stats['total_images']*100:.1f}%)")
    print(f"\nBounding boxes:")
    print(f"  Before: {stats['total_bboxes_before']}")
    print(f"  After: {stats['total_bboxes_after']}")
    print(f"  Removed: {stats['total_bboxes_before'] - stats['total_bboxes_after']}")
    print(f"\n11-Class Distribution (after filtering):")
    
    total_bboxes = stats['total_bboxes_after']
    for class_id in sorted(stats['class_distribution'].keys()):
        count = stats['class_distribution'][class_id]
        percentage = count / total_bboxes * 100 if total_bboxes > 0 else 0
        print(f"  {class_id}: {TARGET_CLASSES[class_id]:15s} - {count:6d} bboxes ({percentage:5.2f}%)")
    
    print("=" * 70)
    print(f"\n✅ Filtered dataset saved to: {output_root}")
    print(f"✅ data.yaml created: {data_yaml_path}")
    print()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter Object Detection 35 - Only Traffic Classes')
    parser.add_argument(
        '--source',
        type=str,
        default='datasets_src/object_detection_35_organized',
        help='Source dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets_src/object_detection_35_traffic_only',
        help='Output directory for filtered dataset'
    )
    
    args = parser.parse_args()
    
    # Get project root
    PROJECT_ROOT = Path(__file__).parent.parent
    
    source_path = PROJECT_ROOT / args.source
    output_path = PROJECT_ROOT / args.output
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_path}")
        exit(1)
    
    filter_dataset(source_path, output_path)
    
    print("\n🎯 Next steps:")
    print("   1. Verify filtered dataset:")
    print(f"      cd {output_path}")
    print("      Check images and labels in train/val/test")
    print("   2. Update balancing script to use new path:")
    print("      datasets_src/object_detection_35_traffic_only")
    print("   3. Re-run dataset balancing pipeline")
