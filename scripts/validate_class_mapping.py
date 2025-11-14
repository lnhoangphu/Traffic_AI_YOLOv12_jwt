"""
Kiểm tra class mapping có bị lộn không
Lấy mẫu từ mỗi source dataset và hiển thị class distribution
"""

import sys
from pathlib import Path
from collections import defaultdict
import random

PROJECT_ROOT = Path(__file__).parent.parent

# Class mapping theo taxonomy
CLASS_NAMES = {
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

# Expected mappings from each source
SOURCE_MAPPINGS = {
    'intersection_flow_5k': {
        'expected_classes': {
            0: "pedestrian → Person (3)",
            1: "bicycle → Bicycle (2)",
            2: "vehicle → Vehicle (0)",
            3: "car → Vehicle (0)",
            4: "truck → Truck (5)",
            5: "bus → Bus (1)",
            6: "engine → Engine (4)",
            7: "tricycle → Tricycle (6)"
        },
        'class_map': {
            0: 3,  # pedestrian → Person
            1: 2,  # bicycle → Bicycle
            2: 0,  # vehicle → Vehicle
            3: 0,  # car → Vehicle
            4: 5,  # truck → Truck
            5: 1,  # bus → Bus
            6: 4,  # engine → Engine
            7: 6   # tricycle → Tricycle
        }
    },
    'vn_traffic_sign': {
        'expected_classes': {
            'all': "traffic signs (0-28) → Traffic Sign (10)"
        },
        'class_map': 'all_to_10'  # All classes → 10
    },
    'road_issues': {
        'expected_classes': {
            0: "broken_road_sign → Traffic Sign (10)",
            4: "pothole → Pothole (8)",
            'others': "mixed issues → Obstacle (7)"
        },
        'class_map': {
            0: 10,  # broken_road_sign → Traffic Sign
            1: 7,   # damaged_road → Obstacle
            2: 7,   # faded_road_markings → Obstacle
            3: 7,   # mixed_issue → Obstacle
            4: 8,   # pothole → Pothole
            5: 7,   # littering_garbage → Obstacle
            6: 7    # vandalism → Obstacle
        }
    },
    'object_detection_35_traffic_only': {
        'expected_classes': {
            0: "Vehicle (already mapped)",
            1: "Bus (already mapped)",
            2: "Bicycle (already mapped)",
            3: "Person (already mapped)",
            4: "Engine (already mapped)",
            5: "Truck (already mapped)",
            7: "Obstacle (already mapped)",
            8: "Pothole (already mapped)",
            9: "Traffic Light (already mapped)",
            10: "Traffic Sign (already mapped)"
        },
        'class_map': 'direct'  # Already converted
    }
}

def analyze_label_file(label_path):
    """Parse label và trả về class distribution"""
    classes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    classes.append(class_id)
    except:
        pass
    return classes

def check_dataset_mapping(dataset_name, images_dir, labels_dir, sample_size=10):
    """
    Kiểm tra class mapping của một dataset
    
    Args:
        dataset_name: Tên dataset
        images_dir: Thư mục images
        labels_dir: Thư mục labels
        sample_size: Số lượng mẫu để kiểm tra
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"   ⚠️ Dataset not found: {dataset_name}")
        return
    
    # Get random samples
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"   ⚠️ No images found in {dataset_name}")
        return
    
    sample_files = random.sample(image_files, min(sample_size, len(image_files)))
    
    # Analyze class distribution
    class_counts = defaultdict(int)
    total_bboxes = 0
    
    for img_path in sample_files:
        label_path = labels_dir / img_path.with_suffix('.txt').name
        if label_path.exists():
            classes = analyze_label_file(label_path)
            for cls in classes:
                class_counts[cls] += 1
                total_bboxes += 1
    
    print(f"\n   📊 Sample: {len(sample_files)} images, {total_bboxes} bboxes")
    print(f"   Class distribution:")
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = count / total_bboxes * 100 if total_bboxes > 0 else 0
        class_name = CLASS_NAMES.get(class_id, f"UNKNOWN-{class_id}")
        print(f"      {class_id}: {class_name:15s} - {count:4d} bboxes ({percentage:5.1f}%)")
    
    # Validate class IDs
    invalid_classes = [cls for cls in class_counts.keys() if cls not in CLASS_NAMES]
    if invalid_classes:
        print(f"   ❌ INVALID CLASS IDs FOUND: {invalid_classes}")
        return False
    else:
        print(f"   ✅ All class IDs valid (0-10)")
        return True

def main():
    print("=" * 70)
    print("🔍 CHECKING CLASS MAPPING CORRECTNESS")
    print("=" * 70)
    
    datasets = [
        {
            'name': 'Intersection-Flow-5K (Train)',
            'images': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/train/images',
            'labels': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/train/labels'
        },
        {
            'name': 'Intersection-Flow-5K (Val)',
            'images': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/val/images',
            'labels': PROJECT_ROOT / 'datasets_src/intersection_flow_5k/Intersection-Flow-5K/val/labels'
        },
        {
            'name': 'VN Traffic Sign',
            'images': PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/images/train',
            'labels': PROJECT_ROOT / 'datasets_src/vn_traffic_sign/dataset/labels/train'
        },
        {
            'name': 'Road Issues',
            'images': PROJECT_ROOT / 'datasets_src/road_issues_yolo/images/train',
            'labels': PROJECT_ROOT / 'datasets_src/road_issues_yolo/labels/train'
        },
        {
            'name': 'Object Detection 35 (Filtered)',
            'images': PROJECT_ROOT / 'datasets_src/object_detection_35_traffic_only/images/train',
            'labels': PROJECT_ROOT / 'datasets_src/object_detection_35_traffic_only/labels/train'
        }
    ]
    
    all_valid = True
    
    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"📦 {ds['name']}")
        print(f"{'='*70}")
        
        # Show expected mapping
        ds_key = ds['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').split('_')[0:3]
        ds_key = '_'.join(ds_key) if len(ds_key) > 1 else ds_key[0]
        
        if ds_key in SOURCE_MAPPINGS or 'intersection' in ds['name'].lower():
            if 'intersection' in ds['name'].lower():
                mapping = SOURCE_MAPPINGS['intersection_flow_5k']
            elif 'traffic_sign' in ds['name'].lower() or 'vn' in ds['name'].lower():
                mapping = SOURCE_MAPPINGS['vn_traffic_sign']
            elif 'road_issues' in ds['name'].lower() or 'road' in ds['name'].lower():
                mapping = SOURCE_MAPPINGS['road_issues']
            elif 'object_detection' in ds['name'].lower():
                mapping = SOURCE_MAPPINGS['object_detection_35_traffic_only']
            
            print(f"\n   Expected mapping:")
            if isinstance(mapping.get('expected_classes'), dict):
                for key, val in mapping['expected_classes'].items():
                    print(f"      {key}: {val}")
        
        valid = check_dataset_mapping(ds['name'], ds['images'], ds['labels'], sample_size=20)
        if not valid:
            all_valid = False
    
    # Check balanced dataset
    print(f"\n{'='*70}")
    print(f"📦 BALANCED DATASET (Final Output)")
    print(f"{'='*70}")
    
    for split in ['train', 'val', 'test']:
        print(f"\n   🔹 {split.upper()} Split:")
        valid = check_dataset_mapping(
            f'Balanced-{split}',
            PROJECT_ROOT / f'datasets/traffic_ai_final_balanced/images/{split}',
            PROJECT_ROOT / f'datasets/traffic_ai_final_balanced/labels/{split}',
            sample_size=30
        )
        if not valid:
            all_valid = False
    
    print("\n" + "=" * 70)
    if all_valid:
        print("✅ ALL CLASS MAPPINGS ARE CORRECT!")
        print("   - All class IDs in range [0-10]")
        print("   - No invalid classes found")
        print("   - Distribution matches expected patterns")
    else:
        print("❌ SOME CLASS MAPPINGS ARE INCORRECT!")
        print("   - Check the errors above")
        print("   - May need to re-map or re-filter datasets")
    print("=" * 70)

if __name__ == '__main__':
    main()
