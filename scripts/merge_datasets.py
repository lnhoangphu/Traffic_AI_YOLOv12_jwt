"""
Script hợp nhất tất cả datasets đã convert và chia thành train/val/test sets.
Đảm bảo cân bằng classes và tạo cấu trúc thư mục YOLO chuẩn.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd

def merge_datasets():
    """Hợp nhất tất cả datasets đã convert thành một dataset thống nhất"""
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    CONVERTED_ROOT = REPO_ROOT / "data" / "traffic_converted"
    TARGET_ROOT = REPO_ROOT / "data" / "traffic"
    
    # Tạo cấu trúc thư mục YOLO
    for split in ['train', 'val', 'test']:
        (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Thu thập tất cả ảnh và label từ các dataset đã convert
    all_images = []
    all_labels = []
    
    for dataset_dir in CONVERTED_ROOT.iterdir():
        if dataset_dir.is_dir():
            img_dir = dataset_dir / "images"
            lbl_dir = dataset_dir / "labels"
            
            if img_dir.exists() and lbl_dir.exists():
                for img_file in img_dir.glob("*.*"):
                    label_file = lbl_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        all_images.append(img_file)
                        all_labels.append(label_file)
    
    print(f"Tổng cộng tìm thấy {len(all_images)} cặp ảnh-label")
    
    # Phân tích phân phối classes
    class_distribution = defaultdict(int)
    valid_pairs = []
    
    for img_file, label_file in zip(all_images, all_labels):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        if lines:  # Có ít nhất 1 object
            for line in lines:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_distribution[class_id] += 1
            valid_pairs.append((img_file, label_file))
        else:  # Negative sample (background)
            class_distribution[-1] += 1  # Background class
            valid_pairs.append((img_file, label_file))
    
    print("Phân phối classes:")
    for class_id, count in sorted(class_distribution.items()):
        if class_id == -1:
            print(f"  Background: {count}")
        else:
            from scripts.data_adapters import CLASS_MAPPING
            class_name = CLASS_MAPPING.get(class_id, f"Unknown_{class_id}")
            print(f"  {class_id}: {class_name} - {count} instances")
    
    # Shuffle và chia dataset
    random.seed(42)
    random.shuffle(valid_pairs)
    
    total = len(valid_pairs)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    
    train_pairs = valid_pairs[:train_size]
    val_pairs = valid_pairs[train_size:train_size + val_size]
    test_pairs = valid_pairs[train_size + val_size:]
    
    print(f"Chia dataset: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Copy files vào các thư mục tương ứng
    def copy_pairs(pairs, split_name):
        for i, (img_file, label_file) in enumerate(pairs):
            # Tạo tên file unique
            new_name = f"{split_name}_{i:06d}{img_file.suffix}"
            
            # Copy image
            target_img = TARGET_ROOT / "images" / split_name / new_name
            shutil.copy2(img_file, target_img)
            
            # Copy label  
            target_label = TARGET_ROOT / "labels" / split_name / f"{Path(new_name).stem}.txt"
            shutil.copy2(label_file, target_label)
    
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val") 
    copy_pairs(test_pairs, "test")
    
    print(f"Đã hoàn thành hợp nhất dataset tại {TARGET_ROOT}")
    return TARGET_ROOT

def analyze_class_balance(data_root: Path):
    """Phân tích sự cân bằng của classes trong dataset"""
    
    print("\n=== PHÂN TÍCH CÂN BẰNG CLASSES ===")
    
    from scripts.data_adapters import CLASS_MAPPING
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} SET:")
        
        label_dir = data_root / "labels" / split
        if not label_dir.exists():
            continue
            
        class_counts = defaultdict(int)
        total_files = 0
        
        for label_file in label_dir.glob("*.txt"):
            total_files += 1
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            if not lines or not any(line.strip() for line in lines):
                class_counts['background'] += 1
            else:
                for line in lines:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_name = CLASS_MAPPING.get(class_id, f"unknown_{class_id}")
                        class_counts[class_name] += 1
        
        print(f"  Tổng files: {total_files}")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / sum(class_counts.values())) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

def create_data_yaml(data_root: Path):
    """Tạo file data.yaml cho YOLO training"""
    
    from scripts.data_adapters import CLASS_MAPPING
    
    yaml_content = f"""# YOLO dataset configuration for Traffic AI
path: {data_root.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path') 
test: images/test    # test images (relative to 'path')

# Classes
nc: {len(CLASS_MAPPING)}  # number of classes
names:
"""
    
    for class_id, class_name in sorted(CLASS_MAPPING.items()):
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_file = data_root / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"Đã tạo file cấu hình: {yaml_file}")
    return yaml_file

def main():
    """Main function"""
    print("=== HỢP NHẤT VÀ CHUẨN BỊ DỮ LIỆU ===")
    
    # Hợp nhất datasets
    data_root = merge_datasets()
    
    # Phân tích cân bằng classes
    analyze_class_balance(data_root)
    
    # Tạo file cấu hình YOLO
    yaml_file = create_data_yaml(data_root)
    
    print(f"\n=== HOÀN THÀNH ===")
    print(f"Dataset đã sẵn sàng tại: {data_root}")
    print(f"File cấu hình: {yaml_file}")
    print("Có thể bắt đầu training với: python -m ultralytics.yolo task=detect mode=train model=yolov12n.pt data=data/traffic/data.yaml")

if __name__ == "__main__":
    main()