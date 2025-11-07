"""
Script tạo balanced dataset cho 11 class giao thông
Sử dụng chiến lược:
1. Oversampling (augmentation) cho class thiếu số
2. Undersampling cho class quá nhiều
3. Đảm bảo mỗi class có ít nhất MIN_SAMPLES và tối đa MAX_SAMPLES
"""

import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

# Cấu hình
CLASS_NAMES = {
    0: "Vehicle", 1: "Bus", 2: "Bicycle", 3: "Person",
    4: "Engine", 5: "Truck", 6: "Tricycle", 7: "Obstacle",
    8: "Pothole", 9: "Traffic Light", 10: "Traffic Sign"
}

# Tham số cân bằng
TARGET_SAMPLES_PER_CLASS = 3000  # Mục tiêu số samples cho mỗi class
MIN_SAMPLES_PER_CLASS = 2000     # Tối thiểu
MAX_SAMPLES_PER_CLASS = 5000     # Tối đa

# Priority classes - cần nhiều hơn cho traffic detection
HIGH_PRIORITY = [0, 3, 9, 10]  # Vehicle, Person, Traffic Light, Traffic Sign
MEDIUM_PRIORITY = [1, 2, 4, 5]  # Bus, Bicycle, Engine, Truck  
LOW_PRIORITY = [6, 7, 8]        # Tricycle, Obstacle, Pothole

def get_class_distribution_per_image(label_file):
    """Trả về Counter của các class trong 1 image"""
    class_counter = Counter()
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1
    except:
        pass
    return class_counter

def build_class_to_images_map(labels_dir):
    """
    Tạo mapping: class_id -> list of (image_name, count_of_that_class_in_image)
    Sắp xếp theo count để ưu tiên images có nhiều objects của class đó
    """
    class_to_images = defaultdict(list)
    
    for label_file in Path(labels_dir).glob('*.txt'):
        img_name = label_file.stem
        class_dist = get_class_distribution_per_image(label_file)
        
        for class_id, count in class_dist.items():
            class_to_images[class_id].append((img_name, count))
    
    # Sắp xếp theo count giảm dần (ưu tiên images có nhiều objects của class này)
    for class_id in class_to_images:
        class_to_images[class_id].sort(key=lambda x: x[1], reverse=True)
    
    return class_to_images

def apply_simple_augmentation(img_path, output_path):
    """
    Copy image (augmentation sẽ được làm bằng YOLO built-in augmentation trong training)
    """
    try:
        shutil.copy2(img_path, output_path)
        return True
    except Exception as e:
        print(f"Error copying {img_path}: {e}")
        return False

def flip_yolo_labels(label_path, output_path):
    """Copy labels (augmentation sẽ được làm bằng YOLO built-in augmentation)"""
    try:
        shutil.copy2(label_path, output_path)
        return True
    except Exception as e:
        print(f"Error copying labels {label_path}: {e}")
        return False

def create_balanced_dataset(source_dir, output_dir, split='train'):
    """
    Tạo balanced dataset từ source
    
    Chiến lược:
    1. Đếm số lượng images có chứa mỗi class
    2. Với class thiếu: oversample bằng cách copy + augment
    3. Với class dư: undersample (random select)
    4. Đảm bảo mỗi class có MIN_SAMPLES <= n <= MAX_SAMPLES
    """
    
    print(f"\n{'='*80}")
    print(f"TẠO BALANCED DATASET CHO {split.upper()}")
    print(f"{'='*80}\n")
    
    source_labels = Path(source_dir) / 'labels' / split
    source_images = Path(source_dir) / 'images' / split
    
    output_labels = Path(output_dir) / 'labels' / split
    output_images = Path(output_dir) / 'images' / split
    
    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)
    
    # Build mapping
    print("📊 Phân tích class distribution...")
    class_to_images = build_class_to_images_map(source_labels)
    
    # In thống kê ban đầu
    print(f"\n📈 Số lượng images chứa mỗi class (trước cân bằng):")
    for class_id in sorted(CLASS_NAMES.keys()):
        count = len(class_to_images.get(class_id, []))
        print(f"   {class_id}: {CLASS_NAMES[class_id]:15s} - {count:5d} images")
    
    # Quyết định target cho mỗi class dựa trên priority
    class_targets = {}
    for class_id in CLASS_NAMES.keys():
        if class_id in HIGH_PRIORITY:
            class_targets[class_id] = int(TARGET_SAMPLES_PER_CLASS * 1.2)
        elif class_id in MEDIUM_PRIORITY:
            class_targets[class_id] = TARGET_SAMPLES_PER_CLASS
        else:  # LOW_PRIORITY
            class_targets[class_id] = int(TARGET_SAMPLES_PER_CLASS * 0.8)
    
    # Track images đã được chọn
    selected_images = set()
    augmented_count = defaultdict(int)
    
    # Xử lý từng class
    for class_id in sorted(CLASS_NAMES.keys()):
        images_with_class = class_to_images.get(class_id, [])
        current_count = len(images_with_class)
        target = class_targets[class_id]
        
        print(f"\n🎯 Class {class_id} ({CLASS_NAMES[class_id]}): {current_count} -> {target}")
        
        if current_count == 0:
            print(f"   ⚠️  Không có images nào chứa class này!")
            continue
        
        # Lấy tất cả images có chứa class này
        class_images = [img_name for img_name, _ in images_with_class]
        
        if current_count >= target:
            # Undersample: chọn ngẫu nhiên
            selected = random.sample(class_images, min(target, current_count))
            print(f"   📉 Undersampling: {len(selected)} images")
        else:
            # Oversample: copy tất cả + augment thêm
            selected = class_images.copy()
            need_more = target - current_count
            
            if need_more > 0:
                # Chọn ngẫu nhiên images để augment (có thể trùng)
                to_augment = random.choices(class_images, k=need_more)
                augmented_count[class_id] = len(to_augment)
                print(f"   📈 Oversampling: {current_count} original + {len(to_augment)} augmented")
                
                # Copy augmented images
                for idx, img_name in enumerate(to_augment):
                    src_img = source_images / f"{img_name}.jpg"
                    src_lbl = source_labels / f"{img_name}.txt"
                    
                    # Tên mới cho augmented sample
                    aug_name = f"{img_name}_aug{idx}"
                    dst_img = output_images / f"{aug_name}.jpg"
                    dst_lbl = output_labels / f"{aug_name}.txt"
                    
                    # Check file extension
                    if not src_img.exists():
                        src_img = source_images / f"{img_name}.png"
                    
                    if src_img.exists() and src_lbl.exists():
                        apply_simple_augmentation(src_img, dst_img)
                        flip_yolo_labels(src_lbl, dst_lbl)
        
        # Add vào selected_images
        selected_images.update(selected)
    
    # Copy tất cả selected images
    print(f"\n📋 Copying {len(selected_images)} selected images...")
    copied = 0
    for img_name in selected_images:
        src_img = source_images / f"{img_name}.jpg"
        src_lbl = source_labels / f"{img_name}.txt"
        
        if not src_img.exists():
            src_img = source_images / f"{img_name}.png"
        
        dst_img = output_images / f"{img_name}{src_img.suffix}"
        dst_lbl = output_labels / f"{img_name}.txt"
        
        if src_img.exists() and src_lbl.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            copied += 1
    
    print(f"   ✅ Copied {copied} images")
    
    # Kiểm tra kết quả
    print(f"\n{'='*80}")
    print("📊 KẾT QUẢ SAU CÂN BẰNG:")
    print(f"{'='*80}\n")
    
    final_class_dist = build_class_to_images_map(output_labels)
    total_final_images = len(list(output_images.glob('*.jpg'))) + len(list(output_images.glob('*.png')))
    
    print(f"Tổng số images: {total_final_images}")
    print(f"\nClass distribution:")
    for class_id in sorted(CLASS_NAMES.keys()):
        count = len(final_class_dist.get(class_id, []))
        target = class_targets[class_id]
        status = "✅" if count >= MIN_SAMPLES_PER_CLASS else "⚠️ "
        print(f"   {status} {class_id}: {CLASS_NAMES[class_id]:15s} - {count:5d} images (target: {target})")

def main():
    source_dir = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_balanced_11class_processed"
    output_dir = r"d:\DH_K47\nam_tu\HK1\Do_an_2\Traffic_AI_YOLOv12_jwt\datasets\traffic_ai_11class_rebalanced"
    
    # Set random seed
    random.seed(42)
    
    # Tạo balanced dataset cho train, val, test
    for split in ['train', 'val', 'test']:
        create_balanced_dataset(source_dir, output_dir, split)
    
    # Tạo data.yaml
    data_yaml_content = f"""# YOLOv12 11-Class Traffic Dataset (Rebalanced)
path: {output_dir}
train: images/train
val: images/val  
test: images/test

# Classes (11)
names:
  0: Vehicle
  1: Bus
  2: Bicycle
  3: Person
  4: Engine
  5: Truck
  6: Tricycle
  7: Obstacle
  8: Pothole
  9: Traffic Light
  10: Traffic Sign

nc: 11

# Dataset info
description: "Rebalanced 11-class traffic dataset for YOLOv12"
version: "1.0"
created: "2025-11-07"
"""
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    
    print(f"\n✅ data.yaml created at: {yaml_path}")
    print(f"\n{'='*80}")
    print("🎉 HOÀN THÀNH TẠO BALANCED DATASET!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
